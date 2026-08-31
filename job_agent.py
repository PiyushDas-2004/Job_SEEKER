#!/usr/bin/env python3
"""
job_agent.py - an AI job-search agent in pure Python.

No frameworks: the `anthropic` SDK, `aiohttp`, and the standard library.
Two Claude calls do the thinking; everything else is plain code.

Pipeline
    natural-language query
      -> Claude call #1: parse into structured search parameters
      -> concurrent fetch from Arbeitnow + RemoteOK (+ optional Adzuna/USAJobs)
      -> normalize to one schema
      -> fuzzy dedupe on (title, company)
      -> local relevance prefilter (keeps the Claude call affordable)
      -> Claude call #2: rank + one-line "why this matches" per job
      -> console table, optional CSV/JSON export, 1-hour result cache

Field names for every keyless source were confirmed by running `probe_apis.py`
against the live endpoints - do not "fix" them from memory. Re-run the probe if
results ever come back empty; job boards rename fields without notice.

Usage
    python job_agent.py "senior python backend, remote, 120k+, k8s"
    python job_agent.py --output jobs.csv --limit 15
    python job_agent.py "data engineer Berlin" --visa-sponsorship
    python job_agent.py "golang remote" --no-rank      # skip Claude entirely
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import html
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Sequence

# Corporate TLS-inspecting proxies present a private root CA that certifi does
# not ship, which makes every HTTPS call fail with CERTIFICATE_VERIFY_FAILED.
# `truststore` routes verification through the OS certificate store, where such
# a CA is installed. It patches the stdlib `ssl` module, so aiohttp and the
# anthropic SDK both pick it up. A no-op when not behind such a proxy.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # pragma: no cover - optional dependency
    pass

import aiohttp

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

MODEL = "claude-opus-5"

# Identify ourselves honestly. RemoteOK rejects requests without a real
# User-Agent, and both APIs ask to be treated politely.
USER_AGENT = "job-seeker-agent/1.0 (personal job search)"

ARBEITNOW_URL = "https://www.arbeitnow.com/api/job-board-api"
REMOTEOK_URL = "https://remoteok.com/api"
ADZUNA_URL = "https://api.adzuna.com/v1/api/jobs/{country}/search/1"
USAJOBS_URL = "https://data.usajobs.gov/api/search"

HTTP_TIMEOUT = 25          # seconds per request
HTTP_RETRIES = 2           # attempts per source before giving up on it
CACHE_TTL = 3600           # 1 hour, per the spec
CACHE_FILE = Path(".job_cache.json")
MAX_JOBS_TO_RANK = 40      # how many candidates Claude sees
DESC_SNIPPET_CHARS = 400   # description characters sent to Claude per job

log = logging.getLogger("job-agent")


# --------------------------------------------------------------------------- #
# .env loading (tiny, dependency-free)
# --------------------------------------------------------------------------- #

def load_dotenv(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines from .env into os.environ without overwriting."""
    p = Path(path)
    if not p.is_file():
        return
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        log.warning("could not read %s: %s", p, exc)
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


# --------------------------------------------------------------------------- #
# Normalized schema
# --------------------------------------------------------------------------- #

@dataclass
class Job:
    """One job, normalized across every source."""

    title: str
    company: str
    location: str
    salary: str | None
    posted_date: str | None          # ISO-8601 date, e.g. "2026-08-30"
    apply_url: str
    source: str
    remote: bool = False
    tags: list[str] = field(default_factory=list)
    description: str = ""            # plain text, HTML stripped

    # Populated later by the ranking stage.
    score: int | None = None
    why: str | None = None

    def snippet(self, n: int = DESC_SNIPPET_CHARS) -> str:
        text = " ".join(self.description.split())
        return text[:n] + ("..." if len(text) > n else "")


@dataclass
class SearchParams:
    """Structured form of the user's natural-language query (Claude call #1)."""

    role_keywords: list[str] = field(default_factory=list)
    seniority: str | None = None
    remote_preference: str = "any"       # remote_only | onsite_or_hybrid | any
    locations: list[str] = field(default_factory=list)
    salary_floor_usd: int | None = None
    tech_stack: list[str] = field(default_factory=list)
    visa_sponsorship_required: bool = False
    exclude_keywords: list[str] = field(default_factory=list)

    @property
    def all_keywords(self) -> list[str]:
        parts = list(self.role_keywords) + list(self.tech_stack)
        if self.seniority:
            parts.append(self.seniority)
        return [p.lower() for p in parts if p]


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# Seniority/format noise that shouldn't stop two postings from being recognized
# as the same role during deduplication.
_TITLE_NOISE = re.compile(
    r"\b(senior|sr|junior|jr|mid|midlevel|lead|staff|principal|"
    r"remote|hybrid|onsite|fulltime|full time|part time|parttime|"
    r"contract|permanent|m/w/d|w/m/d|d/f/m|m/f/d|f/m/d|all genders|gn)\b",
    re.IGNORECASE,
)


def strip_html(raw: str | None) -> str:
    """HTML -> readable plain text. Descriptions arrive as HTML from both APIs."""
    if not raw:
        return ""
    text = re.sub(r"<br\s*/?>|</p>|</li>|</h\d>", "\n", raw, flags=re.IGNORECASE)
    text = _TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", html.unescape(text)).strip()


def fix_mojibake(text: str) -> str:
    """
    RemoteOK serves some descriptions double-encoded (UTF-8 bytes read as
    latin-1), which shows up as 'a?<80><99>' garbage instead of a curly quote.
    Reverse it when the signature is present; leave clean text untouched.
    """
    if "Ã" in text or "â" in text:
        try:
            return text.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text
    return text


def normalize_title(title: str) -> str:
    """Canonical form of a job title, for dedupe comparisons only."""
    t = _TITLE_NOISE.sub(" ", title.lower())
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return _WS_RE.sub(" ", t).strip()


def normalize_company(company: str) -> str:
    c = company.lower()
    c = re.sub(r"\b(gmbh|ag|ltd|limited|inc|llc|bv|nv|se|kg|co|corp|"
               r"corporation|company|group|holding|holdings)\b", " ", c)
    c = re.sub(r"[^a-z0-9 ]+", " ", c)
    return _WS_RE.sub(" ", c).strip()


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def iso_from_epoch(epoch: Any) -> str | None:
    """Arbeitnow's created_at and RemoteOK's epoch are unix seconds."""
    try:
        seconds = int(epoch)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc).date().isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def iso_from_datestring(value: Any) -> str | None:
    """Best-effort ISO date from assorted API date strings."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        pass
    m = re.match(r"(\d{4}-\d{2}-\d{2})", value)
    return m.group(1) if m else None


def format_salary(minimum: Any, maximum: Any, currency: str = "$") -> str | None:
    """Render a salary range, treating 0/None/absent as 'not disclosed'."""
    def clean(v: Any) -> int | None:
        try:
            n = int(float(v))
        except (TypeError, ValueError):
            return None
        return n if n > 0 else None

    lo, hi = clean(minimum), clean(maximum)
    if lo and hi and lo != hi:
        return f"{currency}{lo:,} - {currency}{hi:,}"
    if lo or hi:
        return f"{currency}{(lo or hi):,}"
    return None


def parse_salary_floor(salary: str | None) -> int | None:
    """Pull the lowest number out of a formatted salary string, for filtering."""
    if not salary:
        return None
    numbers = [int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", salary)]
    return min(numbers) if numbers else None


# --------------------------------------------------------------------------- #
# Fetching - one coroutine per source, all failures contained
# --------------------------------------------------------------------------- #

class SourceError(Exception):
    """A source could not be fetched; the run continues without it."""


async def _get_json(
    session: aiohttp.ClientSession,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    label: str,
) -> Any:
    """
    GET + JSON decode with bounded retries.

    Retries 429 and 5xx (honouring a sane Retry-After) and network errors.
    Raises SourceError on definitive failure so the caller can skip the source
    instead of crashing the run.
    """
    last = "unknown error"
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            async with session.get(url, params=params, headers=headers) as resp:
                if resp.status == 429 or resp.status >= 500:
                    retry_after = (resp.headers.get("Retry-After") or "").strip()
                    delay = (min(float(retry_after), 10.0)
                             if retry_after.isdigit() else 1.5 * attempt)
                    last = f"HTTP {resp.status}"
                    log.warning("%s: %s (attempt %d/%d), retrying in %.1fs",
                                label, last, attempt, HTTP_RETRIES, delay)
                    await asyncio.sleep(delay)
                    continue
                if resp.status != 200:
                    body = (await resp.text())[:200]
                    raise SourceError(f"{label}: HTTP {resp.status}: {body}")
                # Some boards return JSON under text/plain; don't be strict.
                return await resp.json(content_type=None)
        except asyncio.TimeoutError:
            last = f"timeout after {HTTP_TIMEOUT}s"
            log.warning("%s: %s (attempt %d/%d)", label, last, attempt, HTTP_RETRIES)
        except aiohttp.ClientError as exc:
            last = f"{type(exc).__name__}: {exc}"
            log.warning("%s: %s (attempt %d/%d)", label, last, attempt, HTTP_RETRIES)
        except ValueError as exc:  # includes json.JSONDecodeError
            raise SourceError(f"{label}: response was not valid JSON: {exc}") from exc
    raise SourceError(f"{label}: giving up after {HTTP_RETRIES} attempts ({last})")


async def fetch_arbeitnow(
    session: aiohttp.ClientSession,
    *,
    pages: int = 2,
    visa_sponsorship: bool = False,
) -> list[Job]:
    """
    Arbeitnow job board API - no key required.

    Verified response shape (probe_apis.py, 2026-08-31):
        {"data": [{slug, company_name, title, description, remote, url,
                   tags[], job_types[], location, created_at}],
         "links": {first, last, prev, next}, "meta": {current_page, per_page,...}}
    ~175 jobs per page. There is no salary field and no server-side keyword
    search, so relevance filtering happens locally.
    """
    jobs: list[Job] = []
    page = 1
    for page in range(1, max(1, pages) + 1):
        params: dict[str, Any] = {"page": page}
        if visa_sponsorship:
            params["visa_sponsorship"] = "true"
        payload = await _get_json(
            session, ARBEITNOW_URL, params=params,
            headers={"Accept": "application/json"},
            label=f"arbeitnow[page={page}]",
        )
        data = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(data, list):
            raise SourceError("arbeitnow: no 'data' array in response")
        for item in data:
            if not isinstance(item, dict):
                continue
            jobs.append(Job(
                title=(item.get("title") or "").strip(),
                company=(item.get("company_name") or "").strip() or "Unknown",
                location=(item.get("location") or "").strip() or "Not specified",
                salary=None,                      # Arbeitnow exposes no salary
                posted_date=iso_from_epoch(item.get("created_at")),
                apply_url=item.get("url") or "",
                source="arbeitnow",
                remote=bool(item.get("remote")),
                tags=[str(t) for t in (item.get("tags") or [])][:8],
                description=strip_html(item.get("description")),
            ))
        # Stop early when the feed says there is no next page.
        if isinstance(payload, dict) and not (payload.get("links") or {}).get("next"):
            break
    log.info("arbeitnow: %d jobs from %d page(s)", len(jobs), page)
    return jobs


async def fetch_remoteok(session: aiohttp.ClientSession) -> list[Job]:
    """
    RemoteOK API - no key required, but a real User-Agent is mandatory.

    Verified response shape (probe_apis.py, 2026-08-31): a JSON *array* whose
    element 0 is a legal/ToS notice, not a job. Job objects:
        {slug, id, epoch, date, company, company_logo, position, tags[],
         description, location, apply_url, salary_min, salary_max, logo, url}
    Salary is populated on only ~4% of postings (0 means "not disclosed").
    Everything is remote by definition, and the feed is a single page of the
    ~100 most recent jobs - there is no pagination or keyword parameter.
    """
    payload = await _get_json(
        session, REMOTEOK_URL, headers={"Accept": "application/json"},
        label="remoteok",
    )
    if not isinstance(payload, list) or not payload:
        raise SourceError("remoteok: expected a non-empty JSON array")

    # Element 0 is the legal notice. Skip it by shape rather than by index:
    # keep only entries that actually look like a job.
    entries = [e for e in payload if isinstance(e, dict) and e.get("position")]

    jobs = [
        Job(
            title=(item.get("position") or "").strip(),
            company=(item.get("company") or "").strip() or "Unknown",
            location=(item.get("location") or "").strip().rstrip(", ") or "Remote",
            salary=format_salary(item.get("salary_min"), item.get("salary_max")),
            posted_date=(iso_from_datestring(item.get("date"))
                         or iso_from_epoch(item.get("epoch"))),
            apply_url=item.get("apply_url") or item.get("url") or "",
            source="remoteok",
            remote=True,
            tags=[str(t) for t in (item.get("tags") or [])][:8],
            description=fix_mojibake(strip_html(item.get("description"))),
        )
        for item in entries
    ]
    log.info("remoteok: %d jobs", len(jobs))
    return jobs


async def fetch_adzuna(
    session: aiohttp.ClientSession,
    params: SearchParams,
    *,
    country: str = "gb",
    results: int = 50,
) -> list[Job]:
    """
    Adzuna (optional) - needs a free app_id/app_key from developer.adzuna.com.

    Request and response shapes follow Adzuna's own Swagger spec
    (https://api.adzuna.com/v1/api-docs/adzuna):
        GET /v1/api/jobs/{country}/search/{page}
        required: country, app_id, app_key, page
        response: {count, mean, results[]}
        results[] item: id, title, description (truncated to 500 chars),
            full_description?, created (ISO 8601), redirect_url, adref,
            company{display_name, canonical_name}, location{display_name,
            area[]}, category{tag, label}, salary_min, salary_max,
            salary_is_predicted, contract_type, contract_time, latitude,
            longitude

    Two Adzuna-specific honesty rules, both handled below:
      * `salary_is_predicted` means the figure is Adzuna's own estimate, not
        something the employer disclosed - it is labelled as such.
      * salaries are in the country's LOCAL CURRENCY, so a USD salary floor is
        passed through unconverted and can only be treated as approximate.

    Field names come from the published spec, but this normalizer has not been
    run against live Adzuna data (no keys at build time), so every read is
    defensive. Run `python probe_apis.py --adzuna` once you have keys.
    """
    app_id, app_key = os.getenv("ADZUNA_APP_ID"), os.getenv("ADZUNA_APP_KEY")
    if not (app_id and app_key):
        raise SourceError("adzuna: ADZUNA_APP_ID / ADZUNA_APP_KEY not set")

    query: dict[str, Any] = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": max(1, min(results, 50)),
        "content-type": "application/json",
        "max_days_old": 30,
    }
    # Unlike the keyless sources, Adzuna has real server-side search - use it.
    keywords = params.role_keywords or params.all_keywords
    if keywords:
        query["what"] = " ".join(keywords[:4])
    if params.tech_stack:
        query["what_or"] = " ".join(params.tech_stack[:6])
    if params.exclude_keywords:
        query["what_exclude"] = " ".join(params.exclude_keywords[:6])
    if params.locations:
        query["where"] = params.locations[0]
    if params.salary_floor_usd:
        query["salary_min"] = params.salary_floor_usd
        # Without this, every posting that doesn't state a salary is discarded
        # server-side - which is most of them.
        query["salary_include_unknown"] = "1"

    payload = await _get_json(
        session, ADZUNA_URL.format(country=country), params=query,
        headers={"Accept": "application/json"}, label="adzuna",
    )
    if not isinstance(payload, dict):
        raise SourceError("adzuna: expected a JSON object with a 'results' array")
    total = payload.get("count")
    results_list = payload.get("results") or []

    jobs = []
    for item in results_list:
        if not isinstance(item, dict):
            continue
        category = (item.get("category") or {}).get("label")
        salary = format_salary(item.get("salary_min"), item.get("salary_max"),
                               currency="")   # local currency, not USD
        # The spec types this as boolean, but the JSON has been seen carrying
        # "0"/"1" strings; accept either.
        if salary and str(item.get("salary_is_predicted", "")).strip().lower() in (
                "1", "true"):
            salary = f"~{salary} (Adzuna estimate, not employer-stated)"

        title = (item.get("title") or "").strip()
        # Adzuna has no remote flag. Infer it only from an explicit statement in
        # the title, so a --remote query isn't silently emptied of Adzuna
        # results, and never from vague description text.
        blob = f"{title} {(item.get('location') or {}).get('display_name') or ''}"
        remote = bool(re.search(r"\b(remote|work from home|wfh|home[- ]based)\b",
                                blob, re.IGNORECASE))

        contract = [v for v in (item.get("contract_type"),
                                item.get("contract_time")) if v]
        jobs.append(Job(
            title=title,
            company=((item.get("company") or {}).get("display_name") or "Unknown"),
            location=((item.get("location") or {}).get("display_name")
                      or "Not specified"),
            salary=salary,
            posted_date=iso_from_datestring(item.get("created")),
            apply_url=item.get("redirect_url") or "",
            source="adzuna",
            remote=remote,
            tags=([category] if category else []) + [str(c) for c in contract],
            # `description` is capped at 500 chars upstream; prefer the full
            # text when Adzuna includes it.
            description=strip_html(item.get("full_description")
                                   or item.get("description")),
        ))
    log.info("adzuna[%s]: %d jobs returned of %s total matches",
             country, len(jobs), total if total is not None else "?")
    return jobs


async def fetch_usajobs(
    session: aiohttp.ClientSession,
    params: SearchParams,
    *,
    results: int = 50,
) -> list[Job]:
    """
    USAJobs (optional) - US federal jobs only. Needs a free API key from
    developer.usajobs.gov plus the registered email address as User-Agent.

    NOTE: shape unverified against live data (no key at build time); reads are
    defensive. Confirm with `python probe_apis.py --usajobs`.
    """
    key, email = os.getenv("USAJOBS_API_KEY"), os.getenv("USAJOBS_EMAIL")
    if not (key and email):
        raise SourceError("usajobs: USAJOBS_API_KEY / USAJOBS_EMAIL not set")

    query: dict[str, Any] = {"ResultsPerPage": min(results, 500)}
    keywords = params.role_keywords or params.all_keywords
    if keywords:
        query["Keyword"] = " ".join(keywords[:4])
    if params.locations:
        query["LocationName"] = params.locations[0]
    if params.salary_floor_usd:
        query["RemunerationMinimumAmount"] = params.salary_floor_usd
    if params.remote_preference == "remote_only":
        query["RemoteIndicator"] = "True"

    payload = await _get_json(
        session, USAJOBS_URL, params=query,
        headers={"Host": "data.usajobs.gov", "User-Agent": email,
                 "Authorization-Key": key, "Accept": "application/json"},
        label="usajobs",
    )
    items = (((payload or {}).get("SearchResult") or {})
             .get("SearchResultItems") or [])
    jobs = []
    for entry in items:
        d = (entry or {}).get("MatchedObjectDescriptor") or {}
        pay = (d.get("PositionRemuneration") or [{}])[0] or {}
        details = (d.get("UserArea") or {}).get("Details") or {}
        jobs.append(Job(
            title=(d.get("PositionTitle") or "").strip(),
            company=(d.get("OrganizationName") or "US Federal Government"),
            location=(d.get("PositionLocationDisplay") or "United States"),
            salary=format_salary(pay.get("MinimumRange"), pay.get("MaximumRange")),
            posted_date=iso_from_datestring(d.get("PublicationStartDate")),
            apply_url=d.get("PositionURI") or "",
            source="usajobs",
            remote=False,
            tags=[str(c.get("Name")) for c in (d.get("JobCategory") or [])][:5],
            description=strip_html(details.get("JobSummary")
                                   or d.get("QualificationSummary")),
        ))
    log.info("usajobs: %d jobs", len(jobs))
    return jobs


async def gather_jobs(params: SearchParams, args: argparse.Namespace) -> list[Job]:
    """Query every enabled source concurrently; skip the ones that fail."""
    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
    headers = {"User-Agent": USER_AGENT}
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        tasks: dict[str, Any] = {}
        if "arbeitnow" in args.sources:
            tasks["arbeitnow"] = fetch_arbeitnow(
                session, pages=args.pages,
                visa_sponsorship=(args.visa_sponsorship
                                  or params.visa_sponsorship_required),
            )
        if "remoteok" in args.sources:
            tasks["remoteok"] = fetch_remoteok(session)
        if args.use_adzuna:
            tasks["adzuna"] = fetch_adzuna(session, params,
                                           country=args.adzuna_country)
        if args.use_usajobs:
            tasks["usajobs"] = fetch_usajobs(session, params)

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    jobs: list[Job] = []
    for name, result in zip(tasks.keys(), results):
        if isinstance(result, BaseException):
            # A source being down or rate-limited must not end the run.
            log.warning("SKIPPING source '%s': %s", name, result)
            continue
        jobs.extend(result)
    if not jobs:
        log.error("every source failed or returned nothing")
    return jobs


# --------------------------------------------------------------------------- #
# Deduplication
# --------------------------------------------------------------------------- #

def deduplicate(jobs: Sequence[Job], threshold: float = 0.86) -> list[Job]:
    """
    Drop near-duplicate postings by (title, company) similarity.

    Same company (fuzzy, legal suffixes ignored) + similar title = duplicate.
    Bucketing by the company's first character keeps this well clear of a full
    O(n^2) scan at real feed sizes. The first posting wins, except that a later
    one is preferred when it carries salary information the first lacked.
    """
    buckets: dict[str, list[tuple[str, str, int]]] = {}
    kept: list[Job] = []
    dropped = 0

    for job in jobs:
        if not job.title or not job.apply_url:
            dropped += 1
            continue
        ncomp, ntitle = normalize_company(job.company), normalize_title(job.title)
        bucket = buckets.setdefault(ncomp[:1], [])

        match_index: int | None = None
        for other_comp, other_title, index in bucket:
            if similar(ncomp, other_comp) < 0.90:
                continue
            if similar(ntitle, other_title) >= threshold:
                match_index = index
                break

        if match_index is None:
            bucket.append((ncomp, ntitle, len(kept)))
            kept.append(job)
        else:
            dropped += 1
            incumbent = kept[match_index]
            if job.salary and not incumbent.salary:
                job.tags = list(dict.fromkeys(job.tags + incumbent.tags))[:8]
                kept[match_index] = job

    log.info("dedupe: %d unique, %d duplicate/invalid removed", len(kept), dropped)
    return kept


# --------------------------------------------------------------------------- #
# Local prefilter - cheap relevance scoring before we pay for tokens
# --------------------------------------------------------------------------- #

def keyword_hits(job: Job, params: SearchParams) -> int:
    """How many of the query's keywords appear in the title or tags."""
    blob = f"{job.title} {' '.join(job.tags)}".lower()
    return sum(1 for kw in set(params.all_keywords) if kw and kw in blob)


def local_score(job: Job, params: SearchParams) -> float:
    """Keyword/constraint score used to pick which jobs Claude sees."""
    title = job.title.lower()
    rest = f"{' '.join(job.tags)} {job.description[:1500]}".lower()

    score = 0.0
    for kw in set(params.all_keywords):
        if not kw:
            continue
        if kw in title:
            score += 3.0
        elif kw in rest:
            score += 1.0

    if params.remote_preference == "remote_only" and job.remote:
        score += 2.0
    if params.locations:
        loc = job.location.lower()
        if any(l.lower() in loc for l in params.locations):
            score += 2.0

    floor = params.salary_floor_usd
    if floor:
        job_floor = parse_salary_floor(job.salary)
        if job_floor:
            score += 1.5 if job_floor >= floor else -1.5

    # Recency nudge, so a stale posting doesn't outrank an equally good fresh one.
    if job.posted_date:
        try:
            age_days = (datetime.now(timezone.utc).date()
                        - datetime.fromisoformat(job.posted_date).date()).days
            score += max(0.0, 1.0 - age_days / 30.0)
        except ValueError:
            pass
    return score


def prefilter(jobs: Sequence[Job], params: SearchParams,
              keep: int = MAX_JOBS_TO_RANK) -> list[Job]:
    """Apply hard exclusions and constraints, then keep the top `keep`."""
    excluded = [e.lower() for e in params.exclude_keywords if e]
    candidates = []
    dropped_remote_but_relevant = 0
    for job in jobs:
        blob = f"{job.title} {job.company}".lower()
        if any(e in blob for e in excluded):
            continue
        if params.remote_preference == "remote_only" and not job.remote:
            if keyword_hits(job, params):
                dropped_remote_but_relevant += 1
            continue
        candidates.append(job)

    # These feeds are small and ordered by recency, so a niche query can easily
    # match nothing. Say so plainly instead of presenting noise as results.
    if dropped_remote_but_relevant:
        log.warning("%d posting(s) matched your keywords but are not remote - "
                    "drop the remote requirement from the query to see them",
                    dropped_remote_but_relevant)
    if candidates and not any(keyword_hits(j, params) for j in candidates):
        log.warning("no posting title matched your keywords; showing the closest "
                    "available (both feeds are small and recency-ordered)")

    scored = sorted(candidates, key=lambda j: local_score(j, params), reverse=True)
    # If nothing scored at all, show the freshest postings rather than an
    # arbitrary slice of an unsorted list.
    if scored and local_score(scored[0], params) <= 0:
        scored = sorted(candidates, key=lambda j: j.posted_date or "", reverse=True)
    log.info("prefilter: %d candidates -> top %d for ranking",
             len(candidates), min(keep, len(scored)))
    return scored[:keep]


# --------------------------------------------------------------------------- #
# Claude - call #1 (parse the query) and call #2 (rank + explain)
# --------------------------------------------------------------------------- #

PARAMS_SCHEMA = {
    "type": "object",
    "properties": {
        "role_keywords": {
            "type": "array", "items": {"type": "string"},
            "description": "Job-title keywords, e.g. ['backend engineer', "
                           "'python developer'].",
        },
        "seniority": {
            "type": ["string", "null"],
            "description": "One of intern, junior, mid, senior, staff, principal, "
                           "lead, manager - or null if unspecified.",
        },
        "remote_preference": {
            "type": "string",
            "enum": ["remote_only", "onsite_or_hybrid", "any"],
        },
        "locations": {
            "type": "array", "items": {"type": "string"},
            "description": "Cities, regions or countries mentioned. Empty if none.",
        },
        "salary_floor_usd": {
            "type": ["integer", "null"],
            "description": "Minimum annual salary in USD ('120k' -> 120000). "
                           "Null if unspecified.",
        },
        "tech_stack": {
            "type": "array", "items": {"type": "string"},
            "description": "Concrete technologies: languages, frameworks, tools.",
        },
        "visa_sponsorship_required": {"type": "boolean"},
        "exclude_keywords": {
            "type": "array", "items": {"type": "string"},
            "description": "Things the user explicitly does NOT want.",
        },
    },
    "required": ["role_keywords", "seniority", "remote_preference", "locations",
                 "salary_floor_usd", "tech_stack", "visa_sponsorship_required",
                 "exclude_keywords"],
    "additionalProperties": False,
}

RANKING_SCHEMA = {
    "type": "object",
    "properties": {
        "rankings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer",
                           "description": "The job's id from the input list."},
                    "score": {"type": "integer",
                              "description": "Match quality, 0-100."},
                    "why": {"type": "string",
                            "description": "One concise line on why this matches "
                                           "(or doesn't). At most 140 characters."},
                },
                "required": ["id", "score", "why"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["rankings"],
    "additionalProperties": False,
}

PARSE_SYSTEM = (
    "You convert a job seeker's free-text search into structured search "
    "parameters. Extract only what the text actually says or clearly implies; "
    "do not invent constraints. Keywords should be short and searchable."
)

RANK_SYSTEM = (
    "You are a pragmatic technical recruiter. You are given a job seeker's "
    "original request and a numbered list of real, current job postings. Score "
    "each posting 0-100 on how well it fits the request, weighing role match "
    "first, then seniority, location/remote fit, tech stack, and salary. Be "
    "honest and discriminating: irrelevant postings must score below 30, and do "
    "not inflate scores to be encouraging. For every posting write one concrete "
    "line explaining the match, citing specifics from the posting rather than "
    "generic praise. Return every posting exactly once."
)


class RefusalError(RuntimeError):
    """Claude declined the request (stop_reason == 'refusal')."""


def _extract_json(response: Any) -> dict[str, Any]:
    """
    Pull the JSON object out of a structured-output response.

    `output_config.format` guarantees a text block containing valid JSON, but
    stop_reason still has to be checked first: a refusal or a max_tokens
    truncation comes back as HTTP 200 with unusable content.
    """
    if response.stop_reason == "refusal":
        raise RefusalError("Claude declined this request "
                           f"({getattr(response, 'stop_details', None)})")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("response hit max_tokens; the JSON is truncated")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if not text:
        raise RuntimeError("no text block in response "
                           f"(stop_reason={response.stop_reason})")
    return json.loads(text)


def get_client() -> Any | None:
    """Build the Anthropic client, or return None if it isn't usable."""
    try:
        import anthropic
    except ImportError:
        log.warning("the `anthropic` package is not installed "
                    "(pip install -r requirements.txt); using local scoring")
        return None
    try:
        # Resolves ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / an `ant auth login`
        # profile. Construction succeeds even with no credential at all - the
        # failure would only surface on the first request - so check up front and
        # degrade cleanly instead of emitting a confusing auth traceback later.
        client = anthropic.Anthropic()
    except Exception as exc:
        log.warning("could not create the Anthropic client (%s); "
                    "using local scoring", exc)
        return None
    if not (client.api_key or client.auth_token or client.credentials):
        log.warning("no Anthropic credentials found (set ANTHROPIC_API_KEY in .env "
                    "or run `ant auth login`); falling back to local keyword "
                    "scoring - run with --no-rank to silence this")
        return None
    return client


def parse_query_with_claude(client: Any, query: str) -> SearchParams:
    """Claude call #1: natural language -> SearchParams."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=PARSE_SYSTEM,
        messages=[{"role": "user", "content": f"Job search request:\n{query}"}],
        output_config={
            "format": {"type": "json_schema", "schema": PARAMS_SCHEMA},
            "effort": "medium",   # extraction; no need for deep reasoning
        },
    )
    data = _extract_json(response)
    params = SearchParams(
        role_keywords=[str(k) for k in data.get("role_keywords") or []],
        seniority=data.get("seniority"),
        remote_preference=data.get("remote_preference") or "any",
        locations=[str(l) for l in data.get("locations") or []],
        salary_floor_usd=data.get("salary_floor_usd"),
        tech_stack=[str(t) for t in data.get("tech_stack") or []],
        visa_sponsorship_required=bool(data.get("visa_sponsorship_required")),
        exclude_keywords=[str(e) for e in data.get("exclude_keywords") or []],
    )
    log.info("parsed query: %s", json.dumps(asdict(params), ensure_ascii=False))
    return params


def rank_with_claude(client: Any, query: str, jobs: Sequence[Job]) -> list[Job]:
    """
    Claude call #2: score the candidate jobs and explain each match.

    Jobs go out as a compact numbered digest; the response is joined back on the
    integer id. Anything Claude omits stays in the list unranked, so no job
    disappears silently.
    """
    digest = [
        {
            "id": i,
            "title": job.title,
            "company": job.company,
            "location": job.location,
            "remote": job.remote,
            "salary": job.salary or "not disclosed",
            "posted": job.posted_date or "unknown",
            "tags": job.tags,
            "source": job.source,
            "description_snippet": job.snippet(),
        }
        for i, job in enumerate(jobs)
    ]
    prompt = (
        f"Job seeker's request:\n{query}\n\n"
        f"Postings to score ({len(digest)}):\n"
        f"{json.dumps(digest, ensure_ascii=False, indent=1)}"
    )

    # Streaming: one explanation per job across a large batch is a long
    # generation, and streaming keeps it clear of HTTP request timeouts.
    with client.messages.stream(
        model=MODEL,
        max_tokens=16000,
        system=RANK_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": RANKING_SCHEMA}},
    ) as stream:
        response = stream.get_final_message()

    rankings = _extract_json(response).get("rankings") or []
    seen: set[int] = set()
    for entry in rankings:
        try:
            index = int(entry["id"])
            job = jobs[index]
        except (KeyError, ValueError, TypeError, IndexError):
            log.warning("ranking entry with a bad id ignored: %r", entry)
            continue
        if index in seen:
            continue
        seen.add(index)
        try:
            job.score = max(0, min(100, int(entry.get("score", 0))))
        except (TypeError, ValueError):
            job.score = 0
        job.why = " ".join(str(entry.get("why", "")).split())[:180]

    if missing := len(jobs) - len(seen):
        log.warning("Claude did not rank %d job(s); keeping them unranked", missing)

    ranked = sorted(jobs, key=lambda j: (j.score if j.score is not None else -1),
                    reverse=True)
    log.info("ranked %d/%d jobs via Claude", len(seen), len(jobs))
    return ranked


# --------------------------------------------------------------------------- #
# Fallbacks used when Claude is unavailable (--no-rank, no key, API failure)
# --------------------------------------------------------------------------- #

def parse_query_locally(query: str) -> SearchParams:
    """Keyword-only query parsing. Deliberately dumb: Claude does this properly."""
    q = query.lower()
    # Constraint words are captured by dedicated fields below, so they must not
    # also become role keywords - otherwise every "remote" posting scores a hit.
    stop = {"a", "an", "the", "and", "or", "for", "with", "in", "at", "to", "of",
            "on", "job", "jobs", "role", "roles", "position", "positions", "i",
            "want", "looking", "please", "find", "me", "my", "any", "that", "is",
            "remote", "hybrid", "onsite", "visa", "sponsorship", "salary", "per",
            "year", "annum", "plus", "minimum", "min"}
    words = [w for w in re.findall(r"[a-z][a-z0-9+#.]{1,}", q) if w not in stop]

    seniority = next((s for s in ("intern", "junior", "mid", "senior", "staff",
                                  "principal", "lead", "manager") if s in q), None)
    floor = None
    if m := re.search(r"(\d{2,3})\s*k\b", q):
        floor = int(m.group(1)) * 1000
    elif m := re.search(r"\$?\s*(\d{5,7})", q):
        floor = int(m.group(1))
    return SearchParams(
        role_keywords=words[:8],
        seniority=seniority,
        remote_preference="remote_only" if "remote" in q else "any",
        salary_floor_usd=floor,
        visa_sponsorship_required=("visa" in q or "sponsorship" in q),
    )


def rank_locally(jobs: Sequence[Job], params: SearchParams) -> list[Job]:
    """Score with the local heuristic and synthesize a terse explanation."""
    for job in jobs:
        job.score = int(max(0, min(100, round(local_score(job, params) * 8))))
        hits = [kw for kw in set(params.all_keywords)
                if kw and kw in f"{job.title} {' '.join(job.tags)}".lower()]
        bits = [f"matches {', '.join(sorted(hits)[:3])}"] if hits else \
               ["no direct keyword match in title/tags"]
        if job.remote:
            bits.append("remote")
        if job.salary:
            bits.append(job.salary)
        job.why = "(local scoring, no Claude) " + "; ".join(bits)
    return sorted(jobs, key=lambda j: j.score or 0, reverse=True)


# --------------------------------------------------------------------------- #
# Cache - identical queries are served from disk for one hour
# --------------------------------------------------------------------------- #

def cache_key(query: str, args: argparse.Namespace) -> str:
    payload = json.dumps({
        "query": " ".join(query.strip().lower().split()),
        "sources": sorted(args.sources),
        "pages": args.pages,
        "visa": args.visa_sponsorship,
        "adzuna": args.use_adzuna,
        "adzuna_country": args.adzuna_country,
        "usajobs": args.use_usajobs,
        "rank": not args.no_rank,
        "model": MODEL,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def cache_load(path: Path, key: str) -> list[Job] | None:
    if not path.is_file():
        return None
    try:
        store = json.loads(path.read_text(encoding="utf-8"))
        entry = store.get(key)
        if not entry or time.time() - entry["ts"] > CACHE_TTL:
            return None
        log.info("cache hit (%.0f min old) - skipping network and Claude calls",
                 (time.time() - entry["ts"]) / 60)
        return [Job(**j) for j in entry["jobs"]]
    except (ValueError, OSError, KeyError, TypeError) as exc:
        log.warning("cache unreadable, ignoring it: %s", exc)
        return None


def cache_store(path: Path, key: str, jobs: Sequence[Job]) -> None:
    try:
        store: dict[str, Any] = {}
        if path.is_file():
            try:
                store = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                store = {}
        now = time.time()
        # Drop expired entries so the file can't grow without bound.
        store = {k: v for k, v in store.items()
                 if isinstance(v, dict) and now - v.get("ts", 0) <= CACHE_TTL}
        store[key] = {"ts": now, "jobs": [asdict(j) for j in jobs]}
        path.write_text(json.dumps(store), encoding="utf-8")
    except OSError as exc:
        log.warning("could not write cache: %s", exc)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def print_jobs(jobs: Sequence[Job], query: str) -> None:
    # Job titles are full of non-ASCII (German umlauts, curly quotes). On a
    # legacy Windows code page an unencodable character would raise
    # UnicodeEncodeError mid-listing; degrade to '?' instead of crashing.
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover - non-standard stdout
        pass

    width = 78
    print()
    print("=" * width)
    print(f" {len(jobs)} match(es) for: {query}")
    print("=" * width)
    if not jobs:
        print("\nNothing found. Try broader keywords, more --pages, or drop the "
              "remote/salary constraints.\n")
        return

    for rank, job in enumerate(jobs, 1):
        score = f"{job.score:3d}" if job.score is not None else " --"
        print(f"\n{rank:2d}. [{score}] {job.title}")
        print(f"    {job.company}  |  {job.location}"
              f"{'  |  remote' if job.remote else ''}")
        meta = [f"source: {job.source}"]
        if job.salary:
            meta.append(f"salary: {job.salary}")
        if job.posted_date:
            meta.append(f"posted: {job.posted_date}")
        print(f"    {'  |  '.join(meta)}")
        if job.why:
            print(f"    why: {job.why}")
        print(f"    {job.apply_url}")
    print("\n" + "=" * width)


def write_output(jobs: Sequence[Job], target: str) -> None:
    path = Path(target)
    rows = [asdict(j) for j in jobs]
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    else:
        if path.suffix.lower() != ".csv":
            log.warning("unrecognized extension '%s'; writing CSV", path.suffix)
        fields = ["score", "title", "company", "location", "remote", "salary",
                  "posted_date", "source", "why", "apply_url", "tags"]
        # utf-8-sig so Excel opens non-ASCII company names correctly.
        with path.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                row["tags"] = ", ".join(row.get("tags") or [])
                writer.writerow(row)
    print(f"wrote {len(jobs)} job(s) to {path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="job_agent.py",
        description="AI job-search agent over free, public job APIs.",
    )
    p.add_argument("query", nargs="*",
                   help='natural-language search, e.g. "senior python, remote, 130k+"')
    p.add_argument("--limit", type=int, default=10,
                   help="how many jobs to display (default: 10)")
    p.add_argument("--pages", type=int, default=2,
                   help="Arbeitnow pages to fetch, ~175 jobs each (default: 2)")
    p.add_argument("--sources", default="arbeitnow,remoteok",
                   help="comma-separated keyless sources (default: both)")
    p.add_argument("--visa-sponsorship", action="store_true",
                   help="Arbeitnow: only jobs offering visa sponsorship")
    p.add_argument("--use-adzuna", action="store_true",
                   help="also query Adzuna (needs ADZUNA_APP_ID/ADZUNA_APP_KEY)")
    p.add_argument("--adzuna-country", default="gb",
                   help="Adzuna country code (default: gb)")
    p.add_argument("--use-usajobs", action="store_true",
                   help="also query USAJobs (needs USAJOBS_API_KEY/USAJOBS_EMAIL)")
    p.add_argument("--output", metavar="FILE",
                   help="also write results to jobs.csv or jobs.json")
    p.add_argument("--no-rank", action="store_true",
                   help="skip both Claude calls; use local keyword scoring only")
    p.add_argument("--no-cache", action="store_true", help="bypass the result cache")
    p.add_argument("--clear-cache", action="store_true",
                   help="delete the cache file and exit")
    p.add_argument("--cache-file", default=str(CACHE_FILE),
                   help=f"cache location (default: {CACHE_FILE})")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(message)s",
        stream=sys.stderr,
    )
    load_dotenv()

    cache_path = Path(args.cache_file)
    if args.clear_cache:
        cache_path.unlink(missing_ok=True)
        print(f"cache cleared: {cache_path}")
        return 0

    args.sources = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
    if unknown := set(args.sources) - {"arbeitnow", "remoteok"}:
        log.error("unknown source(s): %s (valid: arbeitnow, remoteok)",
                  ", ".join(sorted(unknown)))
        return 2
    if not args.sources and not (args.use_adzuna or args.use_usajobs):
        log.error("no sources enabled")
        return 2

    query = " ".join(args.query).strip()
    if not query:
        try:
            query = input("What kind of job are you looking for? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 130
    if not query:
        log.error("empty query")
        return 2

    key = cache_key(query, args)
    if not args.no_cache:
        cached = cache_load(cache_path, key)
        if cached is not None:
            print_jobs(cached[:args.limit], query)
            if args.output:
                write_output(cached[:args.limit], args.output)
            return 0

    # --- Claude call #1: understand the query --------------------------------
    client = None if args.no_rank else get_client()
    params: SearchParams
    if client:
        try:
            params = parse_query_with_claude(client, query)
        except Exception as exc:
            log.warning("query parsing via Claude failed (%s: %s); falling back "
                        "to local keyword parsing", type(exc).__name__, exc)
            params = parse_query_locally(query)
    else:
        params = parse_query_locally(query)

    # --- Fetch, normalize, dedupe, prefilter --------------------------------
    try:
        jobs = asyncio.run(gather_jobs(params, args))
    except KeyboardInterrupt:
        print()
        return 130
    if not jobs:
        print("\nNo jobs retrieved - every enabled source failed. "
              "See the warnings above.\n")
        return 1

    jobs = deduplicate(jobs)
    candidates = prefilter(jobs, params, keep=max(args.limit, MAX_JOBS_TO_RANK))
    if not candidates:
        print_jobs([], query)
        return 0

    # --- Claude call #2: rank + explain -------------------------------------
    if client:
        try:
            ranked = rank_with_claude(client, query, candidates)
        except Exception as exc:
            log.warning("ranking via Claude failed (%s: %s); falling back to "
                        "local scoring", type(exc).__name__, exc)
            ranked = rank_locally(candidates, params)
    else:
        ranked = rank_locally(candidates, params)

    results = ranked[:args.limit]
    if not args.no_cache:
        cache_store(cache_path, key, ranked)

    print_jobs(results, query)
    if args.output:
        write_output(results, args.output)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
