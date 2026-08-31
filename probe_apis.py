"""Schema probe: fetch each job endpoint once and print its real JSON shape.

Run this BEFORE trusting any field name in job_agent.py. Job boards change
field names without notice; this script is the ground truth for the normalizer.

Usage:
    python probe_apis.py                        # the two keyless sources
    python probe_apis.py --adzuna               # also probe Adzuna (needs keys)
    python probe_apis.py --adzuna --country us --what "python developer"
    python probe_apis.py --adzuna --where London --page 2 --raw
    python probe_apis.py --usajobs              # also probe USAJobs (needs keys)

Credentials for the optional sources are read from `.env` (see .env.example) or
from the environment.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests

# Corporate TLS-inspecting proxies present a private root CA that certifi does
# not ship. `truststore` makes Python use the OS certificate store instead,
# which is where such a CA is installed. Harmless when not behind a proxy.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

UA = "job-seeker-agent/1.0 (personal job search)"
TIMEOUT = 25

# Country codes Adzuna publishes an endpoint for, per its docs.
ADZUNA_COUNTRIES = ["gb", "us", "at", "au", "be", "br", "ca", "ch", "de", "es",
                    "fr", "in", "it", "mx", "nl", "nz", "pl", "sg", "za"]


def load_dotenv(path=".env"):
    """Load KEY=VALUE lines from .env without overwriting the environment."""
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def describe(value, depth=0, max_depth=3):
    """Render a compact type/shape sketch of arbitrary JSON."""
    pad = "  " * depth
    if isinstance(value, dict):
        if depth >= max_depth:
            return f"{{...{len(value)} keys...}}"
        lines = []
        for k, v in value.items():
            lines.append(f"{pad}  {k}: {describe(v, depth + 1, max_depth)}")
        return "{\n" + "\n".join(lines) + f"\n{pad}}}"
    if isinstance(value, list):
        if not value:
            return "[] (empty)"
        return f"[{len(value)} x {describe(value[0], depth, max_depth)}]"
    if isinstance(value, str):
        preview = value.replace("\n", " ")[:70]
        return f"str({len(value)}) {preview!r}"
    return f"{type(value).__name__}({value!r})"


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def census(records, fields, label="populated"):
    """
    How often is each field actually usable?

    A field that exists in every object but is empty in 96% of them (RemoteOK's
    salary, say) is a trap: the schema says yes, the data says no. This counts
    non-empty values so the normalizer can be built on what is really there.
    """
    total = len(records)
    print(f"\n{label} field census over {total} record(s):")
    for f in fields:
        n = sum(1 for r in records
                if r.get(f) not in (None, "", [], {}, 0, "0", False))
        bar = "#" * round(20 * n / total) if total else ""
        print(f"   {f:22s} {n:4d}/{total:<4d} {bar}")


def keysets(records, limit=60):
    """Report whether every object really has the same shape."""
    counts = {}
    for r in records[:limit]:
        counts[tuple(sorted(r.keys()))] = counts.get(tuple(sorted(r.keys())), 0) + 1
    print(f"\ndistinct keysets across first {min(limit, len(records))} record(s): "
          f"{len(counts)}")
    for ks, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"   x{n}: {ks}")


def probe_arbeitnow():
    section("ARBEITNOW  GET https://www.arbeitnow.com/api/job-board-api")
    r = requests.get(
        "https://www.arbeitnow.com/api/job-board-api",
        params={"page": 1},
        headers={"User-Agent": UA, "Accept": "application/json"},
        timeout=TIMEOUT,
    )
    print(f"HTTP {r.status_code}  content-type={r.headers.get('content-type')}")
    r.raise_for_status()
    payload = r.json()
    print(f"top-level type: {type(payload).__name__}")
    if isinstance(payload, dict):
        print(f"top-level keys: {list(payload.keys())}")
        for meta_key in ("links", "meta"):
            if meta_key in payload:
                print(f"\n{meta_key} = {json.dumps(payload[meta_key], indent=2)[:600]}")
    data = payload.get("data") if isinstance(payload, dict) else payload
    print(f"\ndata length: {len(data)}")
    print("\nFIRST JOB - shape:")
    print(describe(data[0]))
    print("\nFIRST JOB - raw (truncated):")
    print(json.dumps(data[0], indent=2)[:1800])
    keysets = {tuple(sorted(j.keys())) for j in data[:40]}
    print(f"\ndistinct keysets across first 40 jobs: {len(keysets)}")
    for ks in keysets:
        print("  ", ks)


def probe_remoteok():
    section("REMOTEOK  GET https://remoteok.com/api")
    r = requests.get(
        "https://remoteok.com/api",
        headers={"User-Agent": UA, "Accept": "application/json"},
        timeout=TIMEOUT,
    )
    print(f"HTTP {r.status_code}  content-type={r.headers.get('content-type')}")
    r.raise_for_status()
    payload = r.json()
    print(f"top-level type: {type(payload).__name__}  length: {len(payload)}")
    print("\nELEMENT 0 (expected: legal notice, NOT a job):")
    print(json.dumps(payload[0], indent=2)[:800])
    print("\nELEMENT 1 - shape:")
    print(describe(payload[1]))
    print("\nELEMENT 1 - raw (truncated):")
    print(json.dumps(payload[1], indent=2)[:1800])
    jobs = payload[1:]
    keysets(jobs)
    # `company_name` is listed deliberately: it is Arbeitnow's field name, and
    # the census proves RemoteOK never populates it (the field here is `company`).
    census(jobs, ["position", "company", "company_name", "location", "salary_min",
                  "salary_max", "date", "epoch", "url", "apply_url", "tags",
                  "description", "original"])


def probe_adzuna(country="gb", what="python developer", where=None, page=1,
                 results=20, show_raw=False):
    """
    Probe the Adzuna job search endpoint.

    Request/response shape per Adzuna's own Swagger spec
    (https://api.adzuna.com/v1/api-docs/adzuna), which this probe verifies
    against live data:
        GET /v1/api/jobs/{country}/search/{page}
        required: country, app_id, app_key, page
        optional: what, what_and, what_phrase, what_or, what_exclude,
                  title_only, where, distance, location0..7, category,
                  salary_min, salary_max, salary_include_unknown, max_days_old,
                  full_time, part_time, contract, permanent, company,
                  sort_by, sort_direction, results_per_page
        response: {count, mean, results[]}
        results[] item (Adzuna::API::Response::Job): id, title, description,
                  full_description?, created, redirect_url, adref,
                  company{display_name, canonical_name}, location{display_name,
                  area[]}, category{tag, label}, salary_min, salary_max,
                  salary_is_predicted, contract_type, contract_time,
                  latitude, longitude
    """
    url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
    section(f"ADZUNA  GET {url}")

    app_id, app_key = os.getenv("ADZUNA_APP_ID"), os.getenv("ADZUNA_APP_KEY")
    if not (app_id and app_key):
        print("SKIP: ADZUNA_APP_ID / ADZUNA_APP_KEY are not set.\n"
              "      Get free credentials at https://developer.adzuna.com/ and\n"
              "      put them in .env (see .env.example), then re-run with --adzuna.")
        return "skipped"
    if country not in ADZUNA_COUNTRIES:
        print(f"WARNING: '{country}' is not in Adzuna's published country list "
              f"({', '.join(ADZUNA_COUNTRIES)}); expect HTTP 400.")

    params = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": results,
        "content-type": "application/json",
    }
    if what:
        params["what"] = what
    if where:
        params["where"] = where

    # Echo the request with the credentials masked, so a failure is debuggable
    # without leaking keys into the terminal or a pasted bug report.
    shown = {**params, "app_id": app_id[:4] + "***", "app_key": app_key[:4] + "***"}
    print(f"request params: {shown}")

    r = requests.get(url, params=params,
                     headers={"User-Agent": UA, "Accept": "application/json"},
                     timeout=TIMEOUT)
    print(f"HTTP {r.status_code}  content-type={r.headers.get('content-type')}")

    # Any rate-limit headers Adzuna chooses to send (its docs don't document
    # the quota, so read whatever is actually on the response).
    limits = {k: v for k, v in r.headers.items()
              if any(t in k.lower() for t in ("ratelimit", "quota", "retry-after"))}
    print(f"rate-limit headers: {limits or 'none sent'}")

    if r.status_code != 200:
        print("\nBODY:", r.text[:600])
        hint = {
            400: "bad country code or a malformed parameter",
            401: "app_id/app_key rejected - check for swapped or truncated values",
            403: "credentials not authorised for this endpoint/country",
            404: "wrong path or an unsupported country code",
            410: "endpoint retired",
            429: "rate limit exceeded - wait and retry",
        }.get(r.status_code)
        if hint:
            print(f"\nHINT: {hint}")
        return False

    payload = r.json()
    print(f"\ntop-level type: {type(payload).__name__}")
    print(f"top-level keys: {list(payload.keys())}")
    print(f"count (total ads matching): {payload.get('count')}")
    print(f"mean (avg salary across results): {payload.get('mean')}")

    jobs = payload.get("results") or []
    print(f"results returned: {len(jobs)} (asked for {results})")
    if not jobs:
        print("\nNo results for this query - try a broader --what or another "
              "--country before concluding the schema changed.")
        return True

    print("\nFIRST JOB - shape:")
    print(describe(jobs[0]))
    print("\nFIRST JOB - raw:")
    print(json.dumps(jobs[0], indent=2)[:2000])
    if show_raw and len(jobs) > 1:
        print("\nSECOND JOB - raw:")
        print(json.dumps(jobs[1], indent=2)[:2000])

    keysets(jobs)
    census(jobs, ["id", "title", "description", "full_description", "created",
                  "redirect_url", "adref", "company", "location", "category",
                  "salary_min", "salary_max", "salary_is_predicted",
                  "contract_type", "contract_time", "latitude", "longitude"])

    # The fields job_agent.py reads are nested; confirm the nesting is real
    # rather than assuming company/location are plain strings.
    print("\nnested-field spot check (what the normalizer actually reads):")
    for i, job in enumerate(jobs[:5]):
        comp = (job.get("company") or {}).get("display_name")
        loc = (job.get("location") or {}).get("display_name")
        area = (job.get("location") or {}).get("area")
        cat = (job.get("category") or {}).get("label")
        print(f"   [{i}] company={comp!r}")
        print(f"       location={loc!r}")
        print(f"       area={area!r}")
        print(f"       category={cat!r}  contract={job.get('contract_type')!r}/"
              f"{job.get('contract_time')!r}")
        print(f"       salary={job.get('salary_min')!r}-{job.get('salary_max')!r}"
              f"  predicted={job.get('salary_is_predicted')!r}"
              f" ({type(job.get('salary_is_predicted')).__name__})")
        print(f"       created={job.get('created')!r}")

    # salary_is_predicted is the field that matters most for honesty: a
    # "predicted" salary is Adzuna's own estimate, not something the employer
    # disclosed, and must not be presented to the user as a real figure.
    predicted = [j for j in jobs
                 if str(j.get("salary_is_predicted", "")).strip() in ("1", "True", "true")]
    with_salary = [j for j in jobs if (j.get("salary_min") or j.get("salary_max"))]
    print(f"\nsalary reality check: {len(with_salary)}/{len(jobs)} have a salary; "
          f"{len(predicted)} of those are PREDICTED (Adzuna estimate, not "
          f"employer-disclosed)")
    print("NOTE: salaries are in the country's LOCAL CURRENCY, not USD.")

    # Description truncation: the spec says `description` is capped at 500
    # chars and `full_description` may be absent. Verify both claims.
    lengths = [len(j.get("description") or "") for j in jobs]
    print(f"description length: min={min(lengths)} max={max(lengths)} "
          f"(spec says truncated to 500)")
    print(f"full_description present on "
          f"{sum(1 for j in jobs if j.get('full_description'))}/{len(jobs)}")

    # The categories endpoint provides the valid `category` tags for filtering.
    cat_url = f"https://api.adzuna.com/v1/api/jobs/{country}/categories"
    rc = requests.get(cat_url, params={"app_id": app_id, "app_key": app_key},
                      headers={"User-Agent": UA}, timeout=TIMEOUT)
    print(f"\ncategories endpoint: HTTP {rc.status_code}")
    if rc.status_code == 200:
        tags = [(c.get("tag"), c.get("label"))
                for c in (rc.json().get("results") or [])]
        print(f"   {len(tags)} categories, e.g. {tags[:5]}")


def probe_usajobs():
    section("USAJOBS  GET https://data.usajobs.gov/api/search")
    key, email = os.getenv("USAJOBS_API_KEY"), os.getenv("USAJOBS_EMAIL")
    if not (key and email):
        print("SKIP: USAJOBS_API_KEY / USAJOBS_EMAIL not set.\n"
              "      Free key: https://developer.usajobs.gov/APIRequest/Index")
        return "skipped"
    r = requests.get(
        "https://data.usajobs.gov/api/search",
        params={"Keyword": "software engineer", "ResultsPerPage": 5},
        headers={"Host": "data.usajobs.gov", "User-Agent": email,
                 "Authorization-Key": key},
        timeout=TIMEOUT,
    )
    print(f"HTTP {r.status_code}")
    if r.status_code != 200:
        print(r.text[:500])
        return False
    payload = r.json()
    items = payload["SearchResult"]["SearchResultItems"]
    print(f"result count: {payload['SearchResult'].get('SearchResultCount')}")
    print("\nFIRST ITEM - shape (MatchedObjectDescriptor):")
    print(describe(items[0]["MatchedObjectDescriptor"], max_depth=2))


def main():
    ap = argparse.ArgumentParser(
        description="Inspect the raw JSON schema of each job source.")
    ap.add_argument("--adzuna", action="store_true", help="also probe Adzuna")
    ap.add_argument("--usajobs", action="store_true", help="also probe USAJobs")
    ap.add_argument("--only", choices=["arbeitnow", "remoteok"],
                   help="probe just one of the keyless sources")
    ap.add_argument("--skip-keyless", action="store_true",
                   help="skip Arbeitnow/RemoteOK (probe only the keyed sources)")
    # Adzuna query knobs - its results depend entirely on the query, so a
    # useful probe has to let you vary it.
    ap.add_argument("--country", default="gb",
                   help=f"Adzuna country code (default: gb; known: "
                        f"{', '.join(ADZUNA_COUNTRIES)})")
    ap.add_argument("--what", default="python developer",
                   help="Adzuna keyword query (default: 'python developer')")
    ap.add_argument("--where", help="Adzuna location, e.g. London")
    ap.add_argument("--page", type=int, default=1, help="Adzuna page (default: 1)")
    ap.add_argument("--results", type=int, default=20,
                   help="Adzuna results_per_page (default: 20, max 50)")
    ap.add_argument("--raw", action="store_true",
                   help="dump a second raw record for comparison")
    args = ap.parse_args()

    load_dotenv()

    failures = []
    keyless = [("arbeitnow", probe_arbeitnow), ("remoteok", probe_remoteok)]
    if args.only:
        keyless = [(n, f) for n, f in keyless if n == args.only]
    if args.skip_keyless:
        keyless = []

    for name, fn in keyless:
        try:
            fn()
        except Exception as exc:
            failures.append(name)
            print(f"\n!! {name} probe FAILED: {type(exc).__name__}: {exc}")

    # Keyed sources report their own outcome: False means we reached the API and
    # it refused us (a real failure), "skipped" means no credentials were
    # configured (not a failure - the run was simply not asked to test them).
    skipped = []
    keyed = []
    if args.adzuna:
        keyed.append(("adzuna", lambda: probe_adzuna(
            country=args.country, what=args.what, where=args.where,
            page=args.page, results=args.results, show_raw=args.raw)))
    if args.usajobs:
        keyed.append(("usajobs", probe_usajobs))

    for name, fn in keyed:
        try:
            outcome = fn()
        except Exception as exc:
            failures.append(name)
            print(f"\n!! {name} probe FAILED: {type(exc).__name__}: {exc}")
            continue
        if outcome is False:
            failures.append(name)
        elif outcome == "skipped":
            skipped.append(name)

    print("\n" + "-" * 78)
    summary = "probe complete."
    summary += (f" FAILED: {', '.join(failures)}." if failures
                else " all probed sources reachable.")
    if skipped:
        summary += f" skipped (no credentials): {', '.join(skipped)}."
    print(summary)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
