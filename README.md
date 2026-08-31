# AI Job-Search Agent

A single-file job-search agent in pure Python. It takes a natural-language
query, fetches live postings from free public job APIs, deduplicates them, and
uses Claude to rank the results and explain each match.

No agent frameworks, no LangChain, no MCP, no scraping — just the `anthropic`
SDK, `aiohttp`, and the standard library.

```
$ python job_agent.py "senior python backend, remote, 130k+, kubernetes"

 1. [ 92] Senior Backend Engineer (Python)
    Acme Cloud  |  Remote  |  remote
    source: remoteok  |  salary: $130,000 - $170,000  |  posted: 2026-08-29
    why: Python + Kubernetes backend role at senior level, fully remote, salary above your floor
    https://remoteok.com/remote-jobs/...
```

---

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env          # then paste your ANTHROPIC_API_KEY into .env
```

Python 3.10+ (uses `X | None` type syntax and the walrus/match-free subset of
3.10). Tested on 3.12.

### Run it

```bash
# Basic search
python job_agent.py "senior python backend engineer, remote, 120k+, kubernetes"

# Interactive: prompts for the query
python job_agent.py

# More results, more of the Arbeitnow feed, export to a file
python job_agent.py "data engineer Berlin" --limit 20 --pages 4 --output jobs.csv

# Only jobs offering visa sponsorship (Arbeitnow supports this server-side)
python job_agent.py "backend developer Germany" --visa-sponsorship

# No Claude at all: local keyword scoring only (no API key needed, no cost)
python job_agent.py "golang remote" --no-rank

# Optional extra sources (need keys in .env)
python job_agent.py "python developer london" --use-adzuna --adzuna-country gb
python job_agent.py "software engineer" --use-usajobs
```

Useful flags: `--limit`, `--pages`, `--sources arbeitnow,remoteok`, `--output
jobs.csv|jobs.json`, `--no-rank`, `--no-cache`, `--clear-cache`, `-v`.
Full list: `python job_agent.py --help`.

### Verify the API schemas first (recommended)

```bash
python probe_apis.py                    # the two keyless sources
python probe_apis.py --adzuna           # Adzuna, once ADZUNA_* keys are in .env
python probe_apis.py --adzuna --country us --what "python developer" --where Austin
python probe_apis.py --usajobs          # USAJobs, once USAJOBS_* keys are in .env
python probe_apis.py --skip-keyless --adzuna --raw   # keyed sources only, verbose
```

The probe reads credentials from `.env`, masks them in its request echo, and
exits non-zero when a source it was asked to probe actually failed (a source
skipped for missing keys is not a failure). For Adzuna it additionally reports
the total match count, a populated-field census, whether each salary is
`salary_is_predicted` (Adzuna's own estimate rather than an employer figure),
description truncation, the categories endpoint, any rate-limit headers, and a
specific hint per HTTP status (401 = keys rejected, 400 = bad country code, …).

`probe_apis.py` fetches each endpoint and prints its real JSON structure: field
names, types, which fields are actually populated, and how consistent the shape
is across postings. The normalizer in `job_agent.py` was written **from this
output**, not from assumption. Job boards rename fields without notice, so if
results ever come back empty or unparsed, re-run the probe first — it will show
you exactly what changed.

---

## Data sources

Both default sources are documented, free, keyless, public JSON APIs. Nothing
here scrapes HTML, evades a paywall, or ignores a `robots.txt`.

| Source | Endpoint | Key | Notes |
|---|---|---|---|
| **Arbeitnow** | `GET https://www.arbeitnow.com/api/job-board-api` | none | ~175 jobs/page, `?page=` pagination, `?visa_sponsorship=true` filter |
| **RemoteOK** | `GET https://remoteok.com/api` | none | ~100 most recent remote jobs, single page |
| Adzuna *(optional)* | `GET https://api.adzuna.com/v1/api/jobs/{country}/search/{page}` | free `app_id`+`app_key` | aggregator, country-scoped, real server-side keyword search (`what`, `what_or`, `what_exclude`, `where`, `salary_min`, `max_days_old`) |
| USAJobs *(optional)* | `GET https://data.usajobs.gov/api/search` | free API key + registered email | US federal jobs only |

**Why these:** they are the free job APIs that publish terms permitting
programmatic access, need no commercial agreement, and return structured JSON.
Arbeitnow's own response payload includes its terms ("free public API for jobs,
please do not abuse"); RemoteOK's first array element is a legal notice asking
for attribution. The client honours both: it sends an honest identifying
`User-Agent`, makes a small number of requests per run, caches results for an
hour to avoid repeat traffic, and always links back to the original posting URL
rather than republishing the listing.

**Attribution:** results include the source and link back to the posting on
Arbeitnow / RemoteOK, as RemoteOK's API terms require.

---

## How it works

```
"senior python backend, remote, 130k+, k8s"
        │
        ├─ Claude call #1 ─ parse into structured parameters
        │    role_keywords, seniority, remote_preference, locations,
        │    salary_floor_usd, tech_stack, visa_sponsorship_required,
        │    exclude_keywords   (JSON-schema structured output)
        │
        ├─ concurrent fetch (asyncio + aiohttp) ─ Arbeitnow ‖ RemoteOK ‖ …
        │    per-source retries; a dead or rate-limited source is skipped
        │    with a warning, never fatal
        │
        ├─ normalize ─ one schema: title, company, location, salary,
        │    posted_date, apply_url, source (+ remote, tags, description)
        │
        ├─ dedupe ─ fuzzy (title, company) similarity via difflib; legal
        │    suffixes and seniority noise ignored; the posting that carries
        │    salary data wins
        │
        ├─ local prefilter ─ cheap keyword/constraint scoring, top 40 kept
        │    (this is what keeps the ranking call affordable)
        │
        ├─ Claude call #2 ─ rank 0-100 + one-line "why this matches" per job
        │    (streamed, JSON-schema structured output, joined back by id)
        │
        └─ console listing + optional jobs.csv / jobs.json
```

Exactly two Claude calls per run, both `claude-opus-5`, both using JSON-schema
structured outputs so no output parsing is guesswork. Results are cached in
`.job_cache.json` for one hour keyed on the query plus the flags that affect
it, so repeating a query costs nothing.

### Failure behaviour

Every failure mode degrades instead of crashing:

| What breaks | What happens |
|---|---|
| One source down / 5xx / rate-limited (429) | retried with backoff, then skipped with a `WARNING`; other sources still return |
| All sources down | clear message, exit code 1 |
| No `ANTHROPIC_API_KEY` | warns once, falls back to local keyword scoring and still prints results |
| Claude call fails (auth, network, quota) | warns, falls back to local scoring for that stage |
| Claude declines or truncates | detected via `stop_reason`, treated as a failed stage, falls back |
| Claude omits a job from the ranking | job is kept, shown unranked (`--`), never silently dropped |
| Corrupt cache file | ignored and rewritten |
| Non-ASCII titles on a legacy Windows code page | replaced, not a crash |

---

## Known limitations

These are properties of the data sources, not bugs:

- **Arbeitnow skews European, heavily German.** Most postings are in Germany
  and many are written in German. A US-centric query will match poorly.
- **Arbeitnow has no salary field at all.** Salary is blank for every Arbeitnow
  posting, so a salary floor can only filter RemoteOK/Adzuna/USAJobs results.
- **RemoteOK is remote-only and small.** One page, ~100 most recent postings, no
  pagination and no keyword parameter. Salary is populated on roughly 4% of
  them (`0` means "not disclosed").
- **Neither keyless API supports server-side keyword search.** The tool fetches
  the recent feed and filters locally, so a niche query can legitimately match
  nothing — the run says so rather than dressing up noise as results. Increase
  `--pages` to widen the Arbeitnow window, or add Adzuna, which does have real
  server-side search.
- **Recency-ordered feeds, not a searchable archive.** These are "what was
  posted recently" endpoints. This is a discovery tool, not an exhaustive
  market survey.
- **Deduplication is fuzzy.** `(title, company)` similarity catches reposts and
  cross-source overlap, but the same role advertised under a genuinely different
  title will survive as two entries.
- **Ranking is a language model's judgement**, based only on the title, company,
  location, tags, salary and the first ~400 characters of the description. Read
  the actual posting before applying.
- **RemoteOK descriptions are sometimes double-encoded** upstream; the client
  repairs the common case, but odd characters can still slip through.
- **Adzuna salaries are in local currency and may be estimates.** The API
  returns figures in the country's own currency, so a USD salary floor is passed
  through unconverted and is only approximate outside the US. Where
  `salary_is_predicted` is set, the figure is Adzuna's own estimate, not an
  employer's — those are labelled *"(Adzuna estimate, not employer-stated)"* so
  the ranking never treats a guess as a disclosed salary.
- **Adzuna has no remote flag.** Remote is inferred only from an explicit
  statement in the job title or location, so a remote-only query will
  under-report Adzuna results rather than mislabel them.
- **Adzuna and USAJobs normalizers are field-verified but not live-verified.**
  Adzuna's request parameters and response fields come from its published
  Swagger spec (`https://api.adzuna.com/v1/api-docs/adzuna`) and the normalizer
  is unit-tested against a spec-shaped fixture, but no live call has been made
  with real keys; the same is true of USAJobs, from its docs. Every field access
  is defensive, so a drift degrades a field rather than crashing. Run
  `python probe_apis.py --adzuna` / `--usajobs` once you have keys to confirm.
- **Cost:** two Opus calls per uncached run, one of them carrying ~40 job
  digests. Small, but not free. `--no-rank` makes a run cost nothing, and the
  one-hour cache prevents paying twice for the same query.

---

## Files

| File | Purpose |
|---|---|
| `job_agent.py` | the agent — fetch, normalize, dedupe, rank, output |
| `probe_apis.py` | schema inspector; run this before trusting any field name |
| `requirements.txt` | dependencies |
| `.env.example` | template with instructions for the optional API keys |
| `.job_cache.json` | auto-created result cache (1-hour TTL); safe to delete |

Add `.env` and `.job_cache.json` to `.gitignore` — the first holds secrets, the
second is disposable.
