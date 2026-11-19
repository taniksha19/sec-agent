"""Download filings for the companies in data/companies.csv from SEC EDGAR.

Per company: the last 3 10-Ks, the last 4 10-Qs, and the Exhibit 99.1 earnings release from
each 8-K with Item 2.02 (Results of Operations) filed since the oldest of those 10-Qs.

Usage: uv run python -m ingest.download [--tickers AAPL MSFT]

Already-downloaded documents are skipped, so reruns only fetch what is missing.
"""

import argparse
import csv
import json
import logging
import os
import time
from datetime import date
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from ingest.models import Company, Filing

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPANIES_CSV = REPO_ROOT / "data" / "companies.csv"
RAW_DIR = REPO_ROOT / "data" / "raw"

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{name}"

N_10K = 3
N_10Q = 4
# SEC allows 10 requests/second; stay well under it.
MIN_REQUEST_INTERVAL_S = 0.15
MAX_RETRIES = 5
# Large companies must file within 40-60 days of quarter end; releases come before that.
RELEASE_MAX_LAG_DAYS = 60


class EdgarClient:
    """httpx client that sets the required User-Agent, throttles, and retries 429/5xx."""

    def __init__(self, user_agent: str):
        self._client = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=30.0,
            follow_redirects=True,
        )
        self._last_request = 0.0

    def get(self, url: str) -> httpx.Response:
        for attempt in range(MAX_RETRIES):
            wait = MIN_REQUEST_INTERVAL_S - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            resp = self._client.get(url)
            if resp.status_code == 429 or resp.status_code >= 500:
                backoff = 2**attempt
                log.warning("%s from %s, retrying in %ss", resp.status_code, url, backoff)
                time.sleep(backoff)
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError(f"Gave up on {url} after {MAX_RETRIES} attempts")

    def get_json(self, url: str) -> dict:
        return self.get(url).json()


def check_user_agent(value: str | None) -> str:
    """EDGAR blocks requests without a real contact in the User-Agent."""
    if not value or "@" not in value or "example.com" in value:
        raise SystemExit(
            "Set SEC_USER_AGENT in .env to '<project name> <your email>' "
            "(see .env.example). EDGAR requires a real contact address."
        )
    return value


def load_companies(path: Path = COMPANIES_CSV) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_ciks(tickers_json: dict, tickers: list[str]) -> dict[str, str]:
    """Map ticker -> 10-digit CIK using SEC's company_tickers.json."""
    by_ticker = {row["ticker"].upper(): row["cik_str"] for row in tickers_json.values()}
    missing = [t for t in tickers if t.upper() not in by_ticker]
    if missing:
        raise ValueError(f"Tickers not found on EDGAR: {missing}")
    return {t: f"{int(by_ticker[t.upper()]):010d}" for t in tickers}


def rows_from_columns(columns: dict) -> list[dict]:
    """EDGAR returns filings as parallel arrays; turn them into one dict per filing."""
    keys = list(columns)
    return [dict(zip(keys, values, strict=True)) for values in zip(*columns.values(), strict=True)]


def select_filings(rows: list[dict]) -> list[dict]:
    """Pick the filings we want from rows sorted newest first.

    Amendments (10-K/A, 10-Q/A) are skipped: they are often partial documents.
    """
    tenks = [r for r in rows if r["form"] == "10-K"][:N_10K]
    tenqs = [r for r in rows if r["form"] == "10-Q"][:N_10Q]
    if not tenqs:
        return tenks
    since = min(r["reportDate"] for r in tenqs)
    eightks = [
        r
        for r in rows
        if r["form"] == "8-K"
        and "2.02" in r.get("items", "").split(",")
        and r["filingDate"] >= since
    ]
    return tenks + tenqs + eightks


def fetch_filing_rows(client: EdgarClient, cik: str) -> tuple[dict, list[dict]]:
    """Return the submissions JSON and filing rows, paging into older files until we have
    enough 10-Ks. Needed for banks, whose many prospectus filings push 10-Ks out of 'recent'."""
    sub = client.get_json(SUBMISSIONS_URL.format(name=f"CIK{cik}.json"))
    rows = rows_from_columns(sub["filings"]["recent"])
    for older in sub["filings"].get("files", []):
        if sum(r["form"] == "10-K" for r in rows) >= N_10K:
            break
        rows += rows_from_columns(client.get_json(SUBMISSIONS_URL.format(name=older["name"])))
    for r in rows:
        r["cik"] = cik  # archive URLs need the CIK that actually filed the document
    return sub, rows


def merge_rows(row_lists: list[list[dict]]) -> list[dict]:
    """Combine filings from a company's current and prior CIKs, newest first.

    Around a reorganization the same report can be listed under both CIKs, so duplicates are
    dropped by accession number and, for 10-K/10-Q, by (form, period). row_lists[0] is the
    current CIK and wins ties (sorted() is stable).
    """
    rows = sorted(
        (r for rows in row_lists for r in rows), key=lambda r: r["filingDate"], reverse=True
    )
    seen: set = set()
    unique = []
    for r in rows:
        keys = {r["accessionNumber"]}
        if r["form"] in ("10-K", "10-Q") and r.get("reportDate"):
            keys.add((r["form"], r["reportDate"]))
        if keys & seen:
            continue
        seen |= keys
        unique.append(r)
    return unique


def fiscal_periods_by_accession(company_facts: dict) -> dict[str, tuple[int, str]]:
    """Map accession number -> (fiscal year, fiscal period) as the company tagged it in XBRL.

    Used instead of deriving from dates because companies label fiscal years differently
    (Target's year ending Feb 2025 is 'fiscal 2024'; Walmart's is 'fiscal 2025').
    """
    result: dict[str, tuple[int, str]] = {}
    for taxonomy in company_facts.get("facts", {}).values():
        for concept in taxonomy.values():
            for unit_facts in concept.get("units", {}).values():
                for fact in unit_facts:
                    if fact.get("form") in ("10-K", "10-Q") and fact.get("fy") and fact.get("fp"):
                        result.setdefault(fact["accn"], (fact["fy"], fact["fp"]))
    return result


def find_exhibit_99(index_html: str) -> str | None:
    """Return the file name of the earnings-release exhibit from a filing's -index.htm page.

    Prefers EX-99.1; falls back to the first EX-99.x (some filers number it EX-99 or EX-99.01).
    """
    soup = BeautifulSoup(index_html, "lxml")
    candidates: list[tuple[str, str]] = []
    for row in soup.select("table.tableFile tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        doc_type = cells[3].get_text(strip=True).upper()
        link = cells[2].find("a")
        if doc_type.startswith("EX-99") and link:
            name = link.get_text(strip=True)
            candidates.append((doc_type, name))
    for doc_type, name in candidates:
        if doc_type == "EX-99.1":
            return name
    return candidates[0][1] if candidates else None


def match_release_period(release_date: str, rows: list[dict]) -> dict | None:
    """Find the 10-K/10-Q whose quarter an earnings release (8-K) reports on.

    Releases come out a few weeks after quarter end, so pick the latest period that ended
    before the release, within RELEASE_MAX_LAG_DAYS. Returns None if nothing fits (for example,
    the release is newer than any 10-Q on EDGAR) rather than guessing.
    """
    release = date.fromisoformat(release_date)
    best: dict | None = None
    for r in rows:
        if r["form"] not in ("10-K", "10-Q") or not r.get("reportDate"):
            continue
        period_end = date.fromisoformat(r["reportDate"])
        lag = (release - period_end).days
        if 0 < lag <= RELEASE_MAX_LAG_DAYS and (
            best is None or period_end > date.fromisoformat(best["reportDate"])
        ):
            best = r
    return best


def download_company(client: EdgarClient, info: dict, cik: str) -> list[Filing]:
    ticker = info["ticker"]
    out_dir = RAW_DIR / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    # A company that reorganized (e.g. XOM into a holding company in 2026) gets a new CIK;
    # its history stays under the prior one, listed in companies.csv.
    prior_ciks = [f"{int(c):010d}" for c in info.get("prior_cik", "").split(";") if c.strip()]
    sub, rows = fetch_filing_rows(client, cik)
    row_lists = [rows] + [fetch_filing_rows(client, c)[1] for c in prior_ciks]
    rows = merge_rows(row_lists)
    company = Company(
        ticker=ticker,
        name=info["name"],
        sector=info["sector"],
        cik=cik,
        prior_ciks=prior_ciks,
        fiscal_year_end=sub.get("fiscalYearEnd"),
    )
    (out_dir / "company.json").write_text(company.model_dump_json(indent=2), encoding="utf-8")

    fiscal: dict[str, tuple[int, str]] = {}
    for c in [cik, *prior_ciks]:
        fiscal |= fiscal_periods_by_accession(client.get_json(COMPANY_FACTS_URL.format(cik=c)))

    filings: list[Filing] = []
    for row in select_filings(rows):
        acc = row["accessionNumber"]
        acc_nodash = acc.replace("-", "")
        base = dict(cik_int=int(row["cik"]), acc_nodash=acc_nodash)

        if row["form"] == "8-K":
            index_url = ARCHIVE_URL.format(**base, name=f"{acc}-index.htm")
            doc_name = find_exhibit_99(client.get(index_url).text)
            if doc_name is None:
                log.warning("%s 8-K %s has no EX-99 exhibit, skipping", ticker, acc)
                continue
        else:
            doc_name = row["primaryDocument"]

        url = ARCHIVE_URL.format(**base, name=doc_name)
        period = row.get("reportDate") or None
        stem = f"{row['form']}_{period or row['filingDate']}_{acc}"
        local = out_dir / f"{stem}{Path(doc_name).suffix}"
        if not local.exists():
            local.write_bytes(client.get(url).content)
            log.info("%s %s %s downloaded", ticker, row["form"], period)

        fy, fp = fiscal.get(acc, (None, None))
        if row["form"] == "8-K":
            # The 8-K's own reportDate is the release date; label it with the quarter it covers.
            matched = match_release_period(row["filingDate"], rows)
            period = matched["reportDate"] if matched else None
            fy, fp = (
                fiscal.get(matched["accessionNumber"], (None, None)) if matched else (None, None)
            )
            if fp == "FY":
                fp = "Q4"  # a release matched to a 10-K covers the fourth quarter
        filings.append(
            Filing(
                ticker=ticker,
                cik=row["cik"],
                form_type=row["form"],
                accession_number=acc,
                filing_date=date.fromisoformat(row["filingDate"]),
                period_of_report=date.fromisoformat(period) if period else None,
                fiscal_year=fy,
                fiscal_period=fp,
                document_name=doc_name,
                source_url=url,
                local_path=local.relative_to(REPO_ROOT).as_posix(),
            )
        )

    manifest = [f.model_dump(mode="json") for f in filings]
    (out_dir / "filings.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return filings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tickers", nargs="*", help="Subset of tickers (default: all)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    load_dotenv(REPO_ROOT / ".env")
    client = EdgarClient(check_user_agent(os.getenv("SEC_USER_AGENT")))

    companies = load_companies()
    if args.tickers:
        wanted = {t.upper() for t in args.tickers}
        companies = [c for c in companies if c["ticker"] in wanted]
    ciks = resolve_ciks(client.get_json(TICKERS_URL), [c["ticker"] for c in companies])

    for info in companies:
        filings = download_company(client, info, ciks[info["ticker"]])
        counts = {
            form: sum(f.form_type == form for f in filings) for form in ("10-K", "10-Q", "8-K")
        }
        log.info("%s done: %s", info["ticker"], counts)


if __name__ == "__main__":
    main()
