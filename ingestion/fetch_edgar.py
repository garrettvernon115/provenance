"""Fetch recent SEC EDGAR filings onto local disk (Phase 0 corpus bootstrap).

Downloads the N most recent 10-K and Form 4 filings into
``data/raw/<form>/<accession>/`` — one primary document plus a
``metadata.json`` per filing. "Recent" comes from EDGAR's daily master
indexes, walked backwards from today; the primary document name is resolved
via the data.sec.gov submissions API, falling back to the full submission
``.txt`` when resolution fails.

SEC fair-access policy (https://www.sec.gov/os/accessing-edgar-data) requires
a descriptive User-Agent with contact info and caps clients at 10 requests
per second; this fetcher refuses to run without a User-Agent and throttles
below the cap.

Usage:
    python fetch_edgar.py --num-10k 10 --num-form4 25
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

import requests

log = logging.getLogger("fetch_edgar")

SEC_ARCHIVES = "https://www.sec.gov/Archives"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_MAX_RPS = 10.0  # EDGAR fair-access cap; do not exceed
DEFAULT_RPS = 8.0
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Filing:
    """One row of an EDGAR daily master index."""

    cik: int
    company: str
    form: str
    filed: str  # ISO date, e.g. "2026-06-09"
    txt_path: str  # e.g. "edgar/data/320193/0000320193-26-000008.txt"
    accession: str  # e.g. "0000320193-26-000008"


# --------------------------------------------------------------------------
# Pure helpers (unit-tested, no network)
# --------------------------------------------------------------------------


def quarter_of(d: date) -> int:
    return (d.month - 1) // 3 + 1


def daily_index_url(d: date) -> str:
    """URL of the pipe-delimited daily master index for a given date."""
    return (
        f"{SEC_ARCHIVES}/edgar/daily-index/{d.year}/QTR{quarter_of(d)}/"
        f"master.{d:%Y%m%d}.idx"
    )


def filed_iso(raw: str) -> str:
    """Normalize an index date ("20260609" or "2026-06-09") to ISO form."""
    raw = raw.strip()
    if "-" in raw:
        return raw
    return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"


def accession_from_txt_path(txt_path: str) -> str:
    name = txt_path.rsplit("/", 1)[-1]
    return name[:-4] if name.endswith(".txt") else name


def parse_master_index(text: str) -> list[Filing]:
    """Parse a daily master index: ``CIK|Company Name|Form Type|Date|File``.

    Header/separator lines are dropped by requiring a numeric CIK and a
    ``.txt`` file path.
    """
    filings: list[Filing] = []
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) != 5:
            continue
        cik, company, form, filed, txt_path = (p.strip() for p in parts)
        if not cik.isdigit() or not txt_path.endswith(".txt"):
            continue
        filings.append(
            Filing(
                cik=int(cik),
                company=company,
                form=form,
                filed=filed_iso(filed),
                txt_path=txt_path,
                accession=accession_from_txt_path(txt_path),
            )
        )
    return filings


def strip_xsl(doc: str) -> str:
    """Drop an ``xsl.../`` rendering prefix to reach the raw document.

    The submissions API sometimes reports ownership-form primary documents as
    e.g. ``xslF345X05/wk-form4_1.xml``; the raw XML lives at the unprefixed
    path.
    """
    if "/" in doc:
        first, rest = doc.split("/", 1)
        if first.lower().startswith("xsl"):
            return rest
    return doc


def primary_doc_url(cik: int, accession: str, doc: str) -> str:
    return f"{SEC_ARCHIVES}/edgar/data/{cik}/{accession.replace('-', '')}/{doc}"


def full_txt_url(txt_path: str) -> str:
    return f"{SEC_ARCHIVES}/{txt_path}"


def form_dir_name(form: str) -> str:
    """Filesystem-safe directory name for a form type (e.g. "10-K/A")."""
    return form.replace("/", "-").replace(" ", "_")


def collect_recent(
    get_day: Callable[[date], Optional[list[Filing]]],
    wanted: dict[str, int],
    start: date,
    max_days: int,
) -> dict[str, list[Filing]]:
    """Walk daily indexes backwards from ``start`` until counts are met.

    ``get_day`` returns the day's filings, or None when no index exists
    (weekends, holidays, not yet published). Duplicate accessions are
    dropped — Form 4s appear once per associated CIK (issuer + each
    reporting owner).
    """
    selected: dict[str, list[Filing]] = {form: [] for form in wanted}
    remaining = {form for form, n in wanted.items() if n > 0}
    seen: set[str] = set()

    for offset in range(max_days):
        if not remaining:
            break
        day = start - timedelta(days=offset)
        entries = get_day(day)
        if entries is None:
            continue
        for entry in entries:
            if entry.form not in remaining or entry.accession in seen:
                continue
            seen.add(entry.accession)
            selected[entry.form].append(entry)
            if len(selected[entry.form]) >= wanted[entry.form]:
                remaining.discard(entry.form)

    for form, n in wanted.items():
        if len(selected[form]) < n:
            log.warning(
                "only found %d/%d %s filings within %d days of %s",
                len(selected[form]), n, form, max_days, start,
            )
    return selected


# --------------------------------------------------------------------------
# EDGAR HTTP client (throttled, retrying)
# --------------------------------------------------------------------------


class EdgarClient:
    def __init__(self, user_agent: str, rps: float = DEFAULT_RPS):
        if rps > SEC_MAX_RPS:
            log.warning("clamping rate to SEC fair-access cap of %s req/s", SEC_MAX_RPS)
        self._min_interval = 1.0 / min(max(rps, 0.1), SEC_MAX_RPS)
        self._last_request = 0.0
        self._submissions_cache: dict[int, Optional[dict]] = {}
        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        )

    def _throttle(self) -> None:
        wait = self._last_request + self._min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def get(
        self, url: str, *, ok404: bool = False, ok403: bool = False
    ) -> Optional[requests.Response]:
        """GET with throttling and backoff on 403/429/5xx (EDGAR throttling).

        ``ok403`` exists because EDGAR serves 403 — not 404 — for daily-index
        files that don't exist (yet); callers walking the calendar treat that
        as "no index for this day" rather than as throttling.
        """
        backoff = 1.0
        for attempt in range(4):
            self._throttle()
            resp = self._session.get(url, timeout=30)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 404 and ok404:
                return None
            if resp.status_code == 403 and ok403:
                return None
            if resp.status_code in (403, 429) or resp.status_code >= 500:
                log.warning(
                    "HTTP %d from %s (attempt %d/4); backing off %.0fs",
                    resp.status_code, url, attempt + 1, backoff,
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
        raise RuntimeError(f"giving up on {url} after 4 attempts")

    def daily_master(self, d: date) -> Optional[list[Filing]]:
        resp = self.get(daily_index_url(d), ok404=True, ok403=True)
        if resp is None:
            log.debug("no daily index for %s (weekend/holiday/not yet published)", d)
            return None
        return parse_master_index(resp.text)

    def primary_document(self, cik: int, accession: str) -> Optional[str]:
        """Resolve a filing's primary document name via the submissions API."""
        if cik not in self._submissions_cache:
            try:
                resp = self.get(SUBMISSIONS_URL.format(cik=cik))
                self._submissions_cache[cik] = resp.json() if resp else None
            except Exception as exc:  # noqa: BLE001 - any failure means "use fallback"
                log.warning("submissions lookup failed for CIK %d: %s", cik, exc)
                self._submissions_cache[cik] = None
        subs = self._submissions_cache[cik]
        if not subs:
            return None
        try:
            recent = subs["filings"]["recent"]
            idx = recent["accessionNumber"].index(accession)
            return recent["primaryDocument"][idx] or None
        except (KeyError, ValueError):
            return None


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------


def download_filing(
    client: EdgarClient, filing: Filing, out_root: Path, force: bool = False
) -> str:
    """Fetch one filing's primary document; returns fetched/skipped status."""
    dest_dir = out_root / "raw" / form_dir_name(filing.form) / filing.accession
    meta_path = dest_dir / "metadata.json"
    if meta_path.exists() and not force:
        log.info("skip   %-5s %s (already fetched)", filing.form, filing.accession)
        return "skipped"

    # Prefer the raw primary document; an xsl-prefixed name is also tried
    # verbatim in case the unprefixed path doesn't exist.
    candidates = []
    primary = client.primary_document(filing.cik, filing.accession)
    if primary:
        stripped = strip_xsl(primary)
        candidates.append(stripped)
        if stripped != primary:
            candidates.append(primary)

    resp = None
    doc_name = ""
    source_url = ""
    for candidate in candidates:
        url = primary_doc_url(filing.cik, filing.accession, candidate)
        resp = client.get(url, ok404=True)
        if resp is not None:
            doc_name = candidate.rsplit("/", 1)[-1]
            source_url = url
            break
    if resp is None:
        source_url = full_txt_url(filing.txt_path)
        log.warning(
            "no primary document resolved for %s; falling back to full submission %s",
            filing.accession, source_url,
        )
        resp = client.get(source_url)
        doc_name = f"{filing.accession}.txt"

    content = resp.content
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / doc_name).write_bytes(content)
    metadata = {
        "accession": filing.accession,
        "cik": filing.cik,
        "company": filing.company,
        "form": filing.form,
        "filed": filing.filed,
        "document": doc_name,
        "source_url": source_url,
        "full_txt_url": full_txt_url(filing.txt_path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log.info(
        "fetch  %-5s %s  %s  (%s, %.1f KB)",
        filing.form, filing.accession, filing.company, doc_name, len(content) / 1024,
    )
    return "fetched"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch recent 10-K and Form 4 filings from SEC EDGAR."
    )
    parser.add_argument("--num-10k", type=int, default=10,
                        help="number of recent 10-K filings to fetch (default 10)")
    parser.add_argument("--num-form4", type=int, default=25,
                        help="number of recent Form 4 filings to fetch (default 25)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data",
                        help="output directory (default <repo>/data)")
    parser.add_argument("--max-days-back", type=int, default=60,
                        help="how many calendar days to walk back (default 60)")
    parser.add_argument("--rps", type=float, default=DEFAULT_RPS,
                        help=f"max requests/sec, capped at {SEC_MAX_RPS:.0f} (default {DEFAULT_RPS:.0f})")
    parser.add_argument("--user-agent", default=None,
                        help="overrides the EDGAR_USER_AGENT environment variable")
    parser.add_argument("--force", action="store_true",
                        help="refetch filings that are already on disk")
    args = parser.parse_args(argv if argv is None else list(argv))

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    _load_dotenv()
    user_agent = args.user_agent or os.environ.get("EDGAR_USER_AGENT")
    if not user_agent:
        parser.error(
            "no User-Agent configured. SEC fair-access policy requires one with "
            "contact info. Set EDGAR_USER_AGENT in the repo-root .env (see "
            ".env.example) or pass --user-agent \"Your Name you@example.com\"."
        )

    client = EdgarClient(user_agent, rps=args.rps)
    wanted = {"10-K": args.num_10k, "4": args.num_form4}
    log.info("collecting %d 10-K and %d Form 4 filings, walking back from %s",
             args.num_10k, args.num_form4, date.today())
    selected = collect_recent(
        client.daily_master, wanted, date.today(), args.max_days_back
    )
    if not any(selected.values()):
        log.error(
            "no filings found at all — EDGAR may be rejecting requests. Check that "
            "your User-Agent includes a name and contact email per SEC fair-access "
            "policy (current: %r).", user_agent,
        )
        return 1

    stats: dict[str, Counter] = {form: Counter() for form in wanted}
    for form, filings in selected.items():
        for filing in filings:
            try:
                stats[form][download_filing(client, filing, args.out, args.force)] += 1
            except Exception:
                log.exception("failed to fetch %s %s", form, filing.accession)
                stats[form]["failed"] += 1

    for form, n in wanted.items():
        c = stats[form]
        log.info(
            "%-6s requested %d: fetched %d, skipped %d (already present), failed %d",
            form, n, c["fetched"], c["skipped"], c["failed"],
        )
    log.info("output root: %s", (args.out / "raw").resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
