"""Unit tests for fetch_edgar's pure helpers — no network access."""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fetch_edgar as fe

# Realistic slice of a daily master index: preamble, pipe-delimited header,
# separator, rows. The two Form 4 rows share one accession (issuer + reporting
# owner each get a row for the same filing).
FIXTURE_INDEX = """Description:           Daily Index of EDGAR Dissemination Feed
Last Data Received:    June 9, 2026
Comments:              webmaster@sec.gov

CIK|Company Name|Form Type|Date Filed|File Name
--------------------------------------------------------------------------------
1000045|ACME FINANCIAL INC|8-K|20260609|edgar/data/1000045/0001000045-26-000031.txt
320193|APPLE INC|10-K|20260609|edgar/data/320193/0000320193-26-000008.txt
1318605|EXAMPLE MOTORS INC|4|20260609|edgar/data/1318605/0001318605-26-000222.txt
4567|DOE JANE|4|20260609|edgar/data/1318605/0001318605-26-000222.txt
1652044|BIGCO HOLDINGS|DEF 14A|20260609|edgar/data/1652044/0001652044-26-000123.txt
not|a|valid|row
"""


def _filing(cik, form, accession, filed="2026-06-08", company="OLDCO INC"):
    return fe.Filing(
        cik=cik,
        company=company,
        form=form,
        filed=filed,
        txt_path=f"edgar/data/{cik}/{accession}.txt",
        accession=accession,
    )


def test_parse_master_index_keeps_data_rows_only():
    filings = fe.parse_master_index(FIXTURE_INDEX)
    assert len(filings) == 5  # header/separator/malformed rows dropped
    apple = next(f for f in filings if f.cik == 320193)
    assert apple.form == "10-K"
    assert apple.company == "APPLE INC"
    assert apple.filed == "2026-06-09"
    assert apple.accession == "0000320193-26-000008"
    assert apple.txt_path == "edgar/data/320193/0000320193-26-000008.txt"


def test_parse_master_index_handles_forms_with_spaces():
    filings = fe.parse_master_index(FIXTURE_INDEX)
    assert any(f.form == "DEF 14A" for f in filings)


def test_filed_iso_normalizes_both_formats():
    assert fe.filed_iso("20260609") == "2026-06-09"
    assert fe.filed_iso("2026-06-09") == "2026-06-09"


def test_accession_from_txt_path():
    assert (
        fe.accession_from_txt_path("edgar/data/320193/0000320193-26-000008.txt")
        == "0000320193-26-000008"
    )


def test_strip_xsl():
    assert fe.strip_xsl("xslF345X05/wk-form4_1.xml") == "wk-form4_1.xml"
    assert fe.strip_xsl("aapl-20251231.htm") == "aapl-20251231.htm"
    assert fe.strip_xsl("sub/dir.htm") == "sub/dir.htm"  # only xsl* is stripped


def test_daily_index_url_quarters():
    assert fe.daily_index_url(date(2026, 6, 9)) == (
        "https://www.sec.gov/Archives/edgar/daily-index/2026/QTR2/master.20260609.idx"
    )
    assert "2026/QTR1/master.20260115.idx" in fe.daily_index_url(date(2026, 1, 15))
    assert "2025/QTR4/master.20251231.idx" in fe.daily_index_url(date(2025, 12, 31))


def test_primary_doc_url_strips_accession_dashes():
    assert fe.primary_doc_url(320193, "0000320193-26-000008", "doc.htm") == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019326000008/doc.htm"
    )


def test_form_dir_name_is_filesystem_safe():
    assert fe.form_dir_name("10-K/A") == "10-K-A"
    assert fe.form_dir_name("DEF 14A") == "DEF_14A"


def test_collect_recent_walks_back_dedupes_and_stops():
    older_10k = _filing(1750, "10-K", "0000001750-26-000050")
    older_4 = _filing(1750, "4", "0000001750-26-000051")
    day_map = {
        date(2026, 6, 10): fe.parse_master_index(FIXTURE_INDEX),
        # 2026-06-09 intentionally absent -> get_day returns None (holiday)
        date(2026, 6, 8): [older_10k, older_4],
    }

    selected = fe.collect_recent(
        day_map.get, {"10-K": 2, "4": 1}, start=date(2026, 6, 10), max_days=5
    )

    # newest-first across days; the second 10-K comes from the older day
    assert [f.accession for f in selected["10-K"]] == [
        "0000320193-26-000008",
        "0000001750-26-000050",
    ]
    # count satisfied on the first day -> older Form 4 never selected
    assert [f.accession for f in selected["4"]] == ["0001318605-26-000222"]


def test_collect_recent_dedupes_shared_accessions_and_reports_shortfall():
    day_map = {date(2026, 6, 10): fe.parse_master_index(FIXTURE_INDEX)}
    selected = fe.collect_recent(
        day_map.get, {"4": 5}, start=date(2026, 6, 10), max_days=1
    )
    # fixture has two Form 4 rows but they share one accession
    assert len(selected["4"]) == 1
