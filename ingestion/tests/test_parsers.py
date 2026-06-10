"""Parser tests — fixture documents only, no network."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import parsers
from fixtures import FORM4_XML, TENK_HTML


def _assert_offsets_exact(doc: parsers.ParsedDocument) -> None:
    for block in doc.blocks:
        assert doc.full_text[block.char_start:block.char_end] == block.text


# ---------------------------------------------------------------- 10-K HTML


def test_html_drops_hidden_and_nontext_content():
    doc = parsers.parse_html(TENK_HTML)
    assert "HIDDENXBRL-METADATA" not in doc.full_text
    assert "INVISIBLE-TAGGING-JUNK" not in doc.full_text
    assert "console.log" not in doc.full_text
    assert "color: red" not in doc.full_text
    # page-break furniture is filtered
    assert "Table of Contents" not in doc.full_text
    assert not any(b.text == "47" for b in doc.blocks)


def test_html_keeps_text_with_inline_ixbrl_and_breaks():
    doc = parsers.parse_html(TENK_HTML)
    assert "1,234 million in fiscal 2025" in doc.full_text
    # <br/> creates a block boundary
    assert "First paragraph line." in doc.full_text
    assert "Second line after a break." in doc.full_text
    # table rows survive as row-blocks
    assert any("Net income" in b.text and "567" in b.text for b in doc.blocks)


def test_html_assigns_item_sections():
    doc = parsers.parse_html(TENK_HTML)
    risk = next(b for b in doc.blocks if "widget demand" in b.text)
    assert risk.section == "Item 1A"
    business = next(b for b in doc.blocks if "designs and manufactures" in b.text)
    assert business.section == "Item 1"
    heading = next(b for b in doc.blocks if b.text.startswith("Item 1A"))
    assert heading.section == "Item 1A"


def test_html_offsets_are_exact():
    _assert_offsets_exact(parsers.parse_html(TENK_HTML))


def test_html_with_no_text_raises():
    with pytest.raises(parsers.ParserError):
        parsers.parse_html("<html><body><script>x()</script></body></html>")


# ---------------------------------------------------------------- Form 4 XML


def test_form4_renders_summary():
    doc = parsers.parse_form4_xml(FORM4_XML.encode())
    text = doc.full_text
    assert "Issuer: PG&E Corp (PCG), CIK 1004980" in text
    assert "Reporting owner: DOE JANE — Officer (EVP, General Counsel)" in text
    assert "Period of report: 2026-06-08" in text
    assert "Code S (open-market sale)" in text
    assert "disposed of 1500 shares of Common Stock at $12.34 per share" in text
    assert "50000 shares owned following transaction (direct ownership)" in text
    assert "Footnote F1: Sale executed pursuant to a Rule 10b5-1 trading plan." in text


def test_form4_sections_and_offsets():
    doc = parsers.parse_form4_xml(FORM4_XML.encode())
    assert all(b.section == "Form 4" for b in doc.blocks)
    _assert_offsets_exact(doc)


def test_form4_handles_missing_tables():
    minimal = (
        "<ownershipDocument><documentType>4</documentType>"
        "<issuer><issuerName>MiniCo</issuerName></issuer>"
        "</ownershipDocument>"
    )
    doc = parsers.parse_form4_xml(minimal.encode())
    assert "MiniCo" in doc.full_text


def test_form4_invalid_xml_raises():
    with pytest.raises(parsers.ParserError):
        parsers.parse_form4_xml(b"<not-xml")


# ---------------------------------------------------------------- dispatch


def test_dispatch_rejects_full_submission_txt(tmp_path):
    txt = tmp_path / "0000000000-26-000001.txt"
    txt.write_text("raw submission", encoding="utf-8")
    with pytest.raises(parsers.ParserError):
        parsers.parse_filing(txt, "10-K")
