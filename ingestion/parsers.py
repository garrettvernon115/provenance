"""Parsers: filing documents → canonical text + offset-tracked blocks.

Each parser returns a :class:`ParsedDocument` whose ``full_text`` is the
canonical extracted text and whose ``blocks`` are contiguous slices of it
(``block.text == full_text[block.char_start:block.char_end]``). The chunker
preserves that property, which is what makes exact-passage citations work.

Supported sources:
- 10-K (and other HTML filings): iXBRL-aware HTML extraction with
  Item-section labelling.
- Form 4 (ownership XML): rendered to a compact, readable summary that keeps
  the source's literal values.
- PDF: text layer via pypdf; scanned PDFs fall back to OCR when the optional
  pytesseract + pypdfium2 stack is installed, otherwise fail with a clear
  error.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from bs4 import BeautifulSoup

log = logging.getLogger("ingest.parsers")


class ParserError(Exception):
    """Raised when a document cannot be parsed into usable text."""


@dataclass
class Block:
    text: str
    char_start: int
    char_end: int
    section: Optional[str] = None


@dataclass
class ParsedDocument:
    full_text: str
    blocks: list[Block]
    parser: str
    extra: dict = field(default_factory=dict)


_WS_RE = re.compile(r"\s+")


def normalize_ws(text: str) -> str:
    return _WS_RE.sub(" ", text.replace("\xa0", " ")).strip()


def assemble(texts: Sequence[str], sections: Sequence[Optional[str]], parser: str) -> ParsedDocument:
    """Join normalized block texts with "\\n" and compute exact offsets."""
    blocks: list[Block] = []
    parts: list[str] = []
    pos = 0
    for text, section in zip(texts, sections):
        if not text:
            continue
        blocks.append(Block(text, pos, pos + len(text), section))
        parts.append(text)
        pos += len(text) + 1  # the joining "\n"
    return ParsedDocument("\n".join(parts), blocks, parser)


# --------------------------------------------------------------------------
# 10-K / generic EDGAR HTML
# --------------------------------------------------------------------------

# Block-level elements: a boundary marker goes after each so inline markup
# (spans, ix:* tags) never fragments a sentence.
_BLOCK_TAGS = [
    "p", "div", "li", "tr", "table", "h1", "h2", "h3", "h4", "h5", "h6",
    "section", "blockquote", "pre",
]
_SENTINEL = ""  # private-use char; never appears in filings
_DISPLAY_NONE_RE = re.compile(r"display\s*:\s*none", re.I)
_ITEM_RE = re.compile(r"^item\s+(\d{1,2}[a-c]?)\b", re.I)
_MAX_HEADING_LEN = 150  # an "Item 1A." line longer than this is prose, not a heading
# page-break furniture between sections: a bare page number and/or the
# "Table of Contents" backlink that 10-Ks repeat on every page
_PAGE_ARTIFACT_RE = re.compile(r"^(?:\d{1,4}|table of contents)$", re.I)


def parse_html(markup: str) -> ParsedDocument:
    """Extract block-structured text from an EDGAR HTML filing (iXBRL-aware)."""
    soup = BeautifulSoup(markup, "html.parser")

    for tag in soup(["script", "style", "head", "meta", "link", "title"]):
        tag.decompose()
    # iXBRL machinery: the hidden header block and anything styled invisible
    # carry tagging metadata, not document text.
    for tag in soup.find_all(lambda t: t.name in ("ix:header", "ix:hidden")):
        tag.decompose()
    for tag in soup.find_all(style=_DISPLAY_NONE_RE):
        tag.decompose()

    for tag in soup.find_all(["br", "hr"]):
        tag.insert_after(_SENTINEL)
    for tag in soup.find_all(_BLOCK_TAGS):
        tag.append(_SENTINEL)

    raw = soup.get_text(" ")

    texts: list[str] = []
    sections: list[Optional[str]] = []
    current_section: Optional[str] = None
    for raw_block in raw.split(_SENTINEL):
        text = normalize_ws(raw_block)
        if not text or _PAGE_ARTIFACT_RE.match(text):
            continue
        match = _ITEM_RE.match(text)
        if match and len(text) <= _MAX_HEADING_LEN:
            current_section = f"Item {match.group(1).upper()}"
        texts.append(text)
        sections.append(current_section)

    doc = assemble(texts, sections, parser="html")
    if not doc.full_text.strip():
        raise ParserError("no text extracted from HTML document")
    return doc


# --------------------------------------------------------------------------
# Form 4 (ownership XML)
# --------------------------------------------------------------------------

_TXN_CODES = {
    "P": "open-market purchase",
    "S": "open-market sale",
    "A": "grant or award",
    "D": "disposition to the issuer",
    "F": "tax-withholding share surrender",
    "M": "derivative exercise or conversion",
    "C": "conversion of derivative",
    "X": "option exercise",
    "G": "gift",
    "J": "other transaction",
    "W": "acquired or disposed by will",
}


def _val(el: Optional[ET.Element], path: str) -> Optional[str]:
    """findtext that prefers EDGAR's ``<x><value>…</value></x>`` wrapping."""
    if el is None:
        return None
    for candidate in (f"{path}/value", path):
        text = el.findtext(candidate)
        if text and text.strip():
            return text.strip()
    return None


def _flag(el: Optional[ET.Element], path: str) -> bool:
    value = _val(el, path)
    return value is not None and value.lower() in ("1", "true")


def _owner_roles(rel: Optional[ET.Element]) -> str:
    roles = []
    if _flag(rel, "isDirector"):
        roles.append("Director")
    if _flag(rel, "isOfficer"):
        title = _val(rel, "officerTitle")
        roles.append(f"Officer ({title})" if title else "Officer")
    if _flag(rel, "isTenPercentOwner"):
        roles.append("10% owner")
    if _flag(rel, "isOther"):
        other = _val(rel, "otherText")
        roles.append(other or "Other")
    return ", ".join(roles) if roles else "relationship not stated"


def _code_label(code: Optional[str]) -> str:
    if not code:
        return "transaction"
    gloss = _TXN_CODES.get(code.upper())
    return f"Code {code} ({gloss})" if gloss else f"Code {code}"


def _verb(acquired_disposed: Optional[str]) -> str:
    return {"A": "acquired", "D": "disposed of"}.get(
        (acquired_disposed or "").upper(), "transacted"
    )


def _txn_line(txn: ET.Element, derivative: bool) -> str:
    title = _val(txn, "securityTitle") or "securities"
    when = _val(txn, "transactionDate") or "date n/a"
    code = txn.findtext("transactionCoding/transactionCode")
    shares = _val(txn, "transactionAmounts/transactionShares")
    price = _val(txn, "transactionAmounts/transactionPricePerShare")
    acq_disp = _val(txn, "transactionAmounts/transactionAcquiredDisposedCode")
    owned_after = _val(txn, "postTransactionAmounts/sharesOwnedFollowingTransaction")
    ownership = (_val(txn, "ownershipNature/directOrIndirectOwnership") or "").upper()
    ownership_label = {"D": "direct ownership", "I": "indirect ownership"}.get(ownership, "")

    parts = [f"- {when}: {_code_label(code)} — {_verb(acq_disp)}"]
    parts.append(f" {shares}" if shares else "")
    parts.append(f" shares of {title}" if not derivative else f" {title}")
    if price:
        parts.append(f" at ${price} per share")
    if derivative:
        strike = _val(txn, "conversionOrExercisePrice")
        if strike:
            parts.append(f", conversion/exercise price ${strike}")
        underlying = txn.find("underlyingSecurity")
        u_title = _val(underlying, "underlyingSecurityTitle")
        u_shares = _val(underlying, "underlyingSecurityShares")
        if u_title:
            parts.append(f", underlying {u_shares or '?'} shares of {u_title}")
    if owned_after:
        parts.append(f"; {owned_after} shares owned following transaction")
    if ownership_label:
        parts.append(f" ({ownership_label})")
    return "".join(parts)


def parse_form4_xml(data: bytes) -> ParsedDocument:
    """Render an ownership document (Form 3/4/5 XML) as readable text lines."""
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ParserError(f"invalid ownership XML: {exc}") from exc

    lines: list[str] = []
    doc_type = (root.findtext("documentType") or "4").strip()
    lines.append(f"Form {doc_type} — Statement of Changes in Beneficial Ownership")

    issuer = root.find("issuer")
    if issuer is not None:
        name = issuer.findtext("issuerName") or "unknown issuer"
        symbol = issuer.findtext("issuerTradingSymbol")
        cik = issuer.findtext("issuerCik")
        symbol_part = f" ({symbol.strip()})" if symbol and symbol.strip() else ""
        cik_part = f", CIK {int(cik)}" if cik and cik.strip().isdigit() else ""
        lines.append(f"Issuer: {name.strip()}{symbol_part}{cik_part}")

    for owner in root.findall("reportingOwner"):
        name = owner.findtext("reportingOwnerId/rptOwnerName") or "unknown owner"
        roles = _owner_roles(owner.find("reportingOwnerRelationship"))
        lines.append(f"Reporting owner: {normalize_ws(name)} — {roles}")

    period = root.findtext("periodOfReport")
    if period and period.strip():
        lines.append(f"Period of report: {period.strip()}")

    for table_tag, txn_tag, hold_tag, label, derivative in (
        ("nonDerivativeTable", "nonDerivativeTransaction", "nonDerivativeHolding",
         "Non-derivative", False),
        ("derivativeTable", "derivativeTransaction", "derivativeHolding",
         "Derivative", True),
    ):
        table = root.find(table_tag)
        if table is None:
            continue
        transactions = table.findall(txn_tag)
        if transactions:
            lines.append(f"{label} transactions:")
            lines.extend(_txn_line(t, derivative) for t in transactions)
        for holding in table.findall(hold_tag):
            title = _val(holding, "securityTitle") or "securities"
            shares = _val(holding, "postTransactionAmounts/sharesOwnedFollowingTransaction")
            ownership = (_val(holding, "ownershipNature/directOrIndirectOwnership") or "").upper()
            nature = {"D": "direct", "I": "indirect"}.get(ownership, "unspecified")
            lines.append(
                f"- Holding: {shares or '?'} shares of {title} ({nature} ownership)"
            )

    footnotes = root.find("footnotes")
    if footnotes is not None:
        for fn in footnotes:
            if fn.text and fn.text.strip():
                lines.append(f"Footnote {fn.get('id', '')}: {normalize_ws(fn.text)}")
    remarks = root.findtext("remarks")
    if remarks and remarks.strip():
        lines.append(f"Remarks: {normalize_ws(remarks)}")

    return assemble(lines, [f"Form {doc_type}"] * len(lines), parser="form4-xml")


# --------------------------------------------------------------------------
# PDF (text layer, with optional OCR fallback)
# --------------------------------------------------------------------------

_MIN_CHARS_PER_PAGE = 100  # below this average, assume a scanned document


def _ocr_pdf(path: Path) -> list[str]:
    try:
        import pypdfium2
        import pytesseract
    except ImportError as exc:
        raise ParserError(
            f"{path.name} looks like a scanned PDF; OCR fallback needs "
            "pytesseract + pypdfium2 (see ingestion/requirements.txt) and the "
            "Tesseract binary on PATH"
        ) from exc
    log.info("running OCR fallback on %s", path.name)
    pdf = pypdfium2.PdfDocument(str(path))
    try:
        return [
            pytesseract.image_to_string(page.render(scale=2.0).to_pil())
            for page in pdf
        ]
    finally:
        pdf.close()


def parse_pdf(path: Path) -> ParsedDocument:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    page_texts = [page.extract_text() or "" for page in reader.pages]
    total_chars = sum(len(t.strip()) for t in page_texts)
    if not page_texts or total_chars < _MIN_CHARS_PER_PAGE * len(page_texts):
        page_texts = _ocr_pdf(path)

    texts: list[str] = []
    sections: list[Optional[str]] = []
    for page_number, page_text in enumerate(page_texts, start=1):
        for paragraph in re.split(r"\n\s*\n", page_text):
            text = normalize_ws(paragraph)
            if text:
                texts.append(text)
                sections.append(f"page {page_number}")
    doc = assemble(texts, sections, parser="pdf")
    if not doc.full_text.strip():
        raise ParserError(f"no text extracted from {path.name}")
    return doc


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

_OWNERSHIP_FORMS = {"3", "3/A", "4", "4/A", "5", "5/A"}


def parse_filing(path: Path, form: str) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix in (".htm", ".html"):
        return parse_html(path.read_text(encoding="utf-8", errors="replace"))
    if suffix == ".xml":
        if form in _OWNERSHIP_FORMS:
            return parse_form4_xml(path.read_bytes())
        raise ParserError(f"no XML parser for form {form!r}")
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix == ".txt":
        raise ParserError(
            "full-submission .txt files are not supported yet; "
            "Phase 1 parses primary documents only"
        )
    raise ParserError(f"no parser for {path.name}")
