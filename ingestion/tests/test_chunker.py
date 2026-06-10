"""Chunker tests: exact offsets, overlap, splitting, section carry."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chunker
import parsers


def make_doc(texts, sections=None):
    sections = sections or [None] * len(texts)
    return parsers.assemble(texts, sections, parser="test")


PARAGRAPHS = [
    f"Paragraph {i:02d}: " + "lorem ipsum dolor sit amet consectetur " * 3
    for i in range(30)
]


def test_chunks_are_exact_slices_of_full_text():
    doc = make_doc(PARAGRAPHS)
    chunks = chunker.chunk_blocks(doc.full_text, doc.blocks)
    assert len(chunks) > 1
    for c in chunks:
        assert doc.full_text[c.char_start:c.char_end] == c.text
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # full coverage: first chunk starts at 0, last ends at the end
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(doc.full_text)


def test_consecutive_chunks_overlap():
    doc = make_doc(PARAGRAPHS)
    chunks = chunker.chunk_blocks(doc.full_text, doc.blocks)
    for a, b in zip(chunks, chunks[1:]):
        assert b.char_start < a.char_end  # overlap carried
        assert b.char_start > a.char_start  # but still advancing


def test_chunk_sizes_bounded():
    doc = make_doc(PARAGRAPHS)
    chunks = chunker.chunk_blocks(doc.full_text, doc.blocks)
    for c in chunks:
        assert len(c.text) <= chunker.DEFAULT_MAX_CHARS + chunker.DEFAULT_OVERLAP_CHARS


def test_oversized_block_is_split_with_exact_offsets():
    big = "word " * 1200  # ~6000 chars, no block boundaries
    doc = make_doc([big.strip()])
    chunks = chunker.chunk_blocks(doc.full_text, doc.blocks)
    assert len(chunks) > 1
    for c in chunks:
        assert doc.full_text[c.char_start:c.char_end] == c.text
        assert len(c.text) <= chunker.DEFAULT_MAX_CHARS + chunker.DEFAULT_OVERLAP_CHARS


def test_hard_split_without_spaces():
    doc = make_doc(["x" * 5000])
    chunks = chunker.chunk_blocks(doc.full_text, doc.blocks)
    assert len(chunks) > 1
    for c in chunks:
        assert doc.full_text[c.char_start:c.char_end] == c.text


def test_section_carries_to_chunks():
    sections = ["Item 1"] * 15 + ["Item 1A"] * 15
    doc = make_doc(PARAGRAPHS, sections)
    chunks = chunker.chunk_blocks(doc.full_text, doc.blocks)
    assert chunks[0].section == "Item 1"
    assert chunks[-1].section == "Item 1A"


def test_tiny_trailing_chunk_is_merged():
    texts = ["a" * 700, "b" * 700, "c" * 50]  # residue far below MIN_TAIL_CHARS
    doc = make_doc(texts)
    chunks = chunker.chunk_blocks(doc.full_text, doc.blocks)
    assert chunks[-1].char_end == len(doc.full_text)
    assert len(chunks[-1].text) >= chunker.MIN_TAIL_CHARS


def test_single_small_doc_is_one_chunk():
    doc = make_doc(["Short filing summary.", "One more line."])
    chunks = chunker.chunk_blocks(doc.full_text, doc.blocks)
    assert len(chunks) == 1
    assert chunks[0].text == doc.full_text
