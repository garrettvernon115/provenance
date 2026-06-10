"""Pack parsed blocks into retrieval-sized chunks with exact source offsets.

The invariant that makes exact-passage citations possible:

    chunk.text == full_text[chunk.char_start : chunk.char_end]

Chunks are built by packing consecutive blocks (joined with "\\n", exactly as
``parsers.assemble`` joined them) up to a target size, carrying a small tail
of trailing blocks into the next chunk as overlap. Oversized single blocks
are split at word boundaries. Sizes are character-based; ~4 chars ≈ 1 token,
so the 1500-char default keeps passages comfortably inside a cross-encoder's
512-token pair budget later on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from parsers import Block

DEFAULT_TARGET_CHARS = 1500
DEFAULT_OVERLAP_CHARS = 200
DEFAULT_MAX_CHARS = 2200  # hard cap; single blocks above this are split
MIN_TAIL_CHARS = 300      # a smaller trailing chunk is merged into the previous one


@dataclass
class Chunk:
    text: str
    char_start: int
    char_end: int
    chunk_index: int
    section: Optional[str]

    @property
    def token_estimate(self) -> int:
        return math.ceil(len(self.text) / 4)


def split_oversized_block(block: Block, max_chars: int) -> list[Block]:
    """Split one block into ≤max_chars pieces at word boundaries, offsets exact."""
    pieces: list[Block] = []
    text = block.text
    offset = 0
    while len(text) - offset > max_chars:
        cut = text.rfind(" ", offset + 1, offset + max_chars)
        if cut <= offset:
            cut = offset + max_chars  # no space to break at; hard cut
        pieces.append(
            Block(text[offset:cut], block.char_start + offset,
                  block.char_start + cut, block.section)
        )
        offset = cut + 1 if text[cut:cut + 1] == " " else cut
    pieces.append(
        Block(text[offset:], block.char_start + offset, block.char_end,
              block.section)
    )
    return pieces


def _joined_size(blocks: list[Block]) -> int:
    return sum(len(b.text) for b in blocks) + max(0, len(blocks) - 1)


def chunk_blocks(
    full_text: str,
    blocks: list[Block],
    target_chars: int = DEFAULT_TARGET_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[Chunk]:
    normalized: list[Block] = []
    for block in blocks:
        if not block.text:
            continue
        if len(block.text) > max_chars:
            normalized.extend(split_oversized_block(block, max_chars))
        else:
            normalized.append(block)

    chunks: list[Chunk] = []

    def emit(group: list[Block], section: Optional[str]) -> None:
        for b in group:
            if full_text[b.char_start:b.char_end] != b.text:
                raise ValueError(
                    "block text does not match its full_text slice — "
                    "parser offsets are broken"
                )
        start, end = group[0].char_start, group[-1].char_end
        chunks.append(Chunk(full_text[start:end], start, end, len(chunks), section))

    current: list[Block] = []
    current_section: Optional[str] = None
    for block in normalized:
        if current and _joined_size(current) + 1 + len(block.text) > target_chars:
            emit(current, current_section)
            # carry a small tail of whole blocks into the next chunk as overlap
            tail: list[Block] = []
            for prev in reversed(current):
                if _joined_size(tail) + len(prev.text) + (1 if tail else 0) > overlap_chars:
                    break
                tail.insert(0, prev)
            current = tail
            current_section = block.section
        elif not current:
            current_section = block.section
        current.append(block)

    if current:
        # merge an undersized trailing chunk into the previous one when possible
        if (
            chunks
            and _joined_size(current) < MIN_TAIL_CHARS
            and current[-1].char_end - chunks[-1].char_start <= max_chars + overlap_chars
        ):
            last = chunks.pop()
            start, end = last.char_start, current[-1].char_end
            chunks.append(
                Chunk(full_text[start:end], start, end, len(chunks), last.section)
            )
        else:
            emit(current, current_section)

    return chunks
