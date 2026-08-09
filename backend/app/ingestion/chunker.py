"""
Structure-aware chunker.

Key behaviors:
  1. Never split mid-table or mid-heading — table elements are always kept
     whole (or truncated only if they vastly exceed the chunk size).
  2. Heading elements are prepended to the following text chunk so context
     isn't lost (a chunk starting with "Revenue: $4.2M" is useless without
     knowing it's under "Q3 Financial Results").
  3. Multi-page table elements (page_start != page_end) are kept together and
     their page range is preserved in chunk metadata.
  4. OCR flag is propagated: if any element in a chunk was OCR'd, the whole
     chunk is flagged is_ocr=True and the UI shows a warning on citations.

Chunking algorithm:
  - Accumulate elements into a running buffer (tracked in tokens).
  - When adding the next element would exceed chunk_size, emit the current
    buffer as a chunk, then start a new buffer seeded with the overlap window
    (last overlap_tokens worth of text from the previous chunk).
  - Tables are never split across chunks (they get their own chunk if needed).

Token counting:
  - We use a whitespace-based approximation (len(text.split()) * 1.3) rather
    than loading a full tokenizer — it's fast, good enough for chunking, and
    avoids a runtime dependency on a specific tokenizer that might not match
    the embedding model anyway.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from app.ingestion.parsers import ParsedElement, ParseResult


def _approx_tokens(text: str) -> int:
    """Whitespace-split token count with a 1.3x scaling factor for subwords."""
    return max(1, int(len(text.split()) * 1.3))


@dataclass
class Chunk:
    """A single chunk ready for embedding and storage in Qdrant."""
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    page_start: int | None
    page_end: int | None
    section_heading: str | None
    is_ocr: bool
    token_count: int
    element_types: list[str]   # which element types contributed to this chunk
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def page_label(self) -> str:
        if self.page_start is None:
            return "unknown"
        if self.page_end is None or self.page_end == self.page_start:
            return str(self.page_start)
        return f"{self.page_start}–{self.page_end}"


class StructureAwareChunker:
    def __init__(
        self,
        chunk_size_tokens: int = 800,
        overlap_tokens: int = 100,
    ) -> None:
        self.chunk_size = chunk_size_tokens
        self.overlap = overlap_tokens

    def chunk(self, parse_result: ParseResult, document_id: str) -> list[Chunk]:
        elements = parse_result.elements
        chunks: list[Chunk] = []
        chunk_index = 0

        # Running accumulation state
        buffer_texts: list[str] = []
        buffer_tokens: int = 0
        buffer_page_start: int | None = None
        buffer_page_end: int | None = None
        buffer_section: str | None = None
        buffer_is_ocr: bool = False
        buffer_types: list[str] = []

        def _flush_buffer(overlap_text: str = "") -> None:
            nonlocal chunk_index, buffer_texts, buffer_tokens
            nonlocal buffer_page_start, buffer_page_end
            nonlocal buffer_section, buffer_is_ocr, buffer_types

            combined = "\n\n".join(buffer_texts).strip()
            if not combined:
                return

            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                chunk_index=chunk_index,
                text=combined,
                page_start=buffer_page_start,
                page_end=buffer_page_end,
                section_heading=buffer_section,
                is_ocr=buffer_is_ocr,
                token_count=_approx_tokens(combined),
                element_types=list(set(buffer_types)),
            ))
            chunk_index += 1

            # Seed new buffer with overlap window
            buffer_texts = [overlap_text] if overlap_text.strip() else []
            buffer_tokens = _approx_tokens(overlap_text) if overlap_text.strip() else 0
            buffer_page_start = None
            buffer_page_end = None
            buffer_section = None
            buffer_is_ocr = False
            buffer_types = []

        def _update_page_range(pg_start: int | None, pg_end: int | None) -> None:
            nonlocal buffer_page_start, buffer_page_end
            if pg_start is not None:
                buffer_page_start = min(
                    buffer_page_start or pg_start, pg_start
                )
            if pg_end is not None:
                buffer_page_end = max(
                    buffer_page_end or pg_end, pg_end
                )

        def _overlap_text() -> str:
            """Extract the last `overlap_tokens` worth of text from the buffer."""
            full = "\n\n".join(buffer_texts)
            words = full.split()
            overlap_word_count = int(self.overlap / 1.3)
            tail = words[-overlap_word_count:] if len(words) > overlap_word_count else words
            return " ".join(tail)

        pending_heading: str | None = None

        for el in elements:
            # Headings are stored as context for the next content element
            if el.element_type == "heading":
                pending_heading = el.text.strip()
                buffer_section = pending_heading
                continue

            element_tokens = _approx_tokens(el.text)

            # ── Tables: never split ───────────────────────────────────────
            if el.element_type == "table":
                # Flush current buffer before the table
                if buffer_texts:
                    overlap = _overlap_text()
                    _flush_buffer(overlap)

                # Prepend the pending heading to the table for context
                table_text = el.text
                if pending_heading:
                    table_text = f"{pending_heading}\n\n{table_text}"
                    pending_heading = None

                # If the table itself exceeds chunk_size, we still keep it
                # whole — truncation of a table is worse than a large chunk.
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    chunk_index=chunk_index,
                    text=table_text.strip(),
                    page_start=el.page_start,
                    page_end=el.page_end,
                    section_heading=el.section_heading or buffer_section,
                    is_ocr=el.is_ocr,
                    token_count=_approx_tokens(table_text),
                    element_types=["table"],
                ))
                chunk_index += 1
                continue

            # ── Regular text: accumulate until size limit ────────────────
            # Prepend pending heading to first element after it
            if pending_heading and not buffer_texts:
                buffer_texts.append(pending_heading)
                buffer_tokens += _approx_tokens(pending_heading)
                buffer_section = pending_heading
                pending_heading = None

            if buffer_tokens + element_tokens > self.chunk_size and buffer_texts:
                overlap = _overlap_text()
                _flush_buffer(overlap)

            buffer_texts.append(el.text)
            buffer_tokens += element_tokens
            buffer_is_ocr = buffer_is_ocr or el.is_ocr
            buffer_types.append(el.element_type)
            if el.section_heading:
                buffer_section = el.section_heading
            _update_page_range(el.page_start, el.page_end)

        # Flush remainder
        if buffer_texts:
            _flush_buffer()

        return chunks
