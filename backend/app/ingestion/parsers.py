"""
Document parsing layer.

Parsing strategy:
  - PDF / DOCX: Docling (layout-aware, table-preserving). Falls back to pypdf/
    python-docx only if Docling raises.
  - TXT / MD / CSV: simple text read — no layout to preserve.
  - OCR fallback: after extraction, pages with near-zero usable text are routed
    through Tesseract. OCR'd pages are flagged in metadata so the UI can show
    "(OCR-extracted, may contain errors)" on citations.
  - Multi-page tables: Docling's structure detection is used to identify tables
    that span page breaks. These are merged into a single chunk with a page
    range (pages: [4, 5]) rather than split at the page boundary.

CLOUD-DEPENDENCY AUDIT: No external calls. Docling runs fully locally. Tesseract
is a local binary. No model downloads happen at parse time (Docling's models are
bundled / pre-downloaded into the container image).
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Threshold: if a PDF page has fewer than this many alphanumeric characters
# after extraction, treat it as a scanned/image-only page and run OCR.
_OCR_TRIGGER_CHARS = 50


@dataclass
class ParsedElement:
    """
    A structural element extracted from a document.
    Could be a paragraph, heading, table, list, or code block.
    Elements are later assembled into chunks by the chunker.
    """
    element_type: str               # "text" | "table" | "heading" | "list" | "code"
    text: str                       # plain-text rendering of the element
    page_start: int | None = None   # 1-indexed
    page_end: int | None = None     # same as page_start for single-page elements
    section_heading: str | None = None
    is_ocr: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def page_range(self) -> tuple[int | None, int | None]:
        return self.page_start, self.page_end


@dataclass
class ParseResult:
    elements: list[ParsedElement]
    filename: str
    file_type: str
    page_count: int
    has_ocr_pages: bool
    ocr_pages: list[int]           # 1-indexed page numbers that required OCR


def _run_tesseract(image_bytes: bytes) -> str:
    """Run Tesseract on a raw image. Returns extracted text."""
    try:
        from PIL import Image
        import pytesseract

        img = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(img, config="--psm 6")
    except Exception as exc:
        logger.warning("Tesseract OCR failed: %s", exc)
        return ""


def _is_text_sparse(text: str) -> bool:
    """True if the text is too sparse to be real extracted content."""
    return len([c for c in text if c.isalnum()]) < _OCR_TRIGGER_CHARS


def parse_pdf_with_docling(path: Path) -> ParseResult:
    """
    Parse a PDF using Docling with table-structure preservation.
    Falls back to pypdf + Tesseract for pages with no extractable text.

    Multi-page table merging:
    Docling marks table elements with page provenance. When we detect that a
    table's bounding box starts on one page and ends on another (or that two
    consecutive table elements share the same caption/context), we merge their
    text into a single ParsedElement with page_start and page_end set to the
    full span. This prevents the chunker from splitting a table in half.
    """
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    ocr_pages: list[int] = []
    elements: list[ParsedElement] = []
    page_count = 0

    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False          # we handle OCR ourselves
        pipeline_options.do_table_structure = True

        converter = DocumentConverter()
        result = converter.convert(str(path))
        doc = result.document
        page_count = len(doc.pages) if hasattr(doc, "pages") else 0

        current_heading: str | None = None
        pending_table: ParsedElement | None = None

        for item in doc.iterate_items():
            item_type = type(item).__name__.lower()

            # ── Determine page span ──────────────────────────────────────
            page_start: int | None = None
            page_end: int | None = None
            if hasattr(item, "prov") and item.prov:
                pages = sorted({p.page_no for p in item.prov})
                page_start = pages[0] if pages else None
                page_end = pages[-1] if pages else None

            # ── Extract text ─────────────────────────────────────────────
            if hasattr(item, "text"):
                text = item.text or ""
            elif hasattr(item, "export_to_markdown"):
                text = item.export_to_markdown()
            else:
                continue

            if not text.strip():
                continue

            # ── Classify element type ────────────────────────────────────
            if "sectionheader" in item_type or "heading" in item_type:
                current_heading = text.strip()
                elements.append(ParsedElement(
                    element_type="heading",
                    text=text,
                    page_start=page_start,
                    page_end=page_end,
                    section_heading=current_heading,
                ))
                # Flush any pending table before a new section
                if pending_table is not None:
                    elements.append(pending_table)
                    pending_table = None

            elif "table" in item_type:
                # ── Multi-page table merging ─────────────────────────────
                # If there's a pending table and its page_end is adjacent to
                # this table's page_start, merge them (same table, page break).
                if (
                    pending_table is not None
                    and pending_table.page_end is not None
                    and page_start is not None
                    and page_start <= (pending_table.page_end + 1)
                ):
                    # Merge: extend the pending table's text and update page_end
                    pending_table.text += "\n" + text
                    pending_table.page_end = max(
                        pending_table.page_end or 0, page_end or 0
                    ) or None
                else:
                    # Flush previous pending table (if any) and start a new one
                    if pending_table is not None:
                        elements.append(pending_table)
                    pending_table = ParsedElement(
                        element_type="table",
                        text=text,
                        page_start=page_start,
                        page_end=page_end,
                        section_heading=current_heading,
                    )

            else:
                # Flush pending table before a non-table element
                if pending_table is not None:
                    elements.append(pending_table)
                    pending_table = None
                elements.append(ParsedElement(
                    element_type="text",
                    text=text,
                    page_start=page_start,
                    page_end=page_end,
                    section_heading=current_heading,
                ))

        # Flush any remaining pending table
        if pending_table is not None:
            elements.append(pending_table)

    except Exception as docling_exc:
        logger.warning(
            "Docling failed for %s (%s), falling back to pypdf", path.name, docling_exc
        )
        elements, ocr_pages, page_count = _parse_pdf_pypdf_fallback(path)
        return ParseResult(
            elements=elements,
            filename=path.name,
            file_type="pdf",
            page_count=page_count,
            has_ocr_pages=bool(ocr_pages),
            ocr_pages=ocr_pages,
        )

    # ── OCR fallback for pages with sparse text ───────────────────────────
    # Check each page in the elements; if a page has no/sparse text elements,
    # run Tesseract and inject the result as an OCR element.
    pages_with_text: set[int] = set()
    for el in elements:
        if el.page_start:
            for pg in range(el.page_start, (el.page_end or el.page_start) + 1):
                if not _is_text_sparse(el.text):
                    pages_with_text.add(pg)

    for pg_num in range(1, page_count + 1):
        if pg_num not in pages_with_text:
            # Attempt to rasterize and OCR this page
            ocr_text = _ocr_pdf_page(path, pg_num)
            if ocr_text.strip():
                ocr_pages.append(pg_num)
                elements.append(ParsedElement(
                    element_type="text",
                    text=ocr_text,
                    page_start=pg_num,
                    page_end=pg_num,
                    is_ocr=True,
                ))

    return ParseResult(
        elements=elements,
        filename=path.name,
        file_type="pdf",
        page_count=page_count,
        has_ocr_pages=bool(ocr_pages),
        ocr_pages=ocr_pages,
    )


def _ocr_pdf_page(path: Path, page_num: int) -> str:
    """Rasterize a single PDF page and run Tesseract on it."""
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(str(path), first_page=page_num, last_page=page_num, dpi=200)
        if not images:
            return ""
        img_bytes = io.BytesIO()
        images[0].save(img_bytes, format="PNG")
        return _run_tesseract(img_bytes.getvalue())
    except Exception as exc:
        logger.warning("OCR rasterization failed for page %d: %s", page_num, exc)
        return ""


def _parse_pdf_pypdf_fallback(
    path: Path,
) -> tuple[list[ParsedElement], list[int], int]:
    """Pure pypdf extraction, used when Docling fails."""
    from pypdf import PdfReader

    elements: list[ParsedElement] = []
    ocr_pages: list[int] = []

    reader = PdfReader(str(path))
    page_count = len(reader.pages)

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if _is_text_sparse(text):
            ocr_text = _ocr_pdf_page(path, i)
            if ocr_text.strip():
                ocr_pages.append(i)
                elements.append(ParsedElement(
                    element_type="text",
                    text=ocr_text,
                    page_start=i,
                    page_end=i,
                    is_ocr=True,
                ))
        else:
            elements.append(ParsedElement(
                element_type="text",
                text=text,
                page_start=i,
                page_end=i,
            ))

    return elements, ocr_pages, page_count


def parse_docx(path: Path) -> ParseResult:
    """
    Parse a DOCX file using Docling (layout + heading-aware).
    Falls back to python-docx if Docling raises.
    """
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(str(path))
        doc = result.document

        elements: list[ParsedElement] = []
        current_heading: str | None = None

        for item in doc.iterate_items():
            item_type = type(item).__name__.lower()
            text = getattr(item, "text", None) or (
                item.export_to_markdown() if hasattr(item, "export_to_markdown") else ""
            )
            if not text.strip():
                continue

            if "sectionheader" in item_type or "heading" in item_type:
                current_heading = text.strip()
                elements.append(ParsedElement(
                    element_type="heading",
                    text=text,
                    section_heading=current_heading,
                ))
            elif "table" in item_type:
                elements.append(ParsedElement(
                    element_type="table",
                    text=text,
                    section_heading=current_heading,
                ))
            else:
                elements.append(ParsedElement(
                    element_type="text",
                    text=text,
                    section_heading=current_heading,
                ))

        return ParseResult(
            elements=elements,
            filename=path.name,
            file_type="docx",
            page_count=0,  # DOCX doesn't have fixed pages
            has_ocr_pages=False,
            ocr_pages=[],
        )

    except Exception as exc:
        logger.warning("Docling DOCX parse failed (%s), falling back to python-docx", exc)
        return _parse_docx_fallback(path)


def _parse_docx_fallback(path: Path) -> ParseResult:
    from docx import Document
    doc = Document(str(path))
    elements: list[ParsedElement] = []
    current_heading: str | None = None

    for para in doc.paragraphs:
        if not para.text.strip():
            continue
        if para.style.name.startswith("Heading"):
            current_heading = para.text.strip()
            elements.append(ParsedElement(
                element_type="heading",
                text=para.text,
                section_heading=current_heading,
            ))
        else:
            elements.append(ParsedElement(
                element_type="text",
                text=para.text,
                section_heading=current_heading,
            ))

    return ParseResult(
        elements=elements,
        filename=path.name,
        file_type="docx",
        page_count=0,
        has_ocr_pages=False,
        ocr_pages=[],
    )


def parse_text(path: Path, file_type: str) -> ParseResult:
    """Parse TXT, MD, or CSV files. No layout to preserve — simple read."""
    if file_type == "csv":
        return _parse_csv(path)

    text = path.read_text(encoding="utf-8", errors="replace")
    return ParseResult(
        elements=[ParsedElement(element_type="text", text=text)],
        filename=path.name,
        file_type=file_type,
        page_count=1,
        has_ocr_pages=False,
        ocr_pages=[],
    )


def _parse_csv(path: Path) -> ParseResult:
    """Convert CSV rows to a readable table-like text block."""
    rows: list[list[str]] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return ParseResult(
            elements=[],
            filename=path.name,
            file_type="csv",
            page_count=1,
            has_ocr_pages=False,
            ocr_pages=[],
        )

    # Render as markdown table for better LLM readability
    header = rows[0]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows[1:]:
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(padded) + " |")

    text = "\n".join(lines)
    return ParseResult(
        elements=[ParsedElement(element_type="table", text=text)],
        filename=path.name,
        file_type="csv",
        page_count=1,
        has_ocr_pages=False,
        ocr_pages=[],
    )


def parse_document(path: Path) -> ParseResult:
    """
    Dispatch to the appropriate parser based on file extension.
    This is the single entry point used by the ingestion pipeline.
    """
    suffix = path.suffix.lower().lstrip(".")
    dispatch = {
        "pdf": lambda: parse_pdf_with_docling(path),
        "docx": lambda: parse_docx(path),
        "txt": lambda: parse_text(path, "txt"),
        "md": lambda: parse_text(path, "md"),
        "csv": lambda: parse_text(path, "csv"),
    }
    if suffix not in dispatch:
        raise ValueError(f"Unsupported file type: .{suffix}")
    return dispatch[suffix]()
