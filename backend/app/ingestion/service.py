"""
Document ingestion service.

Orchestrates the full pipeline for a single uploaded file:
  parse → chunk → embed → upsert (Qdrant) → save metadata (SQLite)

Write serialization:
  Writes to the same Qdrant collection are serialized via _qdrant_write_lock
  to prevent race conditions from concurrent uploads. This is a single-user
  app, not multi-tenant — the lock is a resource guard, not auth isolation.

Progress tracking:
  The service updates the Document.status field during ingestion so the UI
  can poll and show progress: pending → ingesting → ready (or error).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.models import Document, DocumentChunk
from app.ingestion.chunker import StructureAwareChunker
from app.ingestion.embedder import get_embedder
from app.ingestion.parsers import parse_document
from app.retrieval.qdrant import get_qdrant_store

logger = logging.getLogger(__name__)

# Serialise all Qdrant writes — prevents races from two simultaneous uploads
_qdrant_write_lock = asyncio.Lock()


async def ingest_document(
    document: Document,
    file_path: Path,
    db: AsyncSession,
) -> Document:
    """
    Full ingestion pipeline for a document.
    Updates document.status throughout; caller should await and handle errors.
    """
    settings = get_settings()
    chunker = StructureAwareChunker(
        chunk_size_tokens=settings.chunk_size_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    embedder = get_embedder()
    qdrant = get_qdrant_store()

    try:
        # ── 1. Parse ──────────────────────────────────────────────────────
        document.status = "ingesting"
        await db.commit()
        logger.info("Parsing %s", file_path.name)

        parse_result = parse_document(file_path)
        document.has_ocr_pages = parse_result.has_ocr_pages
        document.ocr_pages = parse_result.ocr_pages

        # ── 2. Chunk ──────────────────────────────────────────────────────
        logger.info("Chunking %s (%d elements)", file_path.name, len(parse_result.elements))
        chunks = chunker.chunk(parse_result, document.id)

        if not chunks:
            raise ValueError("No chunks produced — document may be empty or unreadable")

        # ── 3. Embed (dense vectors via Ollama) ───────────────────────────
        logger.info("Embedding %d chunks for %s", len(chunks), file_path.name)
        texts = [c.text for c in chunks]
        dense_vectors = await embedder.embed_batch(texts)

        if len(dense_vectors) != len(chunks):
            raise ValueError(
                f"Embedding count mismatch: {len(dense_vectors)} vectors for {len(chunks)} chunks"
            )

        # ── 4. Upsert (dense + sparse) to Qdrant — serialized ────────────
        logger.info("Upserting %d chunks to Qdrant for %s", len(chunks), file_path.name)
        async with _qdrant_write_lock:
            await qdrant.ensure_collection()
            await qdrant.upsert_chunks(chunks, dense_vectors)

        # ── 5. Save chunk metadata to app DB ─────────────────────────────
        for chunk in chunks:
            db_chunk = DocumentChunk(
                id=chunk.chunk_id,
                document_id=document.id,
                chunk_index=chunk.chunk_index,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_heading=chunk.section_heading,
                text_preview=chunk.text[:500],
                token_count=chunk.token_count,
                is_ocr=chunk.is_ocr,
                qdrant_point_id=chunk.chunk_id,
            )
            db.add(db_chunk)

        document.chunk_count = len(chunks)
        document.status = "ready"
        await db.commit()

        logger.info(
            "Ingestion complete: %s — %d chunks, OCR pages: %s",
            file_path.name,
            len(chunks),
            parse_result.ocr_pages or "none",
        )
        return document

    except Exception as exc:
        logger.error("Ingestion failed for %s: %s", file_path.name, exc)
        document.status = "error"
        document.error_message = str(exc)
        await db.commit()
        raise


async def delete_document(document: Document, db: AsyncSession) -> None:
    """Remove a document's chunks from Qdrant and its records from the app DB."""
    qdrant = get_qdrant_store()

    async with _qdrant_write_lock:
        await qdrant.delete_document_chunks(document.id)

    # Remove uploaded file
    settings = get_settings()
    file_path = Path(settings.upload_dir) / document.filename
    if file_path.exists():
        file_path.unlink()

    # Cascading delete in DB (DocumentChunk records deleted by FK cascade)
    await db.delete(document)
    await db.commit()
    logger.info("Deleted document %s (%s)", document.id, document.original_filename)


async def reindex_document(document: Document, db: AsyncSession) -> Document:
    """Delete and re-ingest a document (e.g. after settings change)."""
    settings = get_settings()
    file_path = Path(settings.upload_dir) / document.filename

    if not file_path.exists():
        raise FileNotFoundError(f"Uploaded file not found: {file_path}")

    # Remove existing Qdrant chunks
    qdrant = get_qdrant_store()
    async with _qdrant_write_lock:
        await qdrant.delete_document_chunks(document.id)

    # Delete existing DB chunk records
    from sqlalchemy import delete
    from app.db.models import DocumentChunk
    await db.execute(
        delete(DocumentChunk).where(DocumentChunk.document_id == document.id)
    )
    document.chunk_count = 0
    document.status = "pending"
    await db.commit()

    return await ingest_document(document, file_path, db)
