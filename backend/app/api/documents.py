"""
Documents API router.
Handles file upload, listing, deletion, and re-indexing.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.models import Document
from app.db.session import get_db
from app.ingestion.service import delete_document, ingest_document, reindex_document

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "csv", "md"}
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB


def _ext(filename: str) -> str:
    return Path(filename).suffix.lower().lstrip(".")


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload and ingest a document. Ingestion runs as a background task so the
    response is immediate; the client polls GET /documents/{id} for status.
    """
    if not file.filename:
        raise HTTPException(400, "Filename is required")

    ext = _ext(file.filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400, f"Unsupported file type '.{ext}'. Allowed: {ALLOWED_EXTENSIONS}"
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(413, "File exceeds 100 MB limit")

    settings = get_settings()
    os.makedirs(settings.upload_dir, exist_ok=True)

    # Store with a UUID prefix to avoid collisions
    stored_filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = Path(settings.upload_dir) / stored_filename
    file_path.write_bytes(content)

    doc = Document(
        id=str(uuid.uuid4()),
        filename=stored_filename,
        original_filename=file.filename,
        file_type=ext,
        file_size_bytes=len(content),
        status="pending",
        qdrant_collection=settings.qdrant_collection,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    # Run ingestion in background so upload returns immediately
    background_tasks.add_task(_ingest_background, doc.id, file_path)

    return {
        "id": doc.id,
        "filename": doc.original_filename,
        "status": doc.status,
        "message": "Upload received; ingestion started",
    }


async def _ingest_background(doc_id: str, file_path: Path) -> None:
    """Background task: open a fresh DB session and run ingestion."""
    from app.db.session import get_session_factory

    factory = get_session_factory()
    async with factory() as db:
        result = await db.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if doc is None:
            return
        try:
            await ingest_document(doc, file_path, db)
        except Exception:
            pass  # status already set to "error" inside ingest_document


@router.get("/")
async def list_documents(db: AsyncSession = Depends(get_db)):
    """List all documents with their ingestion status."""
    result = await db.execute(select(Document).order_by(Document.upload_timestamp.desc()))
    docs = result.scalars().all()
    return [
        {
            "id": d.id,
            "filename": d.original_filename,
            "file_type": d.file_type,
            "file_size_bytes": d.file_size_bytes,
            "upload_timestamp": d.upload_timestamp.isoformat(),
            "chunk_count": d.chunk_count,
            "status": d.status,
            "error_message": d.error_message,
            "has_ocr_pages": d.has_ocr_pages,
            "ocr_pages": d.ocr_pages,
        }
        for d in docs
    ]


@router.get("/{doc_id}")
async def get_document(doc_id: str, db: AsyncSession = Depends(get_db)):
    """Get a single document's status and metadata."""
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(404, "Document not found")
    return {
        "id": doc.id,
        "filename": doc.original_filename,
        "file_type": doc.file_type,
        "file_size_bytes": doc.file_size_bytes,
        "upload_timestamp": doc.upload_timestamp.isoformat(),
        "chunk_count": doc.chunk_count,
        "status": doc.status,
        "error_message": doc.error_message,
        "has_ocr_pages": doc.has_ocr_pages,
        "ocr_pages": doc.ocr_pages,
    }


@router.delete("/{doc_id}")
async def delete_document_endpoint(doc_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a document and all its chunks from Qdrant and the app DB."""
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(404, "Document not found")
    await delete_document(doc, db)
    return {"message": f"Document '{doc.original_filename}' deleted"}


@router.post("/{doc_id}/reindex")
async def reindex_document_endpoint(
    doc_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Re-ingest a document (e.g. after chunk size change)."""
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(404, "Document not found")

    settings = get_settings()
    file_path = Path(settings.upload_dir) / doc.filename
    if not file_path.exists():
        raise HTTPException(404, "Original file not found on disk — cannot reindex")

    background_tasks.add_task(_reindex_background, doc.id)
    return {"message": "Re-indexing started", "id": doc.id}


async def _reindex_background(doc_id: str) -> None:
    from app.db.session import get_session_factory
    factory = get_session_factory()
    async with factory() as db:
        result = await db.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if doc:
            try:
                await reindex_document(doc, db)
            except Exception:
                pass
