"""
SQLAlchemy models for application metadata.
This is the LOCAL app database (SQLite) — completely separate from any
user-connected business database. Stores chat history, document records,
query logs, and settings.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


# ── Documents ────────────────────────────────────────────────────────────────

class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_type: Mapped[str] = mapped_column(String(16), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    upload_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(
        String(32), default="pending"
    )  # pending | ingesting | ready | error
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    has_ocr_pages: Mapped[bool] = mapped_column(Boolean, default=False)
    # JSON list of page numbers that required OCR fallback
    ocr_pages: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)
    # Qdrant collection this document's chunks live in
    qdrant_collection: Mapped[str] = mapped_column(String(128), default="documents")

    chunks: Mapped[list[DocumentChunk]] = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    # page_start / page_end support multi-page table chunks
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section_heading: Mapped[str | None] = mapped_column(Text, nullable=True)
    text_preview: Mapped[str] = mapped_column(Text, nullable=False)  # first 500 chars
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    is_ocr: Mapped[bool] = mapped_column(Boolean, default=False)
    # The Qdrant point ID for direct lookup
    qdrant_point_id: Mapped[str] = mapped_column(String(36), nullable=False)

    document: Mapped[Document] = relationship("Document", back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_doc_chunk_idx"),
    )


# ── Chat sessions & messages ─────────────────────────────────────────────────

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(String(256), default="New conversation")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )
    # Optional: scope this session to specific document IDs (JSON list of doc IDs)
    scoped_document_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    messages: Mapped[list[ChatMessage]] = relationship(
        "ChatMessage", back_populates="session", cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    # Structured citations attached to assistant messages
    citations: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    # SQL query used (if SQL path was taken)
    sql_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Router decision: "docs" | "sql" | "both" | "not_found"
    router_decision: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # Top reranker score (for debug display)
    top_rerank_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    session: Mapped[ChatSession] = relationship("ChatSession", back_populates="messages")


# ── Query logs (for RAGAS eval and hallucination auditing) ───────────────────

class QueryLog(Base):
    __tablename__ = "query_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    session_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    message_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    router_decision: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # JSON: list of {chunk_id, score, text_preview, source}
    retrieved_chunks: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    # JSON: {reranked: [{chunk_id, score}], top_score: float}
    rerank_results: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    sql_executed: Mapped[str | None] = mapped_column(Text, nullable=True)
    sql_results_preview: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_gate_passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)


# ── Settings snapshot (persists UI-changed settings) ─────────────────────────

class AppSettings(Base):
    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


# ── DB schema cache (for connected user SQL database) ───────────────────────

class DBSchemaCache(Base):
    __tablename__ = "db_schema_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    connection_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    schema_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    introspected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    table_count: Mapped[int] = mapped_column(Integer, default=0)
