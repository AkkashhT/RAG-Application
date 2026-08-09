"""
Chat API router — SSE streaming responses.

Each chat message streams back:
  data: {"type": "status", "content": "..."}
  data: {"type": "token", "content": "..."}
  data: {"type": "sql", "content": "..."}        ← always shown when SQL is used
  data: {"type": "citations", "content": [...]}
  data: {"type": "done", "content": {...}}
  data: {"type": "error", "content": "..."}

The "waiting for model" status is emitted when the concurrency semaphore is
held — the frontend should show a spinner rather than a blank screen.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.pipeline import get_pipeline
from app.db.models import ChatMessage, ChatSession, QueryLog
from app.db.session import get_db

router = APIRouter(prefix="/chat", tags=["chat"])


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class NewSessionRequest(BaseModel):
    title: str = "New conversation"
    scoped_document_ids: list[str] | None = None


class MessageRequest(BaseModel):
    content: str
    scoped_document_ids: list[str] | None = None


# ── Session management ────────────────────────────────────────────────────────

@router.post("/sessions")
async def create_session(req: NewSessionRequest, db: AsyncSession = Depends(get_db)):
    session = ChatSession(
        id=str(uuid.uuid4()),
        title=req.title,
        scoped_document_ids=req.scoped_document_ids,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return _session_dict(session)


@router.get("/sessions")
async def list_sessions(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ChatSession).order_by(ChatSession.updated_at.desc())
    )
    sessions = result.scalars().all()
    return [_session_dict(s) for s in sessions]


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)):
    session = await _get_session_or_404(session_id, db)
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
    )
    messages = result.scalars().all()
    return {
        **_session_dict(session),
        "messages": [_message_dict(m) for m in messages],
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, db: AsyncSession = Depends(get_db)):
    session = await _get_session_or_404(session_id, db)
    await db.delete(session)
    await db.commit()
    return {"message": "Session deleted"}


@router.patch("/sessions/{session_id}")
async def update_session(
    session_id: str,
    req: NewSessionRequest,
    db: AsyncSession = Depends(get_db),
):
    session = await _get_session_or_404(session_id, db)
    session.title = req.title
    if req.scoped_document_ids is not None:
        session.scoped_document_ids = req.scoped_document_ids
    await db.commit()
    return _session_dict(session)


# ── Message streaming ─────────────────────────────────────────────────────────

@router.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: str,
    req: MessageRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Send a message and stream the response via SSE.
    The full answer, citations, and SQL (if used) are persisted to the DB
    after streaming completes.
    """
    session = await _get_session_or_404(session_id, db)

    # Persist the user message immediately
    user_msg = ChatMessage(
        id=str(uuid.uuid4()),
        session_id=session_id,
        role="user",
        content=req.content,
    )
    db.add(user_msg)
    await db.commit()

    # Determine effective document scope
    doc_ids = req.scoped_document_ids or session.scoped_document_ids

    pipeline = get_pipeline()

    async def event_stream():
        full_answer = []
        citations = []
        sql_query = None
        router_decision = None
        top_rerank_score = 0.0
        latency_ms = 0.0

        try:
            async for event in pipeline.run(
                question=req.content,
                session_id=session_id,
                document_ids=doc_ids,
            ):
                event_type = event["type"]
                content = event["content"]

                if event_type == "token":
                    full_answer.append(content)
                elif event_type == "citations":
                    citations = content
                elif event_type == "sql":
                    sql_query = content
                elif event_type == "done":
                    router_decision = content.get("router_decision")
                    top_rerank_score = content.get("top_rerank_score", 0.0)
                    latency_ms = content.get("latency_ms", 0.0)

                yield f"data: {json.dumps(event)}\n\n"

        except Exception as exc:
            error_event = {"type": "error", "content": str(exc)}
            yield f"data: {json.dumps(error_event)}\n\n"
        finally:
            # Persist assistant message after stream completes
            answer_text = "".join(full_answer)
            if answer_text:
                from app.db.session import get_session_factory
                factory = get_session_factory()
                async with factory() as save_db:
                    asst_msg = ChatMessage(
                        id=str(uuid.uuid4()),
                        session_id=session_id,
                        role="assistant",
                        content=answer_text,
                        citations=citations,
                        sql_query=sql_query,
                        router_decision=router_decision,
                        top_rerank_score=top_rerank_score,
                    )
                    save_db.add(asst_msg)
                    # Update session updated_at
                    result = await save_db.execute(
                        select(ChatSession).where(ChatSession.id == session_id)
                    )
                    s = result.scalar_one_or_none()
                    if s:
                        from datetime import datetime, timezone
                        s.updated_at = datetime.now(timezone.utc)
                    await save_db.commit()

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_session_or_404(session_id: str, db: AsyncSession) -> ChatSession:
    result = await db.execute(
        select(ChatSession).where(ChatSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(404, f"Session '{session_id}' not found")
    return session


def _session_dict(s: ChatSession) -> dict[str, Any]:
    return {
        "id": s.id,
        "title": s.title,
        "created_at": s.created_at.isoformat(),
        "updated_at": s.updated_at.isoformat(),
        "scoped_document_ids": s.scoped_document_ids,
    }


def _message_dict(m: ChatMessage) -> dict[str, Any]:
    return {
        "id": m.id,
        "role": m.role,
        "content": m.content,
        "created_at": m.created_at.isoformat(),
        "citations": m.citations,
        "sql_query": m.sql_query,
        "router_decision": m.router_decision,
        "top_rerank_score": m.top_rerank_score,
    }
