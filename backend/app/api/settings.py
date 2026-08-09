"""
Settings API router.
Reads/writes settings to the app metadata DB and the in-memory config cache.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.models import AppSettings
from app.db.session import get_db

router = APIRouter(prefix="/settings", tags=["settings"])


class SettingsUpdate(BaseModel):
    ollama_generation_model: str | None = None
    ollama_embed_model: str | None = None
    reranker_device: str | None = None
    reranker_top_k: int | None = None
    reranker_initial_top_k: int | None = None
    chunk_size_tokens: int | None = None
    chunk_overlap_tokens: int | None = None
    hybrid_dense_weight: float | None = None
    confidence_threshold: float | None = None
    llm_temperature: float | None = None
    llm_max_tokens: int | None = None
    max_concurrent_llm_calls: int | None = None
    router_mode: str | None = None


@router.get("/")
async def get_all_settings(db: AsyncSession = Depends(get_db)):
    """Return current effective settings (merged env + DB overrides)."""
    s = get_settings()
    return {
        "ollama_base_url": s.ollama_base_url,
        "ollama_generation_model": s.ollama_generation_model,
        "ollama_embed_model": s.ollama_embed_model,
        "reranker_model": s.reranker_model,
        "reranker_device": s.reranker_device,
        "reranker_top_k": s.reranker_top_k,
        "reranker_initial_top_k": s.reranker_initial_top_k,
        "chunk_size_tokens": s.chunk_size_tokens,
        "chunk_overlap_tokens": s.chunk_overlap_tokens,
        "hybrid_dense_weight": s.hybrid_dense_weight,
        "confidence_threshold": s.confidence_threshold,
        "llm_temperature": s.llm_temperature,
        "llm_max_tokens": s.llm_max_tokens,
        "max_concurrent_llm_calls": s.max_concurrent_llm_calls,
        "router_mode": s.router_mode,
    }


@router.patch("/")
async def update_settings(req: SettingsUpdate, db: AsyncSession = Depends(get_db)):
    """
    Update settings. Changes are persisted to the DB and take effect
    on the next request (the settings cache is invalidated).
    """
    updates = req.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(400, "No settings provided")

    for key, value in updates.items():
        result = await db.execute(select(AppSettings).where(AppSettings.key == key))
        existing = result.scalar_one_or_none()
        if existing:
            existing.value = str(value)
        else:
            db.add(AppSettings(key=key, value=str(value)))

    await db.commit()

    # Invalidate the settings cache so next get_settings() reads fresh values
    get_settings.cache_clear()

    return {"message": "Settings updated", "updated": list(updates.keys())}


@router.get("/ollama/models")
async def list_ollama_models():
    """
    Query the local Ollama instance for available models.
    Used to populate model dropdowns in the Settings UI.
    """
    import httpx
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [
                {
                    "name": m["name"],
                    "size_gb": round(m.get("size", 0) / 1e9, 1),
                    "modified_at": m.get("modified_at", ""),
                }
                for m in data.get("models", [])
            ]
            return {"models": models}
    except Exception as exc:
        raise HTTPException(503, f"Cannot reach Ollama: {exc}")


@router.get("/health")
async def health_check():
    """
    Comprehensive startup health check.
    Returns status of Ollama, Qdrant, and the reranker (including GPU detection).
    """
    from app.core.pipeline import get_pipeline
    pipeline = get_pipeline()
    return await pipeline.health_check()
