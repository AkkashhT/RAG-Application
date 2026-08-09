"""
Database connector API router.

Safety notes surfaced to the UI:
- Always recommends creating a dedicated read-only DB role.
- Never returns the raw connection string to the frontend.
- EXPLAIN + statement-type rejection are enforced in db/connector.py.
- Generated SQL is always returned alongside answers (see chat.py).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.connector import get_connector, set_connector
from app.db.session import get_db

router = APIRouter(prefix="/database", tags=["database"])


class ConnectRequest(BaseModel):
    connection_string: str


class TestRequest(BaseModel):
    connection_string: str


@router.post("/connect")
async def connect_database(req: ConnectRequest):
    """
    Set the active SQL database connection.
    The connection string is stored only in memory (and optionally .env).
    It is NEVER returned to the frontend or written to logs.
    """
    connector = set_connector(req.connection_string)
    result = await connector.test_connection()
    if not result["ok"]:
        raise HTTPException(400, f"Connection failed: {result['error']}")
    return {
        "ok": True,
        "dialect": result["dialect"],
        "message": "Connected successfully",
        # Return redacted URL (password replaced with ***) for display
        "url_display": result["url"],
    }


@router.post("/test")
async def test_connection(req: TestRequest):
    """
    Test a connection string without making it the active connection.
    Useful for the Settings UI to validate before saving.
    """
    from app.db.connector import SQLConnector
    tmp = SQLConnector(req.connection_string)
    result = await tmp.test_connection()
    await tmp.dispose()
    if not result["ok"]:
        raise HTTPException(400, f"Connection test failed: {result['error']}")
    return {
        "ok": True,
        "dialect": result["dialect"],
        "url_display": result["url"],
    }


@router.get("/schema")
async def get_schema():
    """
    Return the introspected schema of the connected database.
    Used by the Settings UI to show what tables are available.
    """
    connector = get_connector()
    if connector is None:
        raise HTTPException(400, "No database connected. POST /database/connect first.")
    schema = await connector.introspect_schema()
    return schema


@router.get("/status")
async def get_status():
    """Return connection status without revealing credentials."""
    connector = get_connector()
    if connector is None:
        return {"connected": False}
    result = await connector.test_connection()
    return {
        "connected": result["ok"],
        "dialect": result.get("dialect"),
        "url_display": result.get("url"),
        "error": result.get("error"),
    }


@router.delete("/disconnect")
async def disconnect_database():
    """Disconnect the active database connection."""
    from app.db import connector as connector_module
    if connector_module._connector is not None:
        await connector_module._connector.dispose()
        connector_module._connector = None
    return {"message": "Disconnected"}
