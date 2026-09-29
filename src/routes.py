"""HTTP routes.

Python port of ``src/routes.js``. The two API paths are a 1:1 match with the
original Hapi server:

* ``POST /catalog/ingest``
* ``POST /catalog/query``

Payload and response shapes are identical, including:

* ``ingest`` takes a **bare JSON array** (the original used ``Joi.array()``),
  not an object wrapper.
* Validation failures return **400** with an ``{"error": "..."}`` envelope, the
  way Hapi maps a Joi rejection — not FastAPI's default 422.
* A query that arrives before any ingest returns **409**.
* Error messages are the original strings from ``catalog-service.js``.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("catalog.routes")


# --------------------------------------------------------------------------- #
# Pydantic schemas — the ``Joi.object`` equivalents from routes.js
# --------------------------------------------------------------------------- #

class CatalogEntry(BaseModel):
    """``catalogEntry`` in routes.js.

    ``id`` is declared optional on purpose: the original validates its presence
    inside the service (``Each entry must have an id``), and we want that exact
    message rather than a Pydantic type error.
    """

    id: Optional[str] = Field(None, examples=["shipping"])
    text: Optional[str] = Field(None, examples=["Shipping takes 3 days."])
    question: Optional[str] = Field(None, examples=["How long is shipping?"])
    answer: Optional[str] = Field(None, examples=["Shipping takes 3 days."])
    title: Optional[str] = Field(None, examples=["Shipping"])
    content: Optional[str] = Field(None, examples=["Orders ship within 3 business days."])


class IngestResponse(BaseModel):
    """``ingestResponse`` in routes.js."""

    ingested: bool = Field(True, examples=[True])
    count: int = Field(..., examples=[1])


class QueryPayload(BaseModel):
    """``queryPayload`` in routes.js (``min(1)`` is enforced in the handler so
    the failure surfaces as a 400 ``{"error"}`` like the original)."""

    query: str = Field(..., examples=["How long does shipping take?"])


class QueryResponse(BaseModel):
    """``queryResponse`` in routes.js."""

    matched: bool = Field(..., examples=[True])
    answer: str = Field(..., examples=["Shipping takes 3 days."])
    score: float = Field(..., examples=[0.94])
    degraded: bool = Field(False, examples=[False])


class ErrorResponse(BaseModel):
    """``errorResponse`` in routes.js."""

    error: str = Field(..., examples=["Catalog has not been ingested"])


# --------------------------------------------------------------------------- #
# Route registration
# --------------------------------------------------------------------------- #

def register_catalog_routes(
    app: FastAPI,
    catalog_service: Any,
    logger_: Optional[logging.Logger] = None,
) -> None:
    """Attach the two catalog routes to ``app`` (port of ``registerCatalogRoutes``)."""
    log = logger_ or logger

    @app.post(
        "/catalog/ingest",
        response_model=IngestResponse,
        responses={400: {"model": ErrorResponse}},
        tags=["api"],
        summary="Replace the in-memory catalog with searchable entries.",
    )
    async def ingest(entries: List[CatalogEntry]) -> Any:
        """Body is a bare JSON array of catalog entries."""
        try:
            result = catalog_service.ingest([entry.model_dump() for entry in entries])
            return {"ingested": True, **result}
        except Exception as error:
            log.error("ingest failed: %s", error)
            # A Response instance bypasses response_model validation, so the
            # error envelope from routes.js survives intact.
            return JSONResponse(status_code=400, content={"error": str(error)})

    @app.post(
        "/catalog/query",
        response_model=QueryResponse,
        responses={400: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
        tags=["api"],
        summary="Find the best catalog answer for a natural-language query.",
    )
    async def query(payload: QueryPayload) -> Any:
        if not isinstance(payload.query, str) or not payload.query.strip():
            return JSONResponse(status_code=400, content={"error": "query must be a non-empty string"})
        try:
            return catalog_service.query(payload.query)
        except Exception as error:
            message = str(error)
            status = 409 if message == "Catalog has not been ingested" else 400
            log.error("query failed (status=%d): %s", status, message)
            return JSONResponse(status_code=status, content={"error": message})
