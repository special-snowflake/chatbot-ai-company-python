"""Application factory + entry point.

Python port of ``src/server.js``.

Structural parity with the original:

* ``create_server()`` is async and returns a ready-to-serve app.
* Request/response logging mirrors the Hapi ``onRequest`` / ``onPreResponse``
  extensions, including the ``requestId`` and ``durationMs`` fields.
* Swagger UI is served at ``/documentation`` (the ``hapi-swagger``
  ``documentationPath``).
* ``auto_ingest`` loads ``catalog-index.json`` if present, else ingests
  ``./source`` — identical fallback chain to server.js.

The app is exposed as a module-level ``app`` so ``uvicorn src.server:app`` works,
and ``python -m src.server`` runs it on ``config.port`` like ``npm start``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .catalog_service import CatalogService
from .config import answering, config
from .embeddings import create_embedding_provider
from .laya_engine import create_laya_provider
from .routes import register_catalog_routes
from .source_loader import load_source_entries
from .synthesizer import ApiSynthesizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("catalog.server")


async def create_server(
    *,
    embedder: Any = None,
    llm: Any = None,
    logger_: Optional[logging.Logger] = None,
    auto_ingest: bool = True,
) -> FastAPI:
    """Build and configure the FastAPI application."""
    log = logger_ or logger

    app = FastAPI(
        title="Local Catalog Q&A API",
        version="1.0.0",
        description=(
            "Python port of special-snowflake/chatbot-ai-company. The generative "
            "node-llama-cpp answer provider is replaced by the LAYA non-autoregressive "
            "decision engine."
        ),
        docs_url="/documentation",   # hapi-swagger documentationPath
        redoc_url=None,
    )

    # ``routes: { cors: true }`` in the original Hapi server config.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Hapi + Joi answer a schema rejection with 400 and an ``{error}`` body.
    # FastAPI defaults to 422 with a ``{detail: [...]}`` body, which would be a
    # visible contract difference, so remap it here.
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first.get("loc", []) if part != "body")
        message = first.get("msg", "Invalid request payload")
        return JSONResponse(
            status_code=400,
            content={"error": f"{location}: {message}" if location else message},
        )

    # ------------------------------------------------------------------ #
    # Request logging — port of the Hapi onRequest/onPreResponse extensions
    # ------------------------------------------------------------------ #
    @app.middleware("http")
    async def log_requests(request: Request, call_next):  # type: ignore[override]
        started_at = time.time()
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        log.info(
            json.dumps(
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "requestId": request_id,
                    "method": request.method.upper(),
                    "path": request.url.path,
                }
            )
            + " request received"
        )
        response = await call_next(request)
        details = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "requestId": request_id,
            "method": request.method.upper(),
            "path": request.url.path,
            "status": response.status_code,
            "durationMs": int((time.time() - started_at) * 1000),
        }
        if response.status_code >= 400:
            log.error(json.dumps(details) + " request failed")
        else:
            log.info(json.dumps(details) + " response sent")
        return response

    # ------------------------------------------------------------------ #
    # Service wiring
    # ------------------------------------------------------------------ #
    # Only build a synthesizer when an endpoint is actually configured: leaving
    # SYNTHESIZER_BASE_URL empty keeps the service fully local at zero cost.
    synthesizer = None
    if answering.synthesizer_base_url:
        synthesizer = ApiSynthesizer(
            base_url=answering.synthesizer_base_url,
            model=answering.synthesizer_model,
            api_key=answering.synthesizer_api_key,
            timeout_ms=answering.synthesizer_timeout_ms,
        )
        log.info("synthesizer configured: %s", answering.synthesizer_model)

    catalog_service = CatalogService(
        embedder=embedder or create_embedding_provider(config.embedding_model),
        llm=llm or create_laya_provider(config.laya_model, config.llm_timeout_ms),
        threshold=config.similarity_threshold,
        top_k=config.top_k,
        logger_=log,
        tie_epsilon=answering.tie_epsilon,
        max_tie_nodes=answering.max_tie_nodes,
        drop_contentless_nodes=answering.drop_contentless_nodes,
        enable_compound_split=answering.enable_compound_split,
        enable_hazard_escalation=answering.enable_hazard_escalation,
        enable_budget_listing=answering.enable_budget_listing,
        answer_mode=answering.answer_mode,
        query_cache_size=answering.query_cache_size,
        synthesizer=synthesizer,
    )
    register_catalog_routes(app, catalog_service, log)

    if isinstance(catalog_service.embedder, object) and getattr(catalog_service.embedder, "degraded", False):
        log.warning(
            "sentence-transformers unavailable; using deterministic hashed embeddings "
            "(retrieval quality is degraded — install sentence-transformers for parity)"
        )

    if auto_ingest:
        if answering.warm_layap:
            # ``create_laya_provider`` builds the Router, but LAYA downloads its
            # checkpoint lazily on first predict. Warming it here stops the first
            # real query being slower than every later one — at a cost of roughly
            # 1.6 GB resident. Off by default, because the deterministic answering
            # path means most installs never reach LAYA (WARM_LAYAP, README §7).
            await asyncio.to_thread(_warmup, catalog_service)

        # Unlike the original (which throws), a missing/unreadable source leaves
        # the catalog empty and the server still starts. Queries then return the
        # same 409 "Catalog has not been ingested" the JS version returns before
        # its first ingest, so the failure mode is explicit rather than fatal.
        try:
            if os.path.exists(config.catalog_index_file):
                try:
                    with open(config.catalog_index_file, "r", encoding="utf-8") as handle:
                        result = catalog_service.load_index(json.load(handle))
                    log.info(f"catalog index loaded at startup: {config.catalog_index_file} count={result['count']}")
                except Exception as error:
                    log.warning(f"catalog index unavailable ({error}); ingesting source")
                    entries = load_source_entries(config.catalog_file)
                    result = catalog_service.ingest(entries)
                    log.info(f"catalog ingested at startup: {config.catalog_file} count={result['count']}")
            else:
                entries = load_source_entries(config.catalog_file)
                result = catalog_service.ingest(entries)
                log.info(f"catalog ingested at startup: {config.catalog_file} count={result['count']}")
        except Exception as error:
            log.error(
                f"startup ingest failed ({error}); starting with an empty catalog "
                f"(POST /catalog/ingest to populate it)"
            )

    return app


def _warmup(catalog_service: CatalogService) -> None:
    """Force LAYA's checkpoint load so startup time absorbs the download."""
    try:
        catalog_service.llm._predict(
            "warmup",
            {"warmup": {"type": "noul", "instructions": "Is this a warmup call?"}},
        )
        logger.info("LAYAP checkpoint warm (router ready)")
    except Exception as error:  # pragma: no cover - non-fatal
        logger.warning(f"LAYAP warmup failed (non-fatal): {error}")


# Uvicorn entry point: ``uvicorn src.server:app``
app = asyncio.run(create_server())


def main() -> None:
    """``python -m src.server`` — equivalent to ``node src/server.js``."""
    import uvicorn

    print(f"Catalog API listening on http://127.0.0.1:{config.port}")
    uvicorn.run(app, host="127.0.0.1", port=config.port)


if __name__ == "__main__":
    main()
