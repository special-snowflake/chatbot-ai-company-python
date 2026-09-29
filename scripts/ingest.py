"""CLI ingestion script.

Python port of ``scripts/ingest.js``. Run with::

    python -m scripts.ingest

Loads ``config.catalog_file``, builds the index, and writes it to
``config.catalog_index_file`` so the server can skip re-embedding at startup.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.catalog_service import CatalogService  # noqa: E402
from src.config import config  # noqa: E402
from src.embeddings import create_embedding_provider  # noqa: E402
from src.laya_engine import create_laya_provider  # noqa: E402
from src.source_loader import load_source_entries  # noqa: E402


def main() -> int:
    """Ingest the source catalog and persist the resulting index."""
    entries = load_source_entries(config.catalog_file)
    service = CatalogService(
        embedder=create_embedding_provider(config.embedding_model),
        llm=create_laya_provider(config.laya_model, config.llm_timeout_ms),
        threshold=config.similarity_threshold,
        top_k=config.top_k,
    )
    result = service.ingest(entries)

    os.makedirs(os.path.dirname(config.catalog_index_file) or ".", exist_ok=True)
    serialisable = [
        {**entry, "words": sorted(entry["words"])}
        for entry in service.index
    ]
    with open(config.catalog_index_file, "w", encoding="utf-8") as handle:
        json.dump(serialisable, handle, ensure_ascii=False, indent=2)

    print(f"Ingested {result['count']} entries from {config.catalog_file}")
    print(f"Wrote index to {config.catalog_index_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
