"""Threshold retest + multi-node inspection + performance monitoring.

Run after lowering SIMILARITY_THRESHOLD to 0.45.

What it reports per query:
  * wall-clock latency
  * process CPU time consumed and peak RSS
  * the retrieval score gate (did it pass?)
  * EVERY context node handed to the LAYA engine (title, score, text)
  * the final answer produced from those nodes

Usage:
    python scripts/threshold_retest.py
"""

from __future__ import annotations

import os
import sys
import time
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.catalog_service import (  # noqa: E402
    CatalogService,
    conversational_intent,
    cosine_similarity,
    lexical_similarity,
    meaningful_words,
)
from src.config import config  # noqa: E402
from src.embeddings import create_embedding_provider  # noqa: E402
from src.laya_engine import LayaEngine  # noqa: E402
from src.source_loader import load_source_entries  # noqa: E402

logging.basicConfig(level=logging.WARNING)


# --------------------------------------------------------------------------- #
# Resource sampling — psutil if present, else /proc + os.times fallback
# --------------------------------------------------------------------------- #

try:
    import psutil  # type: ignore

    _PROC = psutil.Process(os.getpid())

    def rss_mb() -> float:
        return _PROC.memory_info().rss / (1024 * 1024)

except ImportError:  # pragma: no cover
    _PROC = None

    def rss_mb() -> float:
        """Resident set size in MB, read straight from /proc."""
        try:
            with open("/proc/self/statm", "r") as handle:
                pages = int(handle.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        except (OSError, ValueError, IndexError):
            return float("nan")


def cpu_seconds() -> float:
    """Total CPU time (user + system) consumed by this process."""
    times = os.times()
    return times.user + times.system


# --------------------------------------------------------------------------- #
# Capture what the engine is actually fed
# --------------------------------------------------------------------------- #

CAPTURED: list = []


def instrument(service: CatalogService) -> None:
    """Wrap the engine so every context node passed to it is recorded."""
    original = service.llm.answer

    def spy(*, context: str, query: str, context_texts=None, **kwargs):
        CAPTURED.append(
            {
                "context": context,
                "query": query,
                "context_texts": list(context_texts or []),
            }
        )
        return original(
            context=context, query=query, context_texts=context_texts, **kwargs
        )

    service.llm.answer = spy  # type: ignore[assignment]


def main() -> int:
    print("=" * 78)
    print("CONFIG")
    print("=" * 78)
    print(f"  similarity_threshold : {config.similarity_threshold}")
    print(f"  top_k                : {config.top_k}")
    print(f"  embedding_model      : {config.embedding_model}")
    print(f"  laya_model           : {config.laya_model}")

    embedder = create_embedding_provider(config.embedding_model)

    t_boot = time.perf_counter()
    cpu_boot = cpu_seconds()
    llm = LayaEngine(model=config.laya_model)
    service = CatalogService(
        embedder=embedder,
        llm=llm,
        threshold=config.similarity_threshold,
        top_k=config.top_k,
    )
    entries = load_source_entries(config.catalog_file)
    ingest_result = service.ingest(entries)
    boot_s = time.perf_counter() - t_boot
    print(
        f"\n  cold start (embeddings + LAYA + ingest of {ingest_result['count']} chunks): "
        f"{boot_s:.2f}s, cpu {cpu_seconds() - cpu_boot:.2f}s, rss {rss_mb():.0f}MB"
    )

    instrument(service)

    # Group 1: the three that failed at 0.58, plus the two controls.
    # Group 2: deliberately multi-node — these should surface several context
    #          items at once (catalog entry + related FAQ rows).
    queries = [
        ("GATE", "how many days does shipment usually takes?"),
        ("GATE", "what's the cheapest product do you have?"),
        ("GATE", "what is the company about?"),
        ("CONTROL", "halo?"),
        ("CONTROL", "what did you do?"),
        ("MULTI", "tell me about the Glow Bulb A19"),
        ("MULTI", "what smart lighting products do you sell and what do they cost?"),
        ("MULTI", "what are the shipping costs and how long does delivery take?"),
        ("MULTI", "compare the Glow Bulb A19 and the Glow Strip 2M"),
        ("MULTI", "which products have the longest warranty and what is the price?"),
    ]

    results = []
    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)

    for kind, query in queries:
        CAPTURED.clear()
        rss_before = rss_mb()
        cpu_before = cpu_seconds()
        t0 = time.perf_counter()

        try:
            response = service.query(query)
        except Exception as error:  # pragma: no cover
            response = {"matched": False, "answer": f"ERROR: {error}", "score": None}

        latency_ms = (time.perf_counter() - t0) * 1000
        cpu_used = cpu_seconds() - cpu_before
        rss_after = rss_mb()

        # Independent recomputation of the retrieval shortlist for inspection.
        # ``conversational_intent`` may rewrite the query (e.g. "halo?" -> greeting),
        # so mirror the same rewrite the service applies before scoring.
        query_embedding = embedder.embed(query)
        intent = conversational_intent(query)
        knowledge_query = query
        if intent and intent.get("query"):
            knowledge_query = intent["query"]
        query_words = meaningful_words(knowledge_query)
        shortlist = []
        for entry in service.index:
            semantic = cosine_similarity(query_embedding, entry["embedding"])
            lexical = lexical_similarity(query_words, entry["words"])
            shortlist.append(
                {
                    "id": entry["id"],
                    "score": max(semantic, (semantic * 0.7) + (lexical * 0.3)),
                    "semantic": semantic,
                    "lexical": lexical,
                    "text": entry["text"],
                }
            )
        shortlist.sort(key=lambda item: item["score"], reverse=True)

        record = {
            "kind": kind,
            "query": query,
            "latency_ms": latency_ms,
            "cpu_ms": cpu_used * 1000,
            "rss_mb": rss_after,
            "rss_delta_mb": rss_after - rss_before,
            "response": response,
            "nodes_used": len(CAPTURED[0]["context_texts"]) if CAPTURED else 0,
            "shortlist": shortlist[:5],
        }
        results.append(record)

        print(f"\n[{kind}] {query}")
        print(
            f"  latency {latency_ms:8.0f}ms | cpu {cpu_used * 1000:7.0f}ms | "
            f"rss {rss_after:6.1f}MB (Δ{rss_after - rss_before:+.1f})"
        )
        print(
            f"  matched={response.get('matched')} score={response.get('score')} "
            f"degraded={response.get('degraded')} nodes_used={record['nodes_used']}"
        )
        print("  shortlist:")
        for item in shortlist[:4]:
            passed = "PASS" if item["score"] >= config.similarity_threshold else "----"
            print(
                f"    [{passed}] {item['score']:.4f} "
                f"(sem {item['semantic']:.4f} / lex {item['lexical']:.4f}) {item['id']}"
            )
        print(f"  ANSWER: {response.get('answer')}")

        if CAPTURED:
            print(f"  --- {len(CAPTURED[0]['context_texts'])} node(s) fed to LAYA ---")
            for index, text in enumerate(CAPTURED[0]["context_texts"], start=1):
                flat = " ".join(text.split())
                print(f"    node {index}: {flat[:220]}")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    gate = [r for r in results if r["kind"] == "GATE"]
    print(f"  previously-rejected queries now answered: "
          f"{sum(1 for r in gate if r['response'].get('matched'))}/{len(gate)}")
    latencies = sorted(r["latency_ms"] for r in results)
    print(
        f"  latency  min {latencies[0]:.0f}ms | median "
        f"{latencies[len(latencies) // 2]:.0f}ms | max {latencies[-1]:.0f}ms"
    )
    print(f"  peak rss {max(r['rss_mb'] for r in results):.0f}MB")
    multi = [r for r in results if r["nodes_used"] > 1]
    print(f"  queries that pulled more than one context node: {len(multi)}")
    for record in multi:
        print(f"    - {record['nodes_used']} nodes: {record['query']}")

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
        "threshold-045-retest.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(
            {"config": {"threshold": config.similarity_threshold, "top_k": config.top_k},
             "results": results},
            handle,
            indent=2,
        )
    print(f"\n  raw results written to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
