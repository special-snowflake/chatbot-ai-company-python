
"""Run the labeled eval set against every configuration and report real numbers.

Port addition. Usage:

    python scripts/run_eval.py                 # every configuration
    python scripts/run_eval.py --config hybrid-045

Each configuration runs in its own process so resident-memory figures are not
polluted by a previously loaded model. Results land in
``results/eval-<config>.json`` plus a combined ``EVAL_REPORT.md``.

The three configurations exist to separate the two changes that were made:

    original-058   as ported: gate at 0.58, the model answers every match
    threshold-045  same pipeline, gate relaxed to 0.45 (first attempted fix)
    hybrid-045     0.45 plus the deterministic strategy in src/answerer.py

``threshold-045`` vs ``original-058`` isolates the gate change;
``hybrid-045`` vs ``threshold-045`` isolates the strategy change.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_questions import CASES  # noqa: E402

RESULTS_DIR = ROOT / "results"

#: One entry per configuration under test.
CONFIGS: Dict[str, Dict[str, Any]] = {
    "original-058": {
        "label": "original pipeline, gate 0.58",
        "threshold": 0.58,
        "answer_mode": "legacy",
        "tie_epsilon": -1.0,
        "drop_contentless_nodes": False,
        "enable_compound_split": False,
        "enable_hazard_escalation": False,
        "enable_budget_listing": False,
        "query_cache_size": 0,
    },
    "threshold-045": {
        "label": "same pipeline, gate relaxed to 0.45",
        "threshold": 0.45,
        "answer_mode": "legacy",
        "tie_epsilon": -1.0,
        "drop_contentless_nodes": False,
        "enable_compound_split": False,
        "enable_hazard_escalation": False,
        "enable_budget_listing": False,
        "query_cache_size": 0,
    },
    "hybrid-045": {
        "label": "deterministic strategy, gate 0.45",
        "threshold": 0.45,
        "answer_mode": "hybrid",
        "tie_epsilon": 0.05,
        "drop_contentless_nodes": True,
        "enable_compound_split": True,
        "enable_hazard_escalation": True,
        "enable_budget_listing": True,
        "query_cache_size": 0,
    },
}

#: Ceiling used when this module cannot read its own resident set size.
UNKNOWN_RSS_KB = -1


def resident_kb() -> int:
    """Current resident set size in kB, or ``UNKNOWN_RSS_KB``.

    Read straight from ``/proc`` so no dependency is needed. This is how the
    "does it still hold 2.5 GB for a FAQ bot" question gets a factual answer.
    """
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            pages = int(handle.read().split()[1])
    except (OSError, IndexError, ValueError):
        return UNKNOWN_RSS_KB
    return pages * 4


def percentile(values: List[float], fraction: float) -> float:
    """Nearest-rank percentile, which is honest for small samples."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(fraction * len(ordered))) - 1))
    return ordered[index]


def contains_ci(haystack: str, needle: str) -> bool:
    """Case-insensitive substring test."""
    return needle.lower() in haystack.lower()


def check(case: Dict[str, Any], response: Dict[str, Any]) -> Tuple[bool, str]:
    """Judge one response against the case's expectation.

    Returns ``(passed, reason)``; the reason is always populated so a failure
    explains itself without re-running the whole eval.
    """
    answer = str(response.get("answer", ""))
    matched = bool(response.get("matched"))
    expect = case.get("expect", "answer")

    if expect == "refuse":
        if matched:
            return False, f"expected refusal, got answer: {answer[:90]!r}"
        return True, "refused as expected"

    if not matched:
        return False, f"expected an answer, refused (score={response.get('score')})"

    if expect == "escalate" and not contains_ci(answer, "escalat"):
        return False, f"hazard not escalated: {answer[:90]!r}"

    return evaluate_content(case, answer)


def evaluate_content(case: Dict[str, Any], answer: str) -> Tuple[bool, str]:
    """Apply the substring expectations of one case."""
    missing = [text for text in case.get("all", []) if not contains_ci(answer, text)]
    if missing:
        return False, f"missing {missing}"

    forbidden = [text for text in case.get("none", []) if contains_ci(answer, text)]
    if forbidden:
        return False, f"contains forbidden {forbidden}"

    wanted = case.get("any", [])
    if wanted:
        hits = [text for text in wanted if contains_ci(answer, text)]
        needed = int(case.get("min_any", 1))
        if len(hits) < needed:
            return False, f"needed {needed} of {wanted}, found {hits}"

    return True, "ok"


def build_service(name: str) -> Tuple[Any, Dict[str, Any]]:
    """Construct the service for one configuration and ingest ``./source``.

    Imported lazily so ``--all`` can spawn clean child processes without paying
    for heavy imports in the parent.
    """
    from src.catalog_service import CatalogService
    from src.config import config
    from src.embeddings import create_embedding_provider
    from src.laya_engine import create_laya_provider
    from src.source_loader import load_source_entries

    spec = CONFIGS[name]
    service = CatalogService(
        embedder=create_embedding_provider(config.embedding_model),
        llm=create_laya_provider(config.laya_model, config.llm_timeout_ms),
        threshold=spec["threshold"],
        top_k=config.top_k,
        tie_epsilon=spec["tie_epsilon"],
        drop_contentless_nodes=spec["drop_contentless_nodes"],
        enable_compound_split=spec["enable_compound_split"],
        enable_hazard_escalation=spec["enable_hazard_escalation"],
        enable_budget_listing=spec["enable_budget_listing"],
        answer_mode=spec["answer_mode"],
        query_cache_size=spec["query_cache_size"],
    )
    entries = load_source_entries(str(ROOT / "source"))
    result = service.ingest(entries)
    return service, {
        "chunks": int(result["count"]),
        "threshold": spec["threshold"],
        "answer_mode": spec["answer_mode"],
    }


def run_case(service: Any, case: Dict[str, Any]) -> Dict[str, Any]:
    """Run one case, timing it and recording what the strategy decided."""
    started = time.perf_counter()
    try:
        response, diagnostics = service.query_with_diagnostics(case["query"])
        error = ""
    except Exception as exc:  # one bad case must not abort the whole run
        response = {"matched": False, "answer": "", "score": 0.0}
        diagnostics = {}
        error = f"{type(exc).__name__}: {exc}"
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if error:
        passed, reason = False, error
    else:
        passed, reason = check(case, response)

    return {
        "id": case["id"],
        "category": case["category"],
        "query": case["query"],
        "expect": case.get("expect", "answer"),
        "note": case.get("note", ""),
        "answer": str(response.get("answer", "")),
        "matched": bool(response.get("matched")),
        "score": response.get("score"),
        "path": str(diagnostics.get("path", "")),
        "detail": str(diagnostics.get("detail", "")),
        "latency_ms": round(elapsed_ms, 1),
        "passed": passed,
        "reason": reason,
    }


def summarise(
    records: List[Dict[str, Any]],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Aggregate per-case records into the numbers worth reporting."""
    latencies = [record["latency_ms"] for record in records]

    by_category: Dict[str, Dict[str, Any]] = {}
    for record in records:
        bucket = by_category.setdefault(
            record["category"], {"total": 0, "passed": 0}
        )
        bucket["total"] += 1
        bucket["passed"] += int(record["passed"])
    for bucket in by_category.values():
        bucket["rate"] = round(bucket["passed"] / bucket["total"], 3)

    paths: Dict[str, int] = {}
    for record in records:
        key = record["path"] or "unknown"
        paths[key] = paths.get(key, 0) + 1

    passed = sum(1 for record in records if record["passed"])
    total = len(records)
    return {
        "config": meta["config"],
        "label": meta.get("label", ""),
        "source": meta,
        "cases": total,
        "passed": passed,
        "accuracy": round(passed / total, 3) if total else 0.0,
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 1) if latencies else 0.0,
            "p50": round(percentile(latencies, 0.50), 1),
            "p95": round(percentile(latencies, 0.95), 1),
            "max": round(max(latencies), 1) if latencies else 0.0,
        },
        "by_category": by_category,
        "paths": paths,
        "rss_kb": meta.get("rss_kb", UNKNOWN_RSS_KB),
        "model_calls": sum(
            1
            for record in records
            if record["path"] in {"layap", "degraded", "synthesizer"}
        ),
        "failures": [
            {
                "id": record["id"],
                "category": record["category"],
                "reason": record["reason"],
                "query": record["query"],
                "answer": record["answer"][:220],
            }
            for record in records
            if not record["passed"]
        ],
    }


def print_report(summary: Dict[str, Any]) -> None:
    """Print a compact, greppable summary for one configuration."""
    latency = summary["latency_ms"]
    print(
        f"accuracy {summary['passed']}/{summary['cases']} "
        f"({summary['accuracy'] * 100:.1f}%)  "
        f"p50 {latency['p50']:.0f}ms  p95 {latency['p95']:.0f}ms  "
        f"model calls {summary['model_calls']}"
    )
    for category, bucket in summary["by_category"].items():
        print(
            f"  {category:<14} {bucket['passed']:>2}/{bucket['total']:<2} "
            f"{bucket['rate'] * 100:>5.1f}%"
        )
    for failure in summary["failures"]:
        print(f"  FAIL {failure['id']}: {failure['reason']}")


def run_config(name: str) -> Dict[str, Any]:
    """Run every case for one configuration and persist the results."""
    service, info = build_service(name)

    # One throwaway query, so cold-start model loading does not land inside the
    # measured window. It is reported separately instead of being hidden.
    started = time.perf_counter()
    try:
        service.query("How long is the return window?")
    except Exception:  # a warmup failure is informative, not fatal
        pass
    warmup_ms = (time.perf_counter() - started) * 1000.0

    records = [run_case(service, case) for case in CASES]
    meta = {
        "config": name,
        "label": CONFIGS[name]["label"],
        "warmup_ms": round(warmup_ms, 1),
        "rss_kb": resident_kb(),
        **info,
    }
    summary = summarise(records, meta)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "records": records}
    (RESULTS_DIR / f"eval-{name}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"\n=== {name}: {CONFIGS[name]['label']} ===")
    print(
        f"warmup {warmup_ms:.0f}ms  rss {summary['rss_kb'] // 1024}MB  "
        f"chunks {info['chunks']}"
    )
    print_report(summary)
    return summary


def write_markdown(summaries: List[Dict[str, Any]]) -> Path:
    """Write the cross-configuration comparison to ``EVAL_REPORT.md``."""
    categories: List[str] = []
    for case in CASES:
        if case["category"] not in categories:
            categories.append(case["category"])

    lines = [
        "# Answering eval report",
        "",
        "Generated by `scripts/run_eval.py` from `scripts/eval_questions.py` "
        f"({len(CASES)} cases). Every expectation in the set is cited to the "
        "source line it came from, so a failure is a regression rather than a "
        "drifting assumption.",
        "",
        "| configuration | accuracy | p50 | p95 | max | model calls | RSS |",
        "|---|---|---|---|---|---|---|",
    ]
    for summary in summaries:
        latency = summary["latency_ms"]
        lines.append(
            f"| `{summary['config']}` | {summary['passed']}/{summary['cases']} "
            f"({summary['accuracy'] * 100:.1f}%) | {latency['p50']:.0f} ms | "
            f"{latency['p95']:.0f} ms | {latency['max']:.0f} ms | "
            f"{summary['model_calls']} | {summary['rss_kb'] // 1024} MB |"
        )

    lines += ["", "## Accuracy by category", ""]
    lines.append(
        "| category | " + " | ".join(s["config"] for s in summaries) + " |"
    )
    lines.append("|" + "---|" * (len(summaries) + 1))
    for category in categories:
        cells = []
        for summary in summaries:
            bucket = summary["by_category"].get(category)
            cells.append(
                f"{bucket['passed']}/{bucket['total']}" if bucket else "-"
            )
        lines.append(f"| {category} | " + " | ".join(cells) + " |")

    lines += ["", "## Decision paths", ""]
    for summary in summaries:
        paths = ", ".join(
            f"`{name}` x{count}"
            for name, count in sorted(summary["paths"].items())
        )
        lines.append(f"- `{summary['config']}`: {paths or 'none'}")

    for summary in summaries:
        if not summary["failures"]:
            continue
        lines += ["", f"## Failures: `{summary['config']}`", ""]
        for failure in summary["failures"]:
            lines.append(
                f"- **{failure['id']}** ({failure['category']}): "
                f"{failure['reason']}"
            )
            lines.append(f"  - query: {failure['query']}")
            lines.append(f"  - answer: {failure['answer']}")

    path = ROOT / "EVAL_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the labeled answer-quality eval set."
    )
    parser.add_argument(
        "--config", choices=sorted(CONFIGS), help="run a single configuration"
    )
    args = parser.parse_args()

    if args.config:
        run_config(args.config)
        return 0

    summaries: List[Dict[str, Any]] = []
    for name in CONFIGS:
        code = subprocess.call(
            [sys.executable, str(Path(__file__).resolve()), "--config", name]
        )
        if code != 0:
            print(f"configuration {name} exited {code}", file=sys.stderr)
            return code
        payload = json.loads(
            (RESULTS_DIR / f"eval-{name}.json").read_text(encoding="utf-8")
        )
        summaries.append(payload["summary"])

    print(f"\nwrote {write_markdown(summaries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
