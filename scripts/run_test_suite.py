"""Multi-model evaluation harness for the LAYA-backed catalog API.

Spins the API up once per LAYA checkpoint (``english`` and ``multilingual`` by
default), replays the grounded question set in ``scripts/questions.json``, and
writes both raw results (JSON) and a readable report (Markdown).

Usage::

    python scripts/run_test_suite.py                    # english + multilingual
    python scripts/run_test_suite.py english            # a single model
    python scripts/run_test_suite.py english multilingual

Every recorded row carries the wall-clock timestamp, the exact request body, the
exact response body, and the round-trip latency in milliseconds.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTIONS_FILE = os.path.join(PROJECT_ROOT, "scripts", "questions.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
BASE_PORT = 8091


# --------------------------------------------------------------------------- #
# Server lifecycle
# --------------------------------------------------------------------------- #

def start_server(model: str, port: int, log_path: str) -> subprocess.Popen:
    """Start ``python -m src.server`` with ``LAYA_MODEL`` pinned to ``model``."""
    env = dict(os.environ)
    env["LAYA_MODEL"] = model
    env["PORT"] = str(port)
    log_handle = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "src.server"],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        # Own process group: lets stop_server() kill the server without taking
        # this harness down with it.
        start_new_session=True,
    )
    process._log_handle = log_handle  # type: ignore[attr-defined]
    return process


def wait_until_ready(port: int, process: subprocess.Popen, timeout_s: float = 600.0) -> float:
    """Block until the OpenAPI document is served. Returns readiness seconds."""
    started = time.time()
    url = f"http://127.0.0.1:{port}/openapi.json"
    while time.time() - started < timeout_s:
        if process.poll() is not None:
            raise RuntimeError(f"server exited early with code {process.returncode}")
        try:
            if requests.get(url, timeout=3).status_code == 200:
                return round(time.time() - started, 2)
        except requests.RequestException:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"server not ready within {timeout_s}s")


def stop_server(process: subprocess.Popen) -> None:
    """Terminate the server and its process group."""
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
    handle = getattr(process, "_log_handle", None)
    if handle:
        handle.close()


# --------------------------------------------------------------------------- #
# Test execution
# --------------------------------------------------------------------------- #

def run_model(model: str, port: int, groups: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Replay every question against a freshly started server. Returns a result bundle."""
    log_path = os.path.join(RESULTS_DIR, f"server-{model}.log")
    print(f"\n=== model={model} port={port} ===", flush=True)
    process = start_server(model, port, log_path)
    try:
        ready_s = wait_until_ready(port, process)
        print(f"ready in {ready_s}s", flush=True)

        # Confirm what the server actually ingested so the report is auditable.
        catalog_size = None
        try:
            ingest_probe = requests.post(
                f"http://127.0.0.1:{port}/catalog/query",
                json={"query": "zzz_probe_nonexistent_zzz"},
                timeout=60,
            ).json()
            catalog_size = "ingested" if "score" in ingest_probe else None
        except requests.RequestException:
            pass

        rows: List[Dict[str, Any]] = []
        for group in groups:
            for variant in group["variants"]:
                request_body = {"query": variant}
                started = time.time()
                timestamp = datetime.now().isoformat(timespec="milliseconds")
                try:
                    response = requests.post(
                        f"http://127.0.0.1:{port}/catalog/query",
                        json=request_body,
                        timeout=120,
                    )
                    latency_ms = round((time.time() - started) * 1000, 2)
                    body = response.json()
                    status = response.status_code
                except Exception as error:  # noqa: BLE001 - recorded, not raised
                    latency_ms = round((time.time() - started) * 1000, 2)
                    body = {"error": str(error)}
                    status = None

                rows.append(
                    {
                        "group": group["id"],
                        "category": group["category"],
                        "expected": group["expected"],
                        "question": variant,
                        "timestamp": timestamp,
                        "latency_ms": latency_ms,
                        "http_status": status,
                        "request": request_body,
                        "response": body,
                    }
                )
                label = "MATCH" if body.get("matched") else "NO_MATCH"
                print(
                    f"  [{label:8s}] {latency_ms:8.2f}ms  score={body.get('score')}  {variant[:58]}",
                    flush=True,
                )

        return {
            "model": model,
            "port": port,
            "ready_seconds": ready_s,
            "catalog_probe": catalog_size,
            "started_at": rows[0]["timestamp"] if rows else None,
            "finished_at": rows[-1]["timestamp"] if rows else None,
            "rows": rows,
        }
    finally:
        stop_server(process)
        print(f"server for {model} stopped", flush=True)


# --------------------------------------------------------------------------- #
# Report rendering
# --------------------------------------------------------------------------- #

def normalise(text: Any) -> str:
    """Collapse whitespace so trivial formatting differences do not mask equality."""
    return " ".join(str(text or "").split())


def build_report(bundles: List[Dict[str, Any]], groups: List[Dict[str, Any]]) -> str:
    """Render the Markdown report."""
    out: List[str] = []
    models = [bundle["model"] for bundle in bundles]

    out.append("# LAYA Chatbot — Test Question Report\n")
    out.append(f"Generated: `{datetime.now().isoformat(timespec='seconds')}`\n")
    out.append(
        "Harness: `scripts/run_test_suite.py` · Question set: `scripts/questions.json` "
        f"· Models: {', '.join(f'`{m}`' for m in models)}\n"
    )
    out.append(
        "\nEach row records the exact request body, the exact response body, a wall-clock "
        "timestamp, and round-trip latency, so every claim below is reproducible.\n"
    )

    # ---------------------------------------------------------------- summary
    out.append("\n## 1. Run summary\n")
    out.append("| Model | Startup | Questions | Matched | Refused | Errors | Mean latency | Median latency |")
    out.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for bundle in bundles:
        rows = bundle["rows"]
        matched = sum(1 for r in rows if r["response"].get("matched") is True)
        errors = sum(1 for r in rows if "error" in r["response"])
        refused = len(rows) - matched - errors
        latencies = sorted(r["latency_ms"] for r in rows)
        mean = sum(latencies) / len(latencies) if latencies else 0
        median = latencies[len(latencies) // 2] if latencies else 0
        out.append(
            f"| `{bundle['model']}` | {bundle['ready_seconds']}s | {len(rows)} | {matched} | "
            f"{refused} | {errors} | {mean:.0f} ms | {median:.0f} ms |"
        )

    # ------------------------------------------------- paraphrase consistency
    out.append("\n## 2. Does rewording change the answer?\n")
    out.append(
        "For every group, the variants are paraphrases of one intent. A group is "
        "**consistent** when all variants return the same answer text; **partial** when "
        "they all match but the answer text differs; **inconsistent** when some variants "
        "match and others are refused.\n"
    )
    for bundle in bundles:
        rows_by_group: Dict[str, List[Dict[str, Any]]] = {}
        for row in bundle["rows"]:
            rows_by_group.setdefault(row["group"], []).append(row)

        consistent = partial = inconsistent = 0
        for group_rows in rows_by_group.values():
            matched_flags = [r["response"].get("matched") is True for r in group_rows]
            answers = {normalise(r["response"].get("answer")) for r in group_rows}
            if all(matched_flags) and len(answers) == 1:
                consistent += 1
            elif all(matched_flags):
                partial += 1
            else:
                inconsistent += 1

        out.append(
            f"\n### Model: `{bundle['model']}`\n\n"
            f"- Fully consistent groups: **{consistent}/{len(rows_by_group)}**\n"
            f"- All matched, wording of answer varied: **{partial}/{len(rows_by_group)}**\n"
            f"- Mixed match/refusal across paraphrases: **{inconsistent}/{len(rows_by_group)}**\n"
        )

        out.append("\n| Group | Category | Variant | Matched | Score | Latency | Timestamp | Answer | Δ vs variant 1 |")
        out.append("| :--- | :--- | :--- | :---: | ---: | ---: | :--- | :--- | :---: |")
        for group in groups:
            group_rows = rows_by_group.get(group["id"], [])
            baseline = normalise(group_rows[0]["response"].get("answer")) if group_rows else ""
            for index, row in enumerate(group_rows):
                answer = normalise(row["response"].get("answer"))
                same = "—" if index == 0 else ("same" if answer == baseline else "**different**")
                score = row["response"].get("score")
                score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "—"
                out.append(
                    f"| {group['id']} | {group['category']} | {row['question']} | "
                    f"{'yes' if row['response'].get('matched') else 'no'} | {score_text} | "
                    f"{row['latency_ms']:.0f} ms | {row['timestamp']} | {answer[:160]} | {same} |"
                )

    # ------------------------------------------------------ cross-model diff
    if len(bundles) > 1:
        out.append("\n## 3. Cross-model comparison\n")
        out.append(
            "Same question, different LAYA checkpoint. `score` and the answer text are "
            "compared directly.\n"
        )
        baseline_bundle = bundles[0]
        baseline_rows = {(r["group"], r["question"]): r for r in baseline_bundle["rows"]}
        for bundle in bundles[1:]:
            out.append(f"\n### `{baseline_bundle['model']}` vs `{bundle['model']}`\n")
            out.append("| Question | Score A | Score B | Δscore | Answer A | Answer B | Same answer |")
            out.append("| :--- | ---: | ---: | ---: | :--- | :--- | :---: |")
            changed = 0
            for row in bundle["rows"]:
                peer = baseline_rows.get((row["group"], row["question"]))
                if not peer:
                    continue
                score_a = peer["response"].get("score")
                score_b = row["response"].get("score")
                delta = (
                    f"{score_b - score_a:+.4f}"
                    if isinstance(score_a, (int, float)) and isinstance(score_b, (int, float))
                    else "—"
                )
                answer_a = normalise(peer["response"].get("answer"))
                answer_b = normalise(row["response"].get("answer"))
                same = answer_a == answer_b
                if not same:
                    changed += 1
                out.append(
                    f"| {row['question']} | "
                    f"{score_a:.4f} | {score_b:.4f} | {delta} | "
                    f"{answer_a[:110]} | {answer_b[:110]} | {'yes' if same else '**no**'} |"
                    if isinstance(score_a, (int, float)) and isinstance(score_b, (int, float))
                    else f"| {row['question']} | — | — | — | {answer_a[:110]} | {answer_b[:110]} | {'yes' if same else '**no**'} |"
                )
            out.append(f"\nAnswers differing between checkpoints: **{changed}/{len(bundle['rows'])}**\n")

    # --------------------------------------------------------------- appendix
    out.append("\n## 4. Raw request/response log\n")
    out.append("Verbatim request and response bodies, including timestamps.\n")
    for bundle in bundles:
        out.append(f"\n### Model: `{bundle['model']}`\n")
        out.append("```json")
        for row in bundle["rows"]:
            out.append(
                json.dumps(
                    {
                        "timestamp": row["timestamp"],
                        "latency_ms": row["latency_ms"],
                        "http_status": row["http_status"],
                        "request": row["request"],
                        "response": row["response"],
                    },
                    ensure_ascii=False,
                )
            )
        out.append("```")

    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #

def main() -> int:
    """Entry point."""
    models = sys.argv[1:] or ["english", "multilingual"]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as handle:
        groups = json.load(handle)["groups"]

    bundles: List[Dict[str, Any]] = []
    for index, model in enumerate(models):
        bundle = run_model(model, BASE_PORT + index, groups)
        with open(os.path.join(RESULTS_DIR, f"results-{model}.json"), "w", encoding="utf-8") as handle:
            json.dump(bundle, handle, ensure_ascii=False, indent=2)
        bundles.append(bundle)

    report = build_report(bundles, groups)
    report_path = os.path.join(PROJECT_ROOT, "TEST_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report)

    print(f"\nReport written to {report_path}", flush=True)
    print(f"Raw results in {RESULTS_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
