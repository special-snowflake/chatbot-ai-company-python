"""Application configuration.

Python port of ``src/config.js`` from ``special-snowflake/chatbot-ai-company``.

The original reads the same values from ``process.env`` with identical
fallbacks, so the Python port keeps the names, defaults and parsing rules
byte-for-byte compatible with the Node implementation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # dotenv is optional; the original hard-depends on it, we degrade gracefully.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is a convenience only
    pass

# Anchor relative paths to the project root (the parent of ``src/``) so the app
# behaves identically whether it is launched from the project directory, from a
# different cwd, or imported by a test runner.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(value: str) -> str:
    """Return an absolute path, resolving relative values against the project root."""
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (PROJECT_ROOT / path))


def _number_from_env(name: str, fallback: float) -> float:
    """Mirror ``numberFromEnv`` in config.js: parse a float, else fall back."""
    try:
        value = float(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return fallback
    return value if value == value and value not in (float("inf"), float("-inf")) else fallback


@dataclass(frozen=True)
class Config:
    """Immutable snapshot of runtime configuration."""

    port: int
    similarity_threshold: float
    top_k: int
    embedding_model: str
    # ``llmModelPath`` in the original pointed at a local ``.gguf``. LAYA is a
    # hosted-checkpoint decision engine, so the equivalent knob selects the
    # LAYA checkpoint family instead of a file on disk.
    laya_model: str
    llm_timeout_ms: int
    catalog_file: str
    catalog_index_file: str


def load_config() -> Config:
    """Build a :class:`Config` from the environment with JS-compatible defaults."""
    return Config(
        port=int(_number_from_env("PORT", 3000)),
        similarity_threshold=_number_from_env("SIMILARITY_THRESHOLD", 0.58),
        top_k=max(1, int(_number_from_env("TOP_K", 3))),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        laya_model=os.environ.get("LAYA_MODEL", "english"),
        llm_timeout_ms=max(1, int(_number_from_env("LLM_TIMEOUT_MS", 30000))),
        catalog_file=_resolve(os.environ.get("CATALOG_FILE", "./source")),
        catalog_index_file=_resolve(os.environ.get("CATALOG_INDEX_FILE", "./data/catalog-index.json")),
    )


# Module-level singleton, matching ``export const config = {...}`` in config.js.
config = load_config()


# --------------------------------------------------------------------------- #
# Port additions: answering-strategy knobs
# --------------------------------------------------------------------------- #
# Nothing in this section exists in ``config.js``. The JS service had a single
# answering path (embed -> LAYA -> return whichever context item LAYA picked).
# Every default below is a superset of that behaviour rather than a regression,
# and every value can be overridden from the environment.


def _bool_from_env(name: str, fallback: bool) -> bool:
    """Parse a boolean env var, accepting the usual truthy/falsy spellings."""
    raw = os.environ.get(name)
    if raw is None:
        return fallback
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    return fallback


@dataclass(frozen=True)
class AnsweringConfig:
    """Tuning for the deterministic answering strategy.

    ``answer_mode`` is the important one:

    * ``"hybrid"`` (default) answers directly from retrieved nodes without a
      model call, and only reaches for a model when no node answer can be
      rendered.
    * ``"legacy"`` reproduces the original pipeline exactly — every match goes
      to LAYA — so the old and new behaviour can be measured against each other.

    ``tie_epsilon`` is the score window treated as a tie: nodes within that
    distance of the best match are all returned, so an ambiguous question (one
    that never names a delivery region, say) yields every plausible answer
    instead of one arbitrary pick.
    """

    answer_mode: str
    tie_epsilon: float
    max_tie_nodes: int
    drop_contentless_nodes: bool
    enable_compound_split: bool
    enable_hazard_escalation: bool
    enable_budget_listing: bool
    query_cache_size: int
    warm_layap: bool
    synthesizer_base_url: str
    synthesizer_model: str
    synthesizer_api_key: str
    synthesizer_timeout_ms: int


def load_answering_config() -> AnsweringConfig:
    """Build an :class:`AnsweringConfig` from the environment."""
    mode = os.environ.get("ANSWER_MODE", "hybrid").strip().lower()
    return AnsweringConfig(
        answer_mode=mode if mode in {"hybrid", "legacy"} else "hybrid",
        tie_epsilon=_number_from_env("TIE_EPSILON", 0.05),
        max_tie_nodes=max(1, int(_number_from_env("MAX_TIE_NODES", 3))),
        drop_contentless_nodes=_bool_from_env("DROP_CONTENTLESS_NODES", True),
        enable_compound_split=_bool_from_env("ENABLE_COMPOUND_SPLIT", True),
        enable_hazard_escalation=_bool_from_env("ENABLE_HAZARD_ESCALATION", True),
        enable_budget_listing=_bool_from_env("ENABLE_BUDGET_LISTING", True),
        query_cache_size=max(0, int(_number_from_env("QUERY_CACHE_SIZE", 256))),
        # Off by default: the checkpoint costs ~1.6 GB and several seconds, and
        # the deterministic path means most installs never need it.
        warm_layap=_bool_from_env("WARM_LAYAP", False),
        synthesizer_base_url=os.environ.get("SYNTHESIZER_BASE_URL", "").rstrip("/"),
        synthesizer_model=os.environ.get("SYNTHESIZER_MODEL", ""),
        synthesizer_api_key=os.environ.get("SYNTHESIZER_API_KEY", ""),
        synthesizer_timeout_ms=max(1000, int(_number_from_env("SYNTHESIZER_TIMEOUT_MS", 20000))),
    )


#: Module-level singleton for the strategy knobs.
answering = load_answering_config()
