"""Application configuration.

Python port of ``src/config.js`` from
``special-snowflake/chatbot-ai-company``.

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
