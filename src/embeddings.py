"""Embedding provider.

Python port of ``src/embeddings.js``.

The original used ``@xenova/transformers`` with ``Xenova/all-MiniLM-L6-v2``
(384-dimensional, mean-pooled, L2-normalised). The Python equivalent is
``sentence-transformers``, which is a drop-in for the same checkpoint.

IMPORTANT (honesty note): if ``sentence-transformers`` is not installed we do
**not** silently pretend to embed with a real model. We fall back to a
deterministic hashed bag-of-words embedder and flag it as ``degraded`` so the
caller can tell the difference. The original code had no such fallback because
the model was a hard dependency.
"""

from __future__ import annotations

import hashlib
import re
from typing import Callable, Dict

_EMBEDDING_DIM = 384  # Dimension of all-MiniLM-L6-v2, same as the JS pipeline.
_WORD_RE = re.compile(r"[a-z0-9]+")

# Cache of provider callables keyed by model name, mirroring ``extractorPromises``.
_provider_cache: Dict[str, Callable[[str], list]] = {}


def _hashed_embedding(text: str) -> list:
    """Deterministic offline fallback: hashed bag-of-words, L2-normalised.

    Not semantically meaningful like a transformer — it exists only so the API
    stays runnable with zero downloads. Callers detect it via ``provider.degraded``.
    """
    vector = [0.0] * _EMBEDDING_DIM
    for token in _WORD_RE.findall(text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % _EMBEDDING_DIM
        vector[index] += 1.0
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class EmbeddingProvider:
    """Callable wrapper exposing ``embed(text) -> list[float]``.

    Attributes:
        degraded: ``True`` when the real transformer model is unavailable and the
            hashed fallback is in use.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self.degraded = False
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(model)
        except Exception as exc:  # pragma: no cover - depends on optional extra
            self.degraded = True
            self._load_error = str(exc)

    def embed(self, text: str) -> list:
        """Embed ``text``, matching the JS pipeline's normalisation behaviour."""
        if self._model is not None:
            vector = self._model.encode(text, normalize_embeddings=True)
            return [float(value) for value in vector]
        return _hashed_embedding(text)


def create_embedding_provider(model: str) -> EmbeddingProvider:
    """Return a cached embedding provider for ``model`` (mirrors embeddings.js)."""
    if model not in _provider_cache:
        _provider_cache[model] = EmbeddingProvider(model)
    return _provider_cache[model]
