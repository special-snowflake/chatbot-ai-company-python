"""Query cache.

Port addition. Support traffic is highly repetitive, and the deterministic
answering path (see :mod:`src.answerer`) makes a cache hit genuinely cheap: a
bounded dict lookup instead of an embedding pass plus retrieval.

The cache is keyed on a normalised form of the query, so trivial formatting
differences (case, punctuation, collapsed whitespace) hit the same entry. It is
invalidated whenever the catalog is replaced, because every answer is derived
from the catalog.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional

_TRAILING_PUNCTUATION_RE = re.compile(r"[\s!?.,;:]+$")
_WHITESPACE_RE = re.compile(r"\s+")

DEFAULT_CACHE_SIZE = 256


def normalize(query: str) -> str:
    """Return the cache key for ``query``.

    Lower-cased, with collapsed whitespace and trailing punctuation stripped.
    The same normalisation the conversational matcher applies, so ``" Halo?! "``
    and ``"halo"`` collapse to one key.
    """
    collapsed = _WHITESPACE_RE.sub(" ", str(query).strip().lower())
    return _TRAILING_PUNCTUATION_RE.sub("", collapsed)


class QueryCache:
    """A bounded, thread-safe LRU cache of query responses.

    Args:
        max_size: Maximum number of retained responses. ``0`` disables caching.
    """

    def __init__(self, max_size: int = DEFAULT_CACHE_SIZE) -> None:
        self.max_size = max(0, int(max_size))
        self._entries: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------ #

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Return a copy of the cached response for ``query``, or ``None``."""
        if self.max_size == 0:
            return None
        key = normalize(query)
        with self._lock:
            response = self._entries.get(key)
            if response is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            # A copy keeps callers from mutating cached state.
            return dict(response)

    def put(self, query: str, response: Dict[str, Any]) -> None:
        """Store ``response`` for ``query``, evicting the least recent entry."""
        if self.max_size == 0:
            return
        key = normalize(query)
        with self._lock:
            self._entries[key] = dict(response)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_size:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        """Drop every entry. Called whenever the catalog is replaced."""
        with self._lock:
            self._entries.clear()

    def stats(self) -> Dict[str, int]:
        """Return a snapshot of cache counters, for logging and eval reporting."""
        with self._lock:
            return {
                "size": len(self._entries),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
            }
