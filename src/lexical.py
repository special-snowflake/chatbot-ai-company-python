"""Lexical helpers — tokenising, aliases, and price/budget parsing.

Direct port of the lexical constants and helpers that live inline in
``src/catalog-service.js``. They were extracted into their own module so that
this port's new intent analysis (:mod:`src.intent`) and node-quality guards
(:mod:`src.answer_text`) can reuse them without importing the service and
creating an import cycle.

Nothing here is rewritten: the stop-word set, the alias map, the regular
expressions and the parsing rules are byte-for-byte the JS behaviour.
:mod:`src.catalog_service` re-exports every public name below, so
``from src.catalog_service import meaningful_words`` keeps working.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Set

# --------------------------------------------------------------------------- #
# Lexical constants — direct ports of the JS constants
# --------------------------------------------------------------------------- #

_STOP_WORDS: Set[str] = {
    "a", "an", "and", "are", "does", "for", "how", "in", "is", "it",
    "of", "on", "the", "to", "what", "when", "where", "which", "who",
}

_WORD_ALIASES: Dict[str, str] = {
    "duration": "long",
    "eta": "long",
    "time": "long",
    "takes": "take",
    "taking": "take",
    # Port addition: "item" and "product" name the same thing in this corpus, so
    # they are normalised. Measured effect on the eval set: none. The case this
    # was added for is cosine-dominated, so a lexical gain cannot lift it over
    # the gate (see EVAL_REPORT.md); this is a correctness tidy-up, not a fix.
    "item": "product",
    "items": "product",
}

_WORD_RE = re.compile(r"[a-z0-9]+")
_PRICE_RE = re.compile(
    r"(?:retail price|price)\s*:\s*\*{0,2}\s*Rp\s*([\d.,]+)", re.IGNORECASE
)
_BUDGET_RE = re.compile(
    r"(?:under|below|less than|up to|maximum|max)\s*(?:rp\s*)?([\d.,]+)\s*(k|rb|ribu)?\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #

def meaningful_words(text: str) -> Set[str]:
    """Tokenise ``text``, drop stop words, and apply alias normalisation."""
    words: Set[str] = set()
    for token in _WORD_RE.findall(text.lower()):
        if token in _STOP_WORDS:
            continue
        alias = _WORD_ALIASES.get(token)
        words.add(alias if alias is not None else token)
    return words


def lexical_similarity(query_words: Set[str], entry_words: Set[str]) -> float:
    """Fraction of query words that also appear in the entry."""
    if not query_words or not entry_words:
        return 0.0
    overlap = len([word for word in query_words if word in entry_words])
    return overlap / len(query_words)


def parse_price(text: str) -> Optional[int]:
    """Extract an ``Rp`` price from catalog text, or ``None``."""
    match = _PRICE_RE.search(text)
    if not match:
        return None
    try:
        return int(match.group(1).replace(".", "").replace(",", ""))
    except ValueError:
        return None


def parse_budget_query(query: str) -> Optional[Dict[str, float]]:
    """Extract a maximum budget from a natural-language query, or ``None``."""
    match = _BUDGET_RE.search(query)
    if not match:
        return None
    try:
        amount = float(match.group(1).replace(".", "").replace(",", ""))
    except ValueError:
        return None
    if amount != amount:  # NaN guard
        return None
    return {"maxPrice": amount * (1000 if match.group(2) else 1)}


def entry_text(entry: Dict[str, Any]) -> str:
    """Normalise a catalog entry to text (port of ``entryText``)."""
    if isinstance(entry.get("text"), str):
        return entry["text"]
    if isinstance(entry.get("question"), str) and isinstance(entry.get("answer"), str):
        return f"Question: {entry['question']}\nAnswer: {entry['answer']}"
    if isinstance(entry.get("title"), str) and isinstance(entry.get("content"), str):
        return f"{entry['title']}\n{entry['content']}"
    raise ValueError("Each entry must contain question/answer, title/content, or text")
