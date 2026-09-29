"""Catalog ingestion + retrieval service.

Python port of ``src/catalog-service.js``.

The scoring pipeline is preserved exactly:
``max(cosine, 0.7 * cosine + 0.3 * lexical)``, threshold rejection, budget
filtering, ``topK`` selection, and the ``NO_MATCH`` fallback path. The only
substitution is the answer synthesis step, which now calls
:class:`src.laya_engine.LayaEngine` instead of ``node-llama-cpp``.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

from .embeddings import EmbeddingProvider
from .laya_engine import NO_MATCH, LayaEngine
from .similarity import cosine_similarity

logger = logging.getLogger("catalog.service")

# --------------------------------------------------------------------------- #
# Lexical helpers — direct ports of the JS constants
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
}

_WORD_RE = re.compile(r"[a-z0-9]+")
_PRICE_RE = re.compile(r"(?:retail price|price)\s*:\s*\*{0,2}\s*Rp\s*([\d.,]+)", re.IGNORECASE)
_BUDGET_RE = re.compile(
    r"(?:under|below|less than|up to|maximum|max)\s*(?:rp\s*)?([\d.,]+)\s*(k|rb|ribu)?\b",
    re.IGNORECASE,
)


def meaningful_words(text: str) -> Set[str]:
    """Tokenise ``text``, drop stop words, and apply alias normalisation."""
    tokens = _WORD_RE.findall(text.lower())
    return {_WORD_ALIASES.get(token, token) for token in tokens if token not in _STOP_WORDS}


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


# --------------------------------------------------------------------------- #
# Conversational intent — direct port of ``conversationalResponses``
# --------------------------------------------------------------------------- #

_CONVERSATIONAL_RESPONSES: List[tuple] = [
    (re.compile(r"^(selamat pagi|selamat siang|selamat sore|selamat malam|halo|hai)$", re.I),
     "Halo! Ada yang bisa saya bantu?"),
    (re.compile(r"^(hi|hello|hey|hey there|good morning|good afternoon|good evening|morning)$", re.I),
     "Hello! How can I help you today?"),
    (re.compile(r"^(how are you|what(?:'s| is) up|nice to meet you)$", re.I),
     "I am here and ready to help. What would you like to know?"),
    (re.compile(r"^(sampai jumpa|dadah|terima kasih,? cukup|sudah,? terima kasih)$", re.I),
     "Sama-sama! Sampai jumpa."),
    (re.compile(
        r"^(bye|goodbye|see you|see ya|see you later|talk to you later|thanks,? bye|"
        r"thank you,? goodbye|that(?:'s| is) all|i(?:'m| am) done|no more questions)$", re.I),
     "Goodbye! Have a great day."),
    (re.compile(r"^(makasih|terima kasih)$", re.I), "Sama-sama!"),
    (re.compile(
        r"^(thanks|thank you|thanks a lot|appreciate it|thank you so much|thanks,? that helps)$",
        re.I),
     "You are welcome!"),
    (re.compile(
        r"^(are you there|what are you doing|nice|cool|great|that(?:'s| is) helpful|good bot)$",
        re.I),
     "I am here and ready to help."),
]

_MIXED_PREFIXES = [
    "terima kasih", "thank you", "hey there", "makasih",
    "thanks", "hello", "halo", "hai", "hey", "hi",
]


def conversational_intent(query: str) -> Optional[Dict[str, str]]:
    """Detect small-talk and return a canned answer, a rewritten query, or ``None``."""
    normalized = query.lower().strip()
    while normalized and normalized[-1] in "!?.,":
        normalized = normalized[:-1].strip()

    for pattern, answer in _CONVERSATIONAL_RESPONSES:
        if pattern.match(normalized):
            return {"answer": answer}

    mixed_prefix = None
    for prefix in _MIXED_PREFIXES:
        if (
            normalized == prefix
            or normalized.startswith(f"{prefix} ")
            or normalized.startswith(f"{prefix}!")
            or normalized.startswith(f"{prefix},")
        ):
            mixed_prefix = prefix
            break

    if mixed_prefix:
        remainder = normalized[len(mixed_prefix):].lstrip(" \t,!.-").strip()
        if remainder.startswith("by the way"):
            remainder = remainder[len("by the way"):].lstrip(" \t,!.-").strip()
        if not remainder:
            return None
        for pattern, answer in _CONVERSATIONAL_RESPONSES:
            if pattern.match(remainder):
                return {"answer": answer}
        if mixed_prefix in ("thanks", "thank you", "makasih", "terima kasih"):
            return {"query": remainder, "prefix": "You are welcome! "}
        if mixed_prefix in ("halo", "hai"):
            return {"query": remainder, "prefix": "Halo! "}
        return {"query": remainder, "prefix": "Hello! "}

    return None


def entry_text(entry: Dict[str, Any]) -> str:
    """Normalise a catalog entry to text (port of ``entryText``)."""
    if isinstance(entry.get("text"), str):
        return entry["text"]
    if isinstance(entry.get("question"), str) and isinstance(entry.get("answer"), str):
        return f"Question: {entry['question']}\nAnswer: {entry['answer']}"
    if isinstance(entry.get("title"), str) and isinstance(entry.get("content"), str):
        return f"{entry['title']}\n{entry['content']}"
    raise ValueError("Each entry must contain question/answer, title/content, or text")


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #

class CatalogService:
    """In-memory catalog with hybrid retrieval and LAYA-backed answering."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        llm: LayaEngine,
        threshold: float,
        top_k: int,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self.embedder = embedder
        self.llm = llm
        self.threshold = threshold
        self.top_k = top_k
        self.logger = logger_ or logger
        self.index: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #

    def ingest(self, entries: List[Dict[str, Any]]) -> Dict[str, int]:
        """Replace the index with ``entries``. Port of ``ingest()``."""
        if not isinstance(entries, list):
            raise ValueError("Document must be an array of entries")

        index: List[Dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("id"):
                raise ValueError("Each entry must have an id")
            text = entry_text(entry)
            index.append(
                {
                    "id": entry["id"],
                    "text": text,
                    "embedding": self.embedder.embed(text),
                    "words": meaningful_words(text),
                    "price": parse_price(text),
                }
            )
        self.index = index
        return {"count": len(index)}

    def load_index(self, index: Any) -> Dict[str, int]:
        """Load a pre-computed index, e.g. from ``catalog-index.json``."""
        if not isinstance(index, list):
            raise ValueError("Catalog index must be an array")
        for entry in index:
            if not isinstance(entry, dict) or not entry.get("id") or not isinstance(entry.get("text"), str):
                raise ValueError("Catalog index contains an invalid entry")
            if not isinstance(entry.get("embedding"), list):
                raise ValueError("Catalog index contains an invalid entry")

        self.index = [
            {
                **entry,
                "words": set(entry["words"]) if isinstance(entry.get("words"), list) else meaningful_words(entry["text"]),
                "price": entry.get("price", parse_price(entry["text"])),
            }
            for entry in index
        ]
        return {"count": len(self.index)}

    # ------------------------------------------------------------------ #

    def query(self, query: str) -> Dict[str, Any]:
        """Return the best answer for ``query``. Port of ``query()``."""
        trimmed = query.strip() if isinstance(query, str) else ""
        if not trimmed:
            raise ValueError("query must be a non-empty string")

        intent = conversational_intent(trimmed)
        if intent and intent.get("answer"):
            self.logger.info(
                "conversation handled without catalog lookup (queryLength=%d)",
                len(trimmed),
            )
            return {"matched": True, "answer": intent["answer"], "score": 1}

        if not self.index:
            raise ValueError("Catalog has not been ingested")

        knowledge_query = intent.get("query") if intent else trimmed
        self.logger.info("catalog query processing (queryLength=%d)", len(knowledge_query))

        query_embedding = self.embedder.embed(knowledge_query)
        query_words = meaningful_words(knowledge_query)
        budget = parse_budget_query(knowledge_query)

        matches: List[Dict[str, Any]] = []
        for entry in self.index:
            semantic_score = cosine_similarity(query_embedding, entry["embedding"])
            lexical_score = lexical_similarity(query_words, entry["words"])
            matches.append(
                {**entry, "score": max(semantic_score, (semantic_score * 0.7) + (lexical_score * 0.3))}
            )
        matches.sort(key=lambda item: item["score"], reverse=True)

        score = matches[0]["score"]
        budget_matches = (
            [entry for entry in matches if entry["price"] is not None and entry["price"] <= budget["maxPrice"]]
            if budget
            else []
        )
        has_budget_match = len(budget_matches) > 0

        if score < self.threshold and not has_budget_match:
            self.logger.info("catalog query rejected (score=%.4f, matched=False)", score)
            return {
                "matched": False,
                "answer": "Sorry, I cannot help with that based on the available catalog.",
                "score": score,
            }

        top_matches = budget_matches if has_budget_match else matches[: self.top_k]
        try:
            answer = self.llm.answer(
                context="\n\n".join(
                    f"Context item {i + 1} (relevance {entry['score']:.3f}):\n{entry['text']}"
                    for i, entry in enumerate(top_matches)
                ),
                query=knowledge_query,
                context_texts=[entry["text"] for entry in top_matches],
            ).strip()

            if not answer or answer == NO_MATCH:
                self.logger.info("catalog query rejected by LAYA (score=%.4f, matched=False)", score)
                return {
                    "matched": False,
                    "answer": "Sorry, I cannot help with that based on the available catalog.",
                    "score": score,
                }

            self.logger.info("catalog query matched (score=%.4f, matched=True)", score)
            return {"matched": True, "answer": f"{intent.get('prefix', '') if intent else ''}{answer}", "score": score}
        except Exception as error:  # pragma: no cover - defensive parity with JS catch
            self.logger.error(
                "catalog LAYA failed (error=%s, score=%.4f); returning raw match", error, score
            )
            return {"matched": True, "answer": matches[0]["text"], "score": score, "degraded": True}
