"""Catalog ingestion and retrieval service.

Python port of ``src/catalog-service.js`` from
``special-snowflake/chatbot-ai-company``.

The scoring pipeline is preserved exactly as the original implemented it:

* hybrid score ``max(cosine, 0.7 * cosine + 0.3 * lexical)``
* threshold rejection before any answer is produced
* budget filtering (``parse_budget_query``) with the original's "a budget match
  bypasses the threshold" rule
* ``topK`` selection handed to the answering provider
* the ``NO_MATCH`` fallback path

What this port adds
-------------------
The original called the answering provider on every match and returned whatever
it selected. Measured against the NOVAHAUS corpus that produced two defects: valid
in-corpus questions refused at the similarity gate, and compound questions
answered with only half their intent, because the LAYA checkpoint is a *selector*
that echoes one context item rather than a generator that merges several.

Both defects are addressed by a strategy layer (:mod:`src.answerer`) that answers
from the retrieved nodes directly and only reaches for a model when no node answer
can be produced. The ported pipeline is kept intact and exposed as the ``legacy``
answer mode so the two strategies can be benchmarked against each other.

The service's public API is unchanged. ``query()`` returns the same keys as
before; callers that want to know *how* a response was produced can use
``query_with_diagnostics()``.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .answer_text import (
    is_contentless,
    node_label,
    probe_text,
    product_name,
    render_answer,
)
from .answerer import AnswerStrategy, Retrieval
from .cache import QueryCache, normalize
from .embeddings import EmbeddingProvider
from .laya_engine import NO_MATCH, LayaEngine
from .lexical import (
    entry_text,
    lexical_similarity,
    meaningful_words,
    parse_budget_query,
    parse_price,
)
from .similarity import cosine_similarity

# --------------------------------------------------------------------------- #
# Lexical helpers
# --------------------------------------------------------------------------- #
# ``meaningful_words``, ``lexical_similarity``, ``entry_text``, ``parse_price``
# and ``parse_budget_query`` now live in :mod:`src.lexical`, so the intent
# analysis in :mod:`src.intent` can reuse them without importing this module.
# They are re-exported here because callers and the test suite import them from
# this module, exactly as ``catalog-service.js`` exported them.

__all__ = [
    "CatalogService",
    "ParserError",
    "conversational_intent",
    "entry_text",
    "lexical_similarity",
    "meaningful_words",
    "parse_budget_query",
    "parse_price",
]

logger = logging.getLogger("catalog.service")


# --------------------------------------------------------------------------- #
# Conversational handling
# --------------------------------------------------------------------------- #
# Unchanged from the original, with one addition: the English pattern did not
# recognise "hi there" / "hello there", so a plain greeting fell through to the
# catalog and came back as a refusal. Both forms are now recognised, and the
# mixed-prefix list gained them too so "hi there, how long is shipping?" strips
# the greeting before retrieval.

_CONVERSATIONAL_RESPONSES = {
    "id": {
        "greeting": "Halo! Saya NOVAHAUS Support. Saya bisa membantu pertanyaan tentang produk, pengiriman, garansi, dan dukungan.",
        "thanks": "Dengan senang hati. Ada lagi yang bisa saya bantu?",
        "bye": "Sampai jumpa! Terima kasih sudah menghubungi NOVAHAUS.",
    },
    "en": {
        "greeting": "Hello! I'm NOVAHAUS Support. I can help with products, shipping, warranty, and support questions.",
        "thanks": "You're welcome. Anything else I can help with?",
        "bye": "Goodbye! Thanks for reaching out to NOVAHAUS.",
    },
}

#: Forms that mean "thanks" once filler words are allowed to trail them
#: ("thanks a lot", "thank you so much"). Matched against the normalised query.
_THANKS_TAIL = r"(?:\s+(?:a\s+lot|so\s+much|very\s+much|a\s+bunch|again|mate|ya|sir|ma'?am))*"

_CONVERSATIONAL_PATTERNS = {
    "id": [
        (
            re.compile(
                r"^(halo|hai|hi|hei|hey|selamat pagi|selamat siang|selamat sore|selamat malam)$",
                re.I,
            ),
            "greeting",
        ),
        (re.compile(r"^(terima kasih|makasih|thanks|thank you|thx|makasih banyak)" + _THANKS_TAIL + r"$", re.I), "thanks"),
        (re.compile(r"^(bye|dadah|sampai jumpa|selamat tinggal)$", re.I), "bye"),
    ],
    "en": [
        (
            re.compile(
                r"^(hi|hi there|hello|hello there|hey|hey there|good morning|"
                r"good afternoon|good evening|morning)$",
                re.I,
            ),
            "greeting",
        ),
        (re.compile(r"^(thanks|thank you|thx|cheers|many thanks)" + _THANKS_TAIL + r"$", re.I), "thanks"),
        (re.compile(r"^(bye|goodbye|see you|farewell)$", re.I), "bye"),
    ],
}

#: Words that carry no question on their own. A greeting prefix followed only by
#: these ("thanks a lot") is small talk, not a greeting wrapped around a query —
#: without this guard the prefix stripper retrieved on the remnant "a lot" and
#: answered with unrelated catalog text.
_FILLER_WORDS = frozenset(
    {
        "a", "lot", "so", "much", "very", "bunch", "again", "mate", "friend",
        "sir", "ma", "am", "ya", "you", "there", "then", "all", "guys", "folks",
    }
)

_MIXED_PREFIXES = [
    "terima kasih",
    "thank you",
    "hello there",
    "hi there",
    "hey there",
    "makasih",
    "thanks",
    "hello",
    "halo",
    "hai",
    "hey",
    "hi",
]

_EN_MARKERS = ("the", "is", "are", "does", "do", "how", "what", "when", "where", "can", "you")
_ID_MARKERS = ("apa", "bagaimana", "berapa", "kapan", "dimana", "di mana", "bisa", "saya", "anda")


def _detect_language(text: str) -> str:
    """Return ``"id"`` or ``"en"`` using the original marker-word heuristic."""
    lowered = text.lower()
    id_score = sum(1 for marker in _ID_MARKERS if marker in lowered)
    en_score = sum(1 for marker in _EN_MARKERS if marker in lowered)
    return "id" if id_score > en_score else "en"


def conversational_intent(query: str) -> Optional[Dict[str, Any]]:
    """Handle greetings/thanks/goodbyes and strip mixed-in prefixes.

    Returns ``None`` when the query is not conversational. When a greeting is
    followed by a question (for example ``"hi there, how long is shipping?"``)
    the greeting is returned as ``prefix`` and the remainder as ``query`` so the
    caller can retrieve on the real question.
    """
    trimmed = query.strip()
    if not trimmed:
        return None

    lowered = trimmed.lower()
    language = _detect_language(trimmed)

    # Match against the same normalisation the query cache keys on (lower-cased,
    # trailing punctuation stripped). Without this, "Hello!" / "Thanks." missed
    # the anchored patterns, fell through to retrieval and came back as a
    # refusal — while the cache stored that refusal under the punctuation-free
    # key "hello", so a later plainly-spelled "Hello" was served the wrong
    # answer. questions.json lists "Hello!" as a greeting variant, so the two
    # must be indistinguishable.
    normalized = normalize(trimmed)

    # A bare Indonesian opener ("terima kasih") was refused because the language
    # detector guessed English and the English table holds no Indonesian
    # entries. Try the detected language first, then the other table, so a
    # purely conversational message is recognised whatever the detector guesses.
    others = ["id", "en"]
    others.remove(language)
    for candidate in [language] + others:
        for pattern, kind in _CONVERSATIONAL_PATTERNS[candidate]:
            if pattern.match(normalized):
                return {
                    "answer": _CONVERSATIONAL_RESPONSES[candidate][kind],
                    "query": "",
                }

    for prefix in _MIXED_PREFIXES:
        if lowered == prefix:
            return None
        if lowered.startswith(prefix + " ") or lowered.startswith(prefix + ","):
            remainder = trimmed[len(prefix):].lstrip(" ,").strip()
            if not remainder:
                return None
            # "thanks a lot" is small talk, not a greeting plus the question
            # "a lot": if nothing but filler follows the prefix, answer the
            # greeting/thanks itself instead of retrieving on the remnant.
            remainder_words = meaningful_words(remainder)
            if all(word in _FILLER_WORDS for word in remainder_words):
                return {
                    "answer": _CONVERSATIONAL_RESPONSES[language]["thanks"]
                    if prefix in {"thanks", "thank you", "makasih", "terima kasih"}
                    else _CONVERSATIONAL_RESPONSES[language]["greeting"],
                    "query": "",
                }
            return {
                "prefix": _CONVERSATIONAL_RESPONSES[language]["greeting"].split(".")[0] + ". ",
                "query": remainder,
            }
    return None


# --------------------------------------------------------------------------- #
# Hazard-escalation sources
# --------------------------------------------------------------------------- #
# The catalog states its own electrical-safety policy. For a report such as "my
# smart switch is sparking and there is a burning smell" the correct answer is
# that policy, quoted verbatim. The original returned a generic refusal, which is
# the worst possible outcome for a safety question: it tells the customer no help
# is available instead of escalating.

_HAZARD_SOURCE_RE = re.compile(
    r"Escalate any report involving electrical|"
    r"Do not instruct customers to bypass|"
    r"Do not install smart switches while mains power is live",
    re.IGNORECASE,
)

_HAZARD_RELEVANT_RE = re.compile(
    r"electrical shock|burning smell|smoke|melting|exposed wiring|circuit trip|"
    r"bypass electrical safety|mains-powered|mains power is live|damaged cable|"
    r"qualified electrician|rated load|water",
    re.IGNORECASE,
)

_BULLET_RE = re.compile(r"^[ \t]*[-*][ \t]+(?P<text>.+?)[ \t]*$", re.MULTILINE)


class ParserError(ValueError):
    """Raised for malformed catalog input, mirroring ``ArgumentError`` in JS."""


class CatalogService:
    """In-memory catalog with hybrid retrieval and a deterministic answer path."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        llm: LayaEngine,
        threshold: float,
        top_k: int,
        logger_: Optional[logging.Logger] = None,
        *,
        tie_epsilon: float = 0.05,
        max_tie_nodes: int = 3,
        drop_contentless_nodes: bool = True,
        enable_compound_split: bool = True,
        enable_hazard_escalation: bool = True,
        enable_budget_listing: bool = True,
        answer_mode: str = "hybrid",
        query_cache_size: int = 256,
        synthesizer: Optional[Any] = None,
    ) -> None:
        self.embedder = embedder
        self.llm = llm
        self.threshold = threshold
        self.top_k = top_k
        self.logger = logger_ or logger
        self.drop_contentless_nodes = drop_contentless_nodes
        self.answer_mode = answer_mode
        self.index: List[Dict[str, Any]] = []
        self.cache = QueryCache(max_size=query_cache_size)
        self._hazard_lines: List[str] = []
        self.strategy = AnswerStrategy(
            self,
            threshold=threshold,
            top_k=top_k,
            tie_epsilon=tie_epsilon,
            max_tie_nodes=max_tie_nodes,
            enable_compound=enable_compound_split,
            enable_hazard=enable_hazard_escalation,
            enable_budget=enable_budget_listing,
            answer_mode=answer_mode,
            synthesizer=synthesizer,
        )

    # ----------------------------------------------------------------- ingest #

    def ingest(self, entries: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        """Embed ``entries`` and replace the index. Returns ``{"count": n}``."""
        if not isinstance(entries, (list, tuple)):
            raise ParserError("Document must be an array of entries")

        self.index = []
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("id"):
                raise ValueError("Each entry must have an id")
            text = entry_text(entry).strip()
            if not text:
                continue
            enriched = dict(entry)
            enriched["text"] = text
            enriched["embedding"] = self.embedder.embed(text)
            enriched.setdefault("words", meaningful_words(text))
            enriched["price"] = parse_price(text)
            enriched.update(_annotate(enriched))
            self.index.append(enriched)

        self._rebuild_hazard_lines()
        self.cache.clear()
        self.logger.info("catalog ingested: %d chunks", len(self.index))
        return {"count": len(self.index)}

    def load_index(self, index: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        """Load a pre-built index, recomputing anything that is missing."""
        if not isinstance(index, (list, tuple)):
            raise ParserError("load_index expects a list of catalog entries")

        self.index = []
        for entry in index:
            text = str(entry.get("text", "")).strip()
            if not text:
                continue
            loaded = dict(entry)
            loaded["text"] = text
            loaded["words"] = loaded.get("words") or meaningful_words(text)
            if loaded.get("price") is None:
                loaded["price"] = parse_price(text)
            if not loaded.get("embedding"):
                loaded["embedding"] = self.embedder.embed(text)
            loaded.update(_annotate(loaded))
            self.index.append(loaded)

        self._rebuild_hazard_lines()
        self.cache.clear()
        self.logger.info("catalog index loaded: %d chunks", len(self.index))
        return {"count": len(self.index)}

    # ------------------------------------------------------------------ query #

    def query(self, query: str) -> Dict[str, Any]:
        """Answer ``query`` and return the original response shape."""
        return self.query_with_diagnostics(query)[0]

    def query_with_diagnostics(self, query: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Answer ``query`` and also report how the answer was produced.

        The second element is diagnostic only: the path taken
        (``deterministic``, ``tie-expansion``, ``hazard-escalation``,
        ``compound``, ``budget-filter``, ``conversational``, ``refused``,
        ``synthesizer``, ``layap`` or ``degraded``), the wall-clock latency, and
        the ids of the nodes used.
        """
        started = time.perf_counter()
        trimmed = query.strip() if isinstance(query, str) else ""
        if not trimmed:
            raise ValueError("query must be a non-empty string")

        cached = self.cache.get(trimmed)
        if cached is not None:
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            return dict(cached), {
                "path": "cache",
                "cached": True,
                "latencyMs": elapsed,
                "nodes": [],
                "detail": "served from the normalised-query cache",
            }

        intent = conversational_intent(trimmed)
        if intent and intent.get("answer"):
            response: Dict[str, Any] = {
                "matched": True,
                "answer": intent["answer"],
                "score": 1,
            }
            diagnostics = {
                "path": "conversational",
                "cached": False,
                "nodes": [],
                "detail": "small talk handled without retrieval",
            }
        else:
            if not self.index:
                raise ValueError("Catalog has not been ingested")
            knowledge_query = str(intent.get("query") or trimmed) if intent else trimmed
            prefix = intent.get("prefix", "") if intent else ""
            resolution = self.strategy.resolve(knowledge_query, self._retrieve, prefix=prefix)
            response = resolution.response
            diagnostics = {
                "path": resolution.path,
                "cached": False,
                "nodes": list(resolution.nodes),
                "detail": resolution.detail,
            }

        elapsed = round((time.perf_counter() - started) * 1000, 3)
        diagnostics["latencyMs"] = elapsed
        diagnostics["score"] = response.get("score")
        self.cache.put(trimmed, response)
        self.logger.info(
            "query=%r path=%s score=%.4f latency=%.1fms nodes=%s",
            trimmed,
            diagnostics["path"],
            float(response.get("score") or 0.0),
            elapsed,
            diagnostics["nodes"] or "-",
        )
        return response, diagnostics

    # -------------------------------------------------------------- retrieval #

    def _retrieve(self, query: str) -> Retrieval:
        """Score every chunk against ``query`` and return the ranked shortlist."""
        query_embedding = self.embedder.embed(query)
        query_words = meaningful_words(query)

        matches: List[Dict[str, Any]] = []
        for entry in self.index:
            cosine = cosine_similarity(query_embedding, entry["embedding"])
            lexical = lexical_similarity(query_words, entry.get("words") or set())
            matches.append({**entry, "score": max(cosine, 0.7 * cosine + 0.3 * lexical)})

        matches.sort(key=lambda item: item["score"], reverse=True)

        budget = parse_budget_query(query)
        budget_matches: List[Dict[str, Any]] = []
        if budget:
            ceiling = budget["maxPrice"]
            budget_matches = sorted(
                (
                    {**match, "budget_max": ceiling}
                    for match in matches
                    if match.get("price") is not None
                    and match["price"] <= ceiling
                    and product_name(match)
                    and not match.get("contentless")
                ),
                key=lambda item: item["price"] or 0,
            )

        return Retrieval(
            matches=matches,
            budget_matches=budget_matches,
            budget_max=budget["maxPrice"] if budget else None,
        )

    # ------------------------------------------------------------- escalation #

    def escalation_answer(self) -> str:
        """Return the catalog's own electrical-safety escalation policy.

        Quoted verbatim from the source documents. Returns ``""`` when the corpus
        carries no escalation material, in which case the caller falls back to
        normal retrieval.
        """
        if not self._hazard_lines:
            return ""
        bullets = "\n".join(f"- {line}" for line in self._hazard_lines)
        return (
            "This sounds like an electrical-safety issue, so this is being "
            "escalated rather than troubleshot. NOVAHAUS's escalation rules for "
            "this kind of report are:\n\n"
            f"{bullets}\n\n"
            "Please contact NOVAHAUS support about this."
        )

    def _rebuild_hazard_lines(self) -> None:
        """Collect the escalation and safety bullets used by hazard answers."""
        collected: List[str] = []
        seen = set()
        for entry in self.index:
            content = probe_text(entry)
            if not _HAZARD_SOURCE_RE.search(content):
                continue
            for bullet in _BULLET_RE.findall(content):
                line = bullet.strip()
                if not line or line in seen or not _HAZARD_RELEVANT_RE.search(line):
                    continue
                seen.add(line)
                collected.append(line)
        self._hazard_lines = collected


def _annotate(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Add the derived fields the answering strategy relies on.

    ``answer_text`` is the rendered, loader-scaffolding-free answer; ``contentless``
    marks heading-only nodes that must never win a retrieval.
    """
    return {
        "answer_text": render_answer(entry),
        "label": node_label(entry),
        "product_name": product_name(entry),
        "contentless": is_contentless(probe_text(entry)),
    }
