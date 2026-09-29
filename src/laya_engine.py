"""LAYAP decision engine — replaces the original LLM provider.

Python port of ``src/llm.js`` from ``special-snowflake/chatbot-ai-company``.

WHAT CHANGED AND WHY (this is the core of the port)
---------------------------------------------------
The original ``src/llm.js`` loaded a local ``.gguf`` model through
``node-llama-cpp`` and used ``LlamaChatSession`` to *generate* a free-text
answer from the retrieved context, returning the literal string ``NO_MATCH``
when the context did not answer the question.

There is no component named "Jev AI" anywhere in that repository (verified:
``grep -ri jev`` over the clone returns zero matches). The thing being replaced
is the generative ``node-llama-cpp`` provider.

LAYAP (``laya``) is a **non-autoregressive System 1 decision engine**: it does
not generate prose. Per its own API it answers exactly three question types —
``choice``, ``score`` and ``noul`` (a calibrated yes/no probability). So the
port is honest about what it can and cannot do:

* LAYA decides **whether** the context answers the question (``noul``) and
  **which** context item wins (``choice``). This is the decision the old LLM
  was implicitly making.
* The answer **text** is extracted from the winning catalog item, because LAYA
  cannot synthesise a novel sentence. The original's ``NO_MATCH`` contract is
  preserved exactly.

Public surface is identical to the JS provider: ``await llm.answer({context,
query})`` becomes ``engine.answer(context=..., query=...)``.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger("catalog.laya")

# Sentinel returned when the engine decides the context does not answer the query.
NO_MATCH = "NO_MATCH"

try:  # Import at module scope so a missing dependency fails loudly, not mid-request.
    from laya.router import DEFAULT_MODELS as _LAYAP_DEFAULT_MODELS
    from laya.router import Router as _LayaRouter
except Exception:  # pragma: no cover - surfaced in the server's startup log
    _LayaRouter = None  # type: ignore[assignment]
    _LAYAP_DEFAULT_MODELS = {}  # type: ignore[assignment]

#: Checkpoint families shipped by LAYAP (``english``, ``multilingual``, ``typed-decisions``).
KNOWN_LAYAP_MODELS = frozenset(_LAYAP_DEFAULT_MODELS) or frozenset(
    {"english", "multilingual", "typed-decisions"}
)


class LayaEngine:
    """Decision engine backed by LAYA's ``Router``.

    Args:
        model: LAYA checkpoint family name (``config.laya_model``).
        timeout_ms: Unused for wall-clock timing (LAYAP is synchronous and
            fast) but kept so the constructor signature matches ``createLlmProvider``.
        min_match_probability: Calibrated ``noul`` probability below which the
            engine returns :data:`NO_MATCH`. Defaults to ``0.5``.
    """

    def __init__(
        self,
        model: str = "english",
        timeout_ms: int = 30000,
        min_match_probability: float = 0.5,
    ) -> None:
        if _LayaRouter is None:
            raise RuntimeError(
                "LAYAP is not installed. Install it with: python -m pip install laya"
            )
        self.model = model
        self.timeout_ms = timeout_ms
        self.min_match_probability = min_match_probability
        self._lock = threading.Lock()  # Router loads checkpoints lazily; serialise first use.

        # Two LAYA behaviours have to be worked around here, and both are silent:
        #
        # 1. ``Router()`` ignores the checkpoint you asked for. Its routing precedence is
        #    explicit ``model`` > ``task`` > workflow > ``lang`` > ``lang_guess`` > *detected
        #    language* > ``default``. Detected English text returns ``key = "english"`` before
        #    ``default`` is ever consulted, so ``Router(default="multilingual")`` still runs the
        #    English checkpoint on English input — byte-identical output.
        #    The checkpoint is therefore pinned per call via ``predict(model=...)``.
        # 2. The built-in ``DEFAULT_MODELS`` map points ``multilingual`` / ``typed-decisions`` at
        #    revisions of the *same* repo that do not exist on the Hub
        #    (``revision/multilingual`` -> HTTP 404 "Invalid rev id"), so LAYA would fall back to
        #    ``main``. The distinct checkpoints are published as standalone repos, selected with
        #    ``standalone_repos=True`` (``convaiinnovations/laya-multilingual`` etc.).
        if model in KNOWN_LAYAP_MODELS:
            self._model_key: Optional[str] = model
            router_kwargs: Dict[str, Any] = {"standalone_repos": True, "default": model}
        else:
            # Unknown name: treat it as a HuggingFace repo id for the default slot.
            self._model_key = None
            router_kwargs = {"models": {"english": model}, "default": "english"}
        logger.info(
            "LAYAP router initialised (requested=%s, pinned_checkpoint=%s, standalone_repos=%s)",
            model,
            self._model_key or "(router decides)",
            router_kwargs.get("standalone_repos", False),
        )
        self._router = _LayaRouter(**router_kwargs)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _predict(self, state: str, questions: Dict[str, Any]) -> Dict[str, Any]:
        """Run one synchronous LAYA forward pass, serialised on the shared router.

        ``model`` is pinned when :data:`KNOWN_LAYAP_MODELS` named the checkpoint, because
        LAYA's language auto-detection would otherwise override it for English text.
        """
        kwargs: Dict[str, Any] = {}
        if self._model_key is not None:
            kwargs["model"] = self._model_key
        with self._lock:
            return self._router.predict(state, questions, **kwargs)

    def _select_context_index(self, query: str, context_items: List[str]) -> int:
        """Use a LAYA ``choice`` question to pick the best context item.

        Returns the 0-based index of the winning item, defaulting to ``0``.
        """
        criteria = {str(i + 1): item for i, item in enumerate(context_items)}
        state = f"User question: {query}\n\n" + "\n\n".join(
            f"Context item {i + 1}: {item}" for i, item in enumerate(context_items)
        )
        result = self._predict(
            state,
            {
                "best_context": {
                    "type": "choice",
                    "instructions": (
                        "Which context item best answers the user question? "
                        "Answer with the item number."
                    ),
                    "criteria": criteria,
                }
            },
        )
        choice = result.get("answers", {}).get("best_context", {}).get("choice")
        try:
            return max(0, int(choice) - 1)
        except (TypeError, ValueError):
            return 0

    def _context_answers_query(self, query: str, context: str) -> float:
        """Use a LAYA ``noul`` question for the calibrated match probability."""
        result = self._predict(
            f"Context:\n{context}\n\nUser question:\n{query}",
            {
                "matched": {
                    "type": "noul",
                    "instructions": (
                        "Does the context contain the information needed to answer "
                        "the user question?"
                    ),
                    "criteria": {
                        "true": "the context answers the question",
                        "false": "the context does not answer the question",
                    },
                }
            },
        )
        answer = result.get("answers", {}).get("matched", {})
        # ``noul`` returns a calibrated probability in [0, 1].
        return float(answer.get("noul", 0.0))

    # ------------------------------------------------------------------ #
    # Public API — mirrors ``llm.answer({ context, query })``
    # ------------------------------------------------------------------ #

    def answer(
        self,
        context: str,
        query: str,
        context_texts: Optional[List[str]] = None,
    ) -> str:
        """Return an answer string for ``query`` grounded in ``context``.

        Mirrors the JS contract: returns :data:`NO_MATCH` when the context does
        not answer the question, otherwise the answer text.
        """
        texts = context_texts or [context]

        probability = self._context_answers_query(query, context)
        logger.info(
            "LAYAP evaluated context relevance (match_probability=%.4f)", probability
        )
        if probability < self.min_match_probability:
            return NO_MATCH

        index = self._select_context_index(query, texts) if len(texts) > 1 else 0
        return texts[index].strip()


def create_laya_provider(
    model: str,
    timeout_ms: int,
    min_match_probability: float = 0.5,
) -> LayaEngine:
    """Factory mirroring ``createLlmProvider({ modelPath, timeoutMs })``."""
    return LayaEngine(model=model, timeout_ms=timeout_ms, min_match_probability=min_match_probability)
