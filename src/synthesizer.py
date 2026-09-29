"""Optional LLM synthesizer.

Port addition, and the only component in this project that talks to the network
by choice. It exists because the LAYA engine *selects* a context item and echoes
it; it cannot write a sentence that merges several facts. Stitching several
nodes together (see :mod:`src.answerer`) covers most of that gap without any
model at all, but genuinely open-ended requests read better as prose.

Design constraints, in order of importance:

* **Off by default.** The service runs fully locally unless
  ``SYNTHESIZER_BASE_URL`` is configured, so a default install costs nothing and
  needs no key.
* **No new dependencies.** Uses :mod:`urllib.request` from the standard library
  rather than pulling in an HTTP client.
* **Never a single point of failure.** Every error path returns ``None``, which
  tells the caller to fall through to the next answerer.
* **Grounded.** The system prompt forbids using anything outside the supplied
  context and requires the exact refusal string when the context is silent, so an
  unavailable or misbehaving model degrades to a refusal rather than to an
  invented answer.

Compatible with any OpenAI-style ``POST {base_url}/chat/completions`` endpoint,
which covers the free tiers of the usual providers as well as a local server.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from .answer_text import REFUSAL

logger = logging.getLogger("catalog.synthesizer")

#: Grounding rules for every request. Mirrors the honesty contract used across
#: this port: no outside knowledge, no guessing, and an explicit refusal.
SYSTEM_PROMPT = (
    "You answer customer-support questions about the NOVAHAUS fictional catalog. "
    "Use ONLY the numbered context items provided. Never use outside knowledge and "
    "never invent specifications, prices, or policies. If the context does not "
    f"contain the answer, reply with exactly: {REFUSAL} "
    "When the question has several parts, answer every part that the context "
    "covers and say plainly which parts it does not cover. Be concise and plain."
)


class ApiSynthesizer:
    """An OpenAI-compatible chat-completions synthesizer.

    Args:
        base_url: Endpoint root, e.g. ``https://api.example.com/v1``. When empty
            the synthesizer reports :meth:`available` as ``False``.
        model: Model name to request.
        api_key: Bearer token. Many free/local endpoints accept any placeholder.
        timeout_ms: Per-request timeout.
    """

    def __init__(
        self,
        base_url: str = "",
        model: str = "",
        api_key: str = "",
        timeout_ms: int = 30000,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_ms = timeout_ms
        self.call_count = 0
        self.error_count = 0

    # ------------------------------------------------------------------ #

    def available(self) -> bool:
        """Whether the synthesizer is configured well enough to be attempted."""
        return bool(self.base_url and self.model)

    def answer(
        self,
        query: str,
        context_items: List[Tuple[str, float]],
        context: str = "",
    ) -> Optional[str]:
        """Return a synthesized answer, or ``None`` so the caller can fall through.

        Args:
            query: The user's question.
            context_items: ``(text, score)`` pairs for the retrieved catalog
                chunks, in rank order. Scores are shown to the model because
                relevance is a useful tie-break signal for it.
            context: Pre-formatted context. Rebuilt from ``context_items`` when
                omitted.
        """
        if not self.available():
            return None

        if not context:
            context = "\n\n".join(
                f"Context item {index + 1} (relevance {score:.3f}):\n{text}"
                for index, (text, score) in enumerate(context_items)
            )

        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"{context}\n\nQuestion: {query}\n\nAnswer:",
                },
            ],
        }

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key or 'not-required'}",
            },
            method="POST",
        )

        self.call_count += 1
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_ms / 1000) as response:
                body = json.loads(response.read().decode("utf-8"))
            choices = body.get("choices") or []
            if not choices:
                raise ValueError("response contained no choices")
            message = choices[0].get("message") or {}
            text = (message.get("content") or "").strip()
            return text or None
        except (urllib.error.URLError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            # Deliberately broad: a synthesizer failure must never fail a request.
            self.error_count += 1
            logger.warning("synthesizer call failed (%s); falling back", error)
            return None

    def stats(self) -> Dict[str, int]:
        """Return a snapshot of call counters."""
        return {"calls": self.call_count, "errors": self.error_count}
