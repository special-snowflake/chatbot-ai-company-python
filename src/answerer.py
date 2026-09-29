"""Answering strategy — deterministic first, generative last.

Port addition, and the heart of the reliability/speed rework.

WHY THIS EXISTS
---------------
The ported pipeline had exactly one answering path: retrieve the top-``k``
chunks, hand them to the LAYA engine, return whatever came back. Measuring that
path against the NOVAHAUS corpus exposed four problems:

1. **LAYAA selects, it does not synthesise.** LAYA is a non-autoregressive
   *classification* engine: it returns an index into the context it was given and
   the service echoes that one item. So ``compare the Glow Bulb A19 and the Glow
   Strip 2M`` could only ever answer about one of them, and a question with two
   halves lost one half. No amount of prompt tuning fixes that — the model
   cannot emit new text.
2. **It sometimes selected a node with no content.** ``source_loader`` splits
   markdown at every heading, so the first node of a document can be its title
   alone. That node embeds close to any question about the document, and
   ``what is the company about?`` came back as the literal document heading.
3. **It cost 2.5 GB and 3-6.5 s per question.** CPU time ran a consistent 2.0x
   wall time, pegging both physical cores of the test machine.
4. **A confident retrieval still paid that price.** The right answer was already
   sitting at rank 1; a 3-6.5 s model call was spent re-choosing it.

WHAT THIS DOES INSTEAD
----------------------
Retrieval stays exactly as it was (it was working). The *answering* decision is
rebuilt as a short circuit chain in which the expensive path is the last resort:

    hazard report  -> verbatim escalation policy from the corpus
    budget query   -> the matching products, priced
    below the gate -> refuse (unchanged from the JS service)
    compound       -> answer each half, then stitch
    near-tie       -> return every node inside the tie window
    clear winner   -> return that node verbatim   [no model call]
    otherwise      -> optional synthesizer, then LAYA

Every answer on the deterministic paths is copied word-for-word out of the
source documents, which is what the project's anti-hallucination rule requires.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from .answer_text import REFUSAL, node_label, product_name, render_answer
from .intent import MAX_COMPOUND_PARTS, hazard_intent, is_comparison, split_compound
from .laya_engine import NO_MATCH
from .lexical import meaningful_words

logger = logging.getLogger("catalog.answerer")

#: Never stitch more than this many nodes into a tie-expanded answer.
MAX_TIE_NODES = 3
#: Never list more than this many products for a budget query.
MAX_BUDGET_NODES = 12

#: Above this score the question matched specifically, so its single best node is
#: the whole answer. Without it, asking about one delivery region would be
#: answered with every region.
STRONG_MATCH = 0.70
#: Wider window used only to recognise variant families. A family is returned
#: whole even when some of its members fall outside the tie window.
FAMILY_WINDOW = 0.12
#: Containment required for two nodes to count as the same question.
VARIANT_CONTAINMENT = 0.60
#: Hard cap on nodes stitched into a single reply.
MAX_FAMILY_NODES = 5


def _stem(node: Dict[str, Any]) -> Set[str]:
    """Content words of a node's question, falling back to its label."""
    question = node.get("question")
    if isinstance(question, str) and question.strip():
        return meaningful_words(question)
    label = node.get("label") or node_label(node)
    return meaningful_words(label) if label else set()


def _same_stem(left: Set[str], right: Set[str]) -> bool:
    """Whether two stems are the same question with a different parameter.

    Containment, not Jaccard: a variant adds qualifier words ("within Java" vs
    "in Jabodetabek"), which would drag a symmetric ratio below the bar.
    """
    if not left or not right:
        return False
    return len(left & right) / min(len(left), len(right)) >= VARIANT_CONTAINMENT


#: Pronouns that signal a follow-up clause which dropped its subject.
_ELLIPTICAL_RE = re.compile(
    r"\b(?:it|its|it's|they|them|their|theirs|this|that|these|those)\b",
    re.IGNORECASE,
)


def _bind_retriever(retrieve: Retriever, retrieval_query: str) -> Retriever:
    """Return a retriever that always scores ``retrieval_query``.

    Used to give an elliptical clause its subject back without rewriting the
    clause itself, so the answer is still reported against the user's wording.
    """

    def bound(_query: str) -> Retrieval:
        return retrieve(retrieval_query)

    return bound


@dataclass
class Retrieval:
    """Scored nodes for one query, plus any budget-filtered products.

    ``budget_matches`` mirrors the JS service's ``budgetMatches``: entries whose
    parsed price is at or below the ceiling the query asked for. Unlike the
    original — which passed them to the model as extra context — these are used
    directly, because *every* one of them is a valid answer.
    """

    matches: List[Dict[str, Any]] = field(default_factory=list)
    budget_matches: List[Dict[str, Any]] = field(default_factory=list)
    budget_max: Optional[float] = None


#: A callable that scores the catalog for one query: ``query -> Retrieval``.
Retriever = Callable[[str], Retrieval]


@dataclass
class Resolution:
    """The API response plus a record of how it was produced."""

    response: Dict[str, Any]
    path: str
    detail: str = ""
    nodes: List[str] = field(default_factory=list)


class AnswerStrategy:
    """Chooses how a retrieved query gets answered.

    The strategy never touches embeddings or scoring itself: the owning
    :class:`~src.catalog_service.CatalogService` supplies a ``retrieve`` callable
    so that compound questions can run their own retrieval per sub-question.
    """

    #: Decision paths, surfaced in logs and in the evaluation report.
    PATH_HAZARD = "hazard-escalation"
    PATH_BUDGET = "budget-filter"
    PATH_COMPOUND = "compound"
    PATH_TIE = "tie-expansion"
    PATH_DETERMINISTIC = "deterministic"
    PATH_REFUSED = "refused"
    PATH_SYNTHESIZER = "synthesizer"
    PATH_LAYAP = "layap"
    PATH_DEGRADED = "degraded"

    #: ``answer_mode`` values that fall straight through to the generative path.
    _LEGACY_MODES = frozenset({"legacy", "layap", "laya"})

    def __init__(
        self,
        service: Any,
        *,
        threshold: float,
        top_k: int,
        tie_epsilon: float = 0.05,
        max_tie_nodes: int = MAX_TIE_NODES,
        enable_compound: bool = True,
        enable_hazard: bool = True,
        enable_budget: bool = True,
        answer_mode: str = "hybrid",
        synthesizer: Any = None,
    ) -> None:
        self.service = service
        self.threshold = float(threshold)
        self.top_k = int(top_k)
        self.tie_epsilon = float(tie_epsilon)
        self.max_tie_nodes = int(max_tie_nodes)
        self.enable_compound = bool(enable_compound)
        self.enable_hazard = bool(enable_hazard)
        self.enable_budget = bool(enable_budget)
        self.answer_mode = str(answer_mode).strip().lower() or "hybrid"
        self.synthesizer = synthesizer

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def resolve(
        self,
        query: str,
        retrieve: Retriever,
        prefix: str = "",
        depth: int = 0,
    ) -> Resolution:
        """Answer ``query`` using ``retrieve`` for scoring.

        ``depth`` guards the compound-recursion: sub-questions are resolved with
        ``depth=1`` and therefore never split again.
        """
        retrieval = retrieve(query)
        matches = retrieval.matches
        if not matches:
            return self._refusal(0.0, detail="empty catalog")

        score = float(matches[0]["score"])
        usable = [node for node in matches if not node.get("contentless")]

        # 1. Safety first. A report of sparking, smoke or a burning smell must
        #    never be gated on how well it scored against the corpus.
        hazard = self._hazard(query, score, prefix)
        if hazard is not None:
            return hazard

        # 2. Budget queries ("under 500k") bypass the similarity gate in the JS
        #    service; every product under the ceiling is a correct answer.
        budget = self._budget(retrieval, score, prefix)
        if budget is not None:
            return budget

        # 3. Unchanged similarity gate from the original service.
        if score < self.threshold:
            return self._refusal(score, detail=f"score {score:.4f} < {self.threshold:g}")

        # 4. Everything retrieved was scaffolding (headings, labels).
        if not usable:
            return self._refusal(score, detail="all nodes failed the content guard")

        # 5. Compound question: answer each half on its own terms.
        if self.enable_compound and depth == 0:
            parts = split_compound(query)
            if parts:
                return self._compound(query, parts, retrieve, prefix, score)

        # Legacy mode exists only so the original pipeline can be benchmarked
        # against this one (see scripts/run_eval.py): it skips the deterministic
        # paths and always calls the generative engine, exactly as the JS
        # service did.
        if self.answer_mode != "hybrid":
            return self._delegate(query, usable, prefix, score)

        # 6. Near-tie: several nodes explain the query about equally well.
        tie = self._tie(usable, score, prefix)
        if tie is not None:
            return tie

        # 7. Clear winner: answer without paying for a model call.
        confident = self._deterministic(usable[0], score, prefix)
        if confident is not None:
            return confident

        # 8. Nothing renderable — fall through to the generative paths.
        return self._delegate(query, usable, prefix, score)

    # ------------------------------------------------------------------ #
    # Steps
    # ------------------------------------------------------------------ #

    def _hazard(self, query: str, score: float, prefix: str) -> Optional[Resolution]:
        """Escalate electrical-safety reports using the corpus's own rules."""
        if not self.enable_hazard:
            return None
        term = hazard_intent(query)
        if not term:
            return None
        getter = getattr(self.service, "escalation_answer", None)
        escalation = getter() if callable(getter) else ""
        if not escalation:
            return None
        logger.info("hazard query escalated (term=%r, matched=True)", term)
        return Resolution(
            self._answer(f"{prefix}{escalation}", score),
            self.PATH_HAZARD,
            f"matched hazard term {term!r}",
        )

    def _budget(self, retrieval: Retrieval, score: float, prefix: str) -> Optional[Resolution]:
        """List every real product at or below the ceiling the query named."""
        if not self.enable_budget:
            return None
        if retrieval.budget_max is None:
            return None
        products = [
            node
            for node in retrieval.budget_matches
            if node.get("product_name") and not node.get("contentless")
        ]
        ceiling = int(retrieval.budget_max)
        if not products:
            # The ceiling is below the cheapest product. Naming the real floor is
            # the honest reply; returning None here would let an unrelated node
            # (a delivery window, a warranty term) answer a price question.
            floor = self._cheapest_product()
            if floor is None or not floor.get("price"):
                return None
            floor_name = floor.get("product_name") or node_label(floor)
            answer = (
                f"No NOVAHAUS product is priced at or under Rp{ceiling:,}. "
                f"The lowest-priced product in the catalog is the {floor_name} "
                f"at Rp{int(floor['price']):,}."
            )
            return Resolution(
                self._answer(f"{prefix}{answer}", score),
                self.PATH_BUDGET,
                f"no product at or under Rp{ceiling:,}; floor is {floor_name}",
                [str(floor["id"])],
            )

        listed = products[:MAX_BUDGET_NODES]
        lines = []
        for node in listed:
            name = node.get("product_name") or node_label(node)
            price = node.get("price")
            lines.append(f"- {name}: Rp{int(price):,}" if price else f"- {name}")
        if len(products) > len(listed):
            lines.append(f"- ...and {len(products) - len(listed)} more")

        ceiling = int(retrieval.budget_max or 0)
        answer = f"Catalog products at or under Rp{ceiling:,}:\n" + "\n".join(lines)
        logger.info("budget query answered (products=%d, matched=True)", len(products))
        return Resolution(
            self._answer(f"{prefix}{answer}", score),
            self.PATH_BUDGET,
            f"{len(products)} product(s) under Rp{ceiling:,}",
            [str(node["id"]) for node in listed],
        )

    def _cheapest_product(self) -> Optional[Dict[str, Any]]:
        """The lowest-priced catalog product, or ``None`` if none is priced.

        Used only for below-floor budget replies. ``price`` is parsed from the
        node text at ingest time, and ``product_name`` is set only for real
        ``### Name`` catalog sections, so FAQ entries that merely mention a
        price are not mistaken for products.
        """
        priced = [
            node
            for node in getattr(self.service, "index", [])
            if node.get("product_name") and node.get("price")
        ]
        if not priced:
            return None
        return min(priced, key=lambda node: float(node["price"]))

    def _compound(
        self,
        query: str,
        parts: List[str],
        retrieve: Retriever,
        prefix: str,
        score: float,
    ) -> Resolution:
        """Answer each sub-question separately and stitch the results."""
        labelled = is_comparison(query)
        blocks: List[str] = []
        missing: List[str] = []

        for index, part in enumerate(parts[:MAX_COMPOUND_PARTS]):
            # A follow-up half frequently drops its subject ("...and how much it
            # costs"). Scoring that fragment on its own retrieves nothing useful,
            # so blend it with the first part, which carries the subject.
            part_retrieve = retrieve
            if index and _ELLIPTICAL_RE.search(part):
                part_retrieve = _bind_retriever(retrieve, f"{part} {parts[0]}")
            sub = self.resolve(part, part_retrieve, prefix="", depth=1)
            if sub.response.get("matched"):
                text = str(sub.response.get("answer", "")).strip()
                if text:
                    blocks.append(f"**{part}**\n{text}" if labelled else text)
                    continue
            missing.append(part)

        if not blocks:
            return self._refusal(score, detail="no compound part answerable")

        if missing:
            listed = "; ".join(f'"{part}"' for part in missing)
            blocks.append(f"The catalog does not contain information about {listed}.")

        logger.info(
            "compound query answered (parts=%d, unanswered=%d, matched=True)",
            len(parts),
            len(missing),
        )
        return Resolution(
            self._answer(f"{prefix}" + "\n\n".join(blocks), score),
            self.PATH_COMPOUND,
            f"{len(parts) - len(missing)}/{len(parts)} parts answered",
        )

    def _tie(self, usable: List[Dict[str, Any]], score: float, prefix: str) -> Optional[Resolution]:
        """Answer with every node the question could plausibly mean.

        Two mechanisms, because either alone is insufficient:

        * the **tie window** returns nodes scoring within ``tie_epsilon`` of the
          best, which covers answers that are genuinely interchangeable; and
        * the **variant family** returns every node that is the same question with
          one parameter changed, even when the best-scoring node is a different
          one -- such as FAQ 8's "delivery can take longer than the estimate"
          caveat outranking the three regional answers.

        The family is what fixes the shipping-times case: answering with a single
        region states a regional fact as if it were universal. A question that
        matched strongly is answered from its single best node instead, so asking
        about one region does not return all of them.
        """
        if score >= STRONG_MATCH:
            return None

        window = [node for node in usable if score - float(node["score"]) <= self.tie_epsilon]
        window = window[: self.max_tie_nodes]

        near = [node for node in usable if score - float(node["score"]) <= FAMILY_WINDOW]
        family = self._variant_family(near)

        chosen: List[Dict[str, Any]] = list(window)
        for node in family:
            if not any(node["id"] == seen["id"] for seen in chosen):
                chosen.append(node)
        chosen = chosen[:MAX_FAMILY_NODES]
        if len(chosen) < 2:
            return None

        text = self._stitch(chosen)
        if not text:
            return None
        logger.info(
            "tie expanded (nodes=%d, window=%g, family=%d, matched=True)",
            len(chosen),
            self.tie_epsilon,
            len(family),
        )
        return Resolution(
            self._answer(f"{prefix}{text}", score),
            self.PATH_TIE,
            f"{len(chosen)} nodes within {self.tie_epsilon:g} (family {len(family)})",
            [str(node["id"]) for node in chosen],
        )

    def _variant_family(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """The largest group of candidates that share a question stem."""
        best: List[Dict[str, Any]] = []
        for anchor in candidates:
            stem = _stem(anchor)
            if not stem:
                continue
            group = [node for node in candidates if _same_stem(stem, _stem(node))]
            if len(group) > len(best):
                best = group
        return best if len(best) > 1 else []

    def _deterministic(
        self, node: Dict[str, Any], score: float, prefix: str
    ) -> Optional[Resolution]:
        """Return the winning node verbatim — no model call at all."""
        text = str(node.get("answer_text") or render_answer(node)).strip()
        if not text:
            return None
        return Resolution(
            self._answer(f"{prefix}{text}", score),
            self.PATH_DETERMINISTIC,
            f"node {node.get('id')} above the gate",
            [str(node["id"])],
        )

    # ------------------------------------------------------------------ #
    # Generative fallbacks
    # ------------------------------------------------------------------ #

    def _delegate(
        self,
        query: str,
        nodes: List[Dict[str, Any]],
        prefix: str,
        score: float,
    ) -> Resolution:
        """Hand the question to the synthesizer, else to LAYA."""
        if self.answer_mode in self._LEGACY_MODES:
            return self._legacy(query, nodes, prefix, score)

        synthesizer = self.synthesizer
        if synthesizer is not None and synthesizer.available():
            items = [(str(node["text"]), float(node["score"])) for node in nodes[: self.top_k]]
            text = synthesizer.answer(query, items)
            if text:
                logger.info(
                    "answer synthesised via %s (matched=True)",
                    getattr(synthesizer, "model", "synthesizer"),
                )
                return Resolution(
                    self._answer(f"{prefix}{text}", score),
                    self.PATH_SYNTHESIZER,
                    f"model {getattr(synthesizer, 'model', 'unknown')!r}",
                )
        return self._legacy(query, nodes, prefix, score)

    def _legacy(
        self,
        query: str,
        nodes: List[Dict[str, Any]],
        prefix: str,
        score: float,
    ) -> Resolution:
        """The original single path: context in, LAYA's choice out.

        Retained for parity and used as the baseline in the evaluation, so the
        before/after numbers are produced by the same code path the JS service
        used.
        """
        context_nodes = nodes[: self.top_k]
        try:
            answer = self.service.llm.answer(
                context="\n\n".join(
                    f"Context item {index + 1} (relevance {node['score']:.3f}):\n{node['text']}"
                    for index, node in enumerate(context_nodes)
                ),
                query=query,
                context_texts=[node["text"] for node in context_nodes],
            ).strip()

            if not answer or answer == NO_MATCH:
                logger.info("layap rejected the query (score=%.4f, matched=False)", score)
                return self._refusal(score, path=self.PATH_LAYAP, detail="layap returned NO_MATCH")

            logger.info("layap answered (score=%.4f, matched=True)", score)
            return Resolution(self._answer(f"{prefix}{answer}", score), self.PATH_LAYAP)
        except Exception as error:  # pragma: no cover - defensive parity with JS catch
            logger.error("layap failed (%s, score=%.4f); returning raw match", error, score)
            return Resolution(
                {
                    "matched": True,
                    "answer": nodes[0]["text"],
                    "score": score,
                    "degraded": True,
                },
                self.PATH_DEGRADED,
                str(error),
            )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _stitch(self, nodes: List[Dict[str, Any]]) -> str:
        """Join several node answers, each under its own label."""
        blocks: List[str] = []
        for node in nodes:
            text = str(node.get("answer_text") or render_answer(node)).strip()
            if not text:
                continue
            label = node.get("label") or node_label(node)
            blocks.append(f"**{label}**\n{text}" if label else text)
        return "\n\n".join(blocks)

    @staticmethod
    def _answer(text: str, score: float) -> Dict[str, Any]:
        """Build a matched response, identical in shape to the JS service."""
        return {"matched": True, "answer": text, "score": score}

    def _refusal(
        self,
        score: float,
        path: Optional[str] = None,
        detail: str = "",
    ) -> Resolution:
        """Build the unchanged refusal envelope from ``catalog-service.js``."""
        return Resolution(
            {"matched": False, "answer": REFUSAL, "score": score},
            path or self.PATH_REFUSED,
            detail,
        )
