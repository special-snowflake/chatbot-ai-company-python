"""Tests for the deterministic answering strategy.

Port addition. ``test/test_api.py`` covers the ported HTTP contract; this file
covers what was added on top of it: the deterministic path, the contentless-node
guard, hazard escalation, compound questions, variant families, budget listings
and the query cache.

Each test asserts *how* an answer was produced, not merely that one came back, so
a silent regression to "ask the model about everything" fails the run rather than
passing quietly.
"""

from __future__ import annotations

import math
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.catalog_service import CatalogService  # noqa: E402
from src.intent import is_comparison, split_compound  # noqa: E402

SHIPPING = [
    {
        "id": "shipping",
        "question": "How long is shipping?",
        "answer": "Shipping takes 3 days.",
    },
    {
        "id": "returns",
        "question": "What is the return policy?",
        "answer": "Returns are accepted within 30 days.",
    },
]

HEADING_ONLY = [
    {
        "id": "doc-heading",
        "title": "NOVAHAUS_Company_Information",
        "content": "# NOVAHAUS Company Information & Corporate Knowledge Base",
    },
    {
        "id": "overview",
        "title": "NOVAHAUS_Company_Information",
        "content": (
            "## Company Overview\n"
            "NOVAHAUS is a company based in Jakarta building smart-home products."
        ),
    },
]

HAZARD = [
    {
        "id": "switch-help",
        "question": "Can I install a smart switch myself?",
        "answer": "Qualified electrician work is recommended.",
    },
    {
        "id": "escalation-rules",
        "title": "NOVAHAUS_FAQ",
        "content": (
            "## 13. Escalation Rules\n"
            "- Escalate any report involving electrical shock, burning smell, "
            "smoke, melting, exposed wiring, or repeated circuit trips.\n"
            "- Do not instruct customers to bypass electrical safety protections "
            "or open mains-powered devices.\n"
        ),
    },
]

PRODUCTS = [
    {
        "id": "bulb",
        "title": "Glow Bulb A19",
        "content": "### Glow Bulb A19\n- **Retail price:** Rp129,000",
    },
    {
        "id": "plug",
        "title": "Connect Plug Mini",
        "content": "### Connect Plug Mini\n- **Retail price:** Rp149,000",
    },
    {
        "id": "cam",
        "title": "Secure Cam Pan",
        "content": "### Secure Cam Pan\n- **Retail price:** Rp749,000",
    },
]

DIFFERENT_QUESTIONS = [
    {
        "id": "java",
        "title": "NOVAHAUS_FAQ",
        "content": "**Q: How long does delivery take within Java?**\n"
        "A: Typical delivery is 2-5 business days.",
    },
    {
        "id": "return-window",
        "title": "NOVAHAUS_FAQ",
        "content": "**Q: How long is the return window?**\n"
        "A: Eligible products can be returned within 7 calendar days.",
    },
]


class StubEmbedder:
    """A deterministic bag-of-words embedder with a collision-free vocabulary.

    Real embeddings are unnecessary here: these tests assert *decisions* (which
    path answered, whether the engine was consulted). Two texts that share
    vocabulary must score higher than two that do not, and two that share nothing
    must score exactly zero — otherwise an unrelated question can clear the gate
    by accident and the test would be measuring the stub, not the service. Each
    new token therefore gets its own dimension rather than a hashed one.
    """

    DIMENSIONS = 1024

    def __init__(self) -> None:
        # One vocabulary per service, so it stays small enough that distinct
        # tokens never share a dimension (a shared vocabulary grows across every
        # test in the process and starts wrapping).
        self._vocabulary: dict = {}

    def embed(self, text: str) -> list:
        vector = [0.0] * self.DIMENSIONS
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            index = self._vocabulary.setdefault(token, len(self._vocabulary))
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class StubEngine:
    """Stand-in for the LAYA engine that counts how often it is consulted."""

    def __init__(self) -> None:
        self.calls = 0

    def answer(self, context: str = "", query: str = "", context_texts=None) -> str:
        self.calls += 1
        return "ENGINE_ANSWER"


def make_service(entries, *, threshold=0.45, engine=None, **kwargs) -> CatalogService:
    """Build a CatalogService over ``entries`` using the stubs above."""
    service = CatalogService(
        # The stubs are structurally compatible with the real providers but do not
        # subclass them, so type checking is suppressed for these two arguments
        # only.
        embedder=StubEmbedder(),  # type: ignore[arg-type]
        llm=engine if engine is not None else StubEngine(),  # type: ignore[arg-type]
        threshold=threshold,
        top_k=3,
        **kwargs,
    )
    if entries:
        service.ingest(entries)
    return service


class DeterministicPathTest(unittest.TestCase):
    """The common case must be answered from the corpus, with no model call."""

    def test_answer_comes_from_the_corpus_without_calling_the_engine(self) -> None:
        engine = StubEngine()
        service = make_service(SHIPPING, engine=engine)
        response, diagnostics = service.query_with_diagnostics("how long is shipping?")
        self.assertTrue(response["matched"])
        self.assertIn("3 days", response["answer"])
        self.assertEqual(diagnostics["path"], "deterministic")
        self.assertEqual(engine.calls, 0)

    def test_unrelated_question_is_refused(self) -> None:
        engine = StubEngine()
        service = make_service(SHIPPING, engine=engine)
        response, _ = service.query_with_diagnostics("what did you eat yesterday?")
        self.assertFalse(response["matched"])
        self.assertEqual(engine.calls, 0)

    def test_bare_heading_is_never_an_answer(self) -> None:
        # The original answered "what is the company about?" with the document
        # title. Even at a perfect score, a heading-only node is not an answer.
        service = make_service(HEADING_ONLY[:1], threshold=0.10)
        response, _ = service.query_with_diagnostics(
            "NOVAHAUS Company Information & Corporate Knowledge Base"
        )
        self.assertFalse(response["matched"])
        self.assertNotIn("Corporate Knowledge Base", response["answer"])

    def test_heading_only_node_loses_to_the_prose_node(self) -> None:
        service = make_service(HEADING_ONLY, threshold=0.10)
        response, _ = service.query_with_diagnostics(
            "NOVAHAUS Company Information & Corporate Knowledge Base"
        )
        self.assertTrue(response["matched"])
        self.assertIn("Jakarta", response["answer"])


class HazardEscalationTest(unittest.TestCase):
    """A hazard report must never be gated on how well it scored."""

    def test_hazard_query_escalates_without_the_engine(self) -> None:
        engine = StubEngine()
        service = make_service(HAZARD, engine=engine)
        response, diagnostics = service.query_with_diagnostics(
            "my smart switch is sparking and there is a burning smell"
        )
        self.assertTrue(response["matched"])
        self.assertIn("burning smell", response["answer"])
        self.assertIn("escalat", response["answer"].lower())
        self.assertEqual(diagnostics["path"], "hazard-escalation")
        self.assertEqual(engine.calls, 0)

    def test_absence_of_a_hazard_uses_the_normal_path(self) -> None:
        service = make_service(HAZARD)
        _, diagnostics = service.query_with_diagnostics(
            "can I install a smart switch myself?"
        )
        self.assertNotEqual(diagnostics["path"], "hazard-escalation")


class CompoundTest(unittest.TestCase):
    """Both halves of a joined question must survive."""

    def test_compound_question_answers_both_halves(self) -> None:
        service = make_service(DIFFERENT_QUESTIONS, threshold=0.20)
        response, diagnostics = service.query_with_diagnostics(
            "how long is delivery and how long is the return window?"
        )
        self.assertEqual(diagnostics["path"], "compound")
        self.assertIn("2-5 business days", response["answer"])
        self.assertIn("7 calendar days", response["answer"])

    def test_comparison_answers_and_labels_both_sides(self) -> None:
        entries = [
            {
                "id": "a19",
                "title": "Glow Bulb A19",
                "content": "### Glow Bulb A19\n- Socket: E27",
            },
            {
                "id": "rgb",
                "title": "Glow Bulb RGB",
                "content": "### Glow Bulb RGB\n- Lighting: RGB colour control",
            },
        ]
        service = make_service(entries, threshold=0.20)
        response, diagnostics = service.query_with_diagnostics(
            "compare the Glow Bulb A19 and the Glow Bulb RGB"
        )
        self.assertEqual(diagnostics["path"], "compound")
        self.assertIn("E27", response["answer"])
        self.assertIn("RGB colour control", response["answer"])

    def test_atomic_questions_are_not_split(self) -> None:
        self.assertEqual(split_compound("how long is shipping?"), [])
        self.assertEqual(split_compound("what is the return policy?"), [])
        self.assertGreaterEqual(
            len(split_compound("how long is delivery and what is the return window")),
            2,
        )

    def test_comparison_detection(self) -> None:
        self.assertTrue(is_comparison("compare the A and the B"))
        self.assertTrue(is_comparison("A vs B"))
        self.assertFalse(is_comparison("how long is shipping?"))


VARIANT_NODES = [
    {
        "id": "java",
        "score": 0.60,
        "question": "How long does delivery take within Java?",
        "answer_text": "Typical delivery is 2-5 business days.",
    },
    {
        "id": "outside",
        "score": 0.58,
        "question": "How long does delivery take outside Java?",
        "answer_text": "Typical delivery is 3-8 business days.",
    },
    {
        "id": "jabodetabek",
        "score": 0.55,
        "question": "How long does delivery take in Jabodetabek?",
        "answer_text": "Typical delivery is 1-3 business days.",
    },
    {
        "id": "caveat",
        "score": 0.50,
        "question": "Can delivery take longer than the estimate?",
        "answer_text": "Yes. Courier delays can extend delivery time.",
    },
]


class VariantFamilyTest(unittest.TestCase):
    """Underspecified questions must return every parameter variant."""

    def test_family_groups_parameterised_questions(self) -> None:
        strategy = make_service(None).strategy
        family = strategy._variant_family(list(VARIANT_NODES))
        self.assertEqual({node["id"] for node in family}, {"java", "outside", "jabodetabek"})

    def test_tie_stitches_the_whole_family(self) -> None:
        strategy = make_service(None).strategy
        result = strategy._tie(list(VARIANT_NODES), 0.60, "")
        self.assertIsNotNone(result)
        assert result is not None  # narrow the Optional for the reader and the checker
        self.assertEqual(result.path, "tie-expansion")
        for window in ("1-3 business days", "2-5 business days", "3-8 business days"):
            self.assertIn(window, result.response["answer"])

    def test_strong_match_suppresses_expansion(self) -> None:
        # Asking about one region specifically must not return all of them.
        strategy = make_service(None).strategy
        self.assertIsNone(strategy._tie(list(VARIANT_NODES), 0.90, ""))


class BudgetTest(unittest.TestCase):
    """Every product at or below the stated ceiling is a valid answer."""

    def test_budget_lists_every_product_under_the_ceiling(self) -> None:
        service = make_service(PRODUCTS)
        response, diagnostics = service.query_with_diagnostics(
            "what products can I get for under Rp200,000?"
        )
        self.assertEqual(diagnostics["path"], "budget-filter")
        self.assertIn("Glow Bulb A19", response["answer"])
        self.assertIn("Connect Plug Mini", response["answer"])
        self.assertNotIn("Secure Cam Pan", response["answer"])

    def test_budget_below_the_floor_names_the_cheapest_product(self) -> None:
        service = make_service(PRODUCTS)
        response, diagnostics = service.query_with_diagnostics(
            "is there anything under Rp50,000?"
        )
        self.assertEqual(diagnostics["path"], "budget-filter")
        self.assertIn("129,000", response["answer"])
        self.assertIn("no novahaus product", response["answer"].lower())


class CacheTest(unittest.TestCase):
    """A repeated question must not be recomputed."""

    def test_repeat_query_is_served_from_the_cache(self) -> None:
        engine = StubEngine()
        service = make_service(SHIPPING, engine=engine, query_cache_size=8)
        first, first_diagnostics = service.query_with_diagnostics("how long is shipping?")
        second, second_diagnostics = service.query_with_diagnostics("how long is shipping?")
        self.assertEqual(first["answer"], second["answer"])
        self.assertEqual(first_diagnostics["path"], "deterministic")
        self.assertEqual(second_diagnostics["path"], "cache")
        self.assertEqual(engine.calls, 0)


class ConversationalTest(unittest.TestCase):
    """Greetings and thanks are answered before retrieval runs."""

    def test_greeting_variants_are_handled_without_the_engine(self) -> None:
        engine = StubEngine()
        service = make_service(SHIPPING, engine=engine)
        for query in ("hello", "hi there", "terima kasih"):
            response, diagnostics = service.query_with_diagnostics(query)
            self.assertTrue(response["matched"], query)
            self.assertEqual(response["score"], 1, query)
            self.assertEqual(diagnostics["path"], "conversational", query)
        self.assertEqual(engine.calls, 0)

    def test_greeting_prefix_is_stripped_before_retrieval(self) -> None:
        service = make_service(SHIPPING)
        response, diagnostics = service.query_with_diagnostics(
            "hey there, how long is shipping?"
        )
        self.assertTrue(response["matched"])
        self.assertIn("3 days", response["answer"])
        self.assertEqual(diagnostics["path"], "deterministic")


class LegacyModeTest(unittest.TestCase):
    """The original single path is preserved so the two can be compared."""

    def test_legacy_mode_consults_the_engine(self) -> None:
        engine = StubEngine()
        service = make_service(SHIPPING, engine=engine, answer_mode="legacy")
        response, diagnostics = service.query_with_diagnostics("how long is shipping?")
        self.assertEqual(engine.calls, 1)
        self.assertEqual(response["answer"], "ENGINE_ANSWER")
        self.assertEqual(diagnostics["path"], "layap")


if __name__ == "__main__":
    unittest.main()
