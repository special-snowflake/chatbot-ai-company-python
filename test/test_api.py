"""End-to-end tests for the Python port.

Python equivalent of ``test/routes.test.js`` and ``test/catalog-service.test.js``.

Run with::

    python -m pytest test/ -v
    # or, without pytest:
    python -m unittest discover -s test -v
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from src.server import create_server  # noqa: E402

CATALOG = [
    {"id": "shipping", "question": "How long is shipping?", "answer": "Shipping takes 3 days."},
    {"id": "returns", "question": "What is the return policy?", "answer": "Returns are accepted within 30 days."},
    {"id": "hours", "title": "Support", "content": "Support is available Monday to Friday, 9am to 5pm."},
]


def fresh_client() -> TestClient:
    """A client backed by a brand-new app with an empty catalog."""
    return TestClient(asyncio.run(create_server(auto_ingest=False)))


class CatalogApiTest(unittest.TestCase):
    """Exercises the two API paths against a live in-process app."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = asyncio.run(create_server(auto_ingest=False))
        cls.client = TestClient(cls.app)

    def test_swagger_is_served_at_documentation(self) -> None:
        response = self.client.get("/documentation")
        self.assertEqual(response.status_code, 200)
        self.assertIn("swagger", response.text.lower())

    def test_query_before_ingest_returns_409(self) -> None:
        # Isolated app: the shared client is populated by the ingest tests.
        with fresh_client() as client:
            response = client.post("/catalog/query", json={"query": "how long is shipping?"})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"], "Catalog has not been ingested")

    def test_ingest_accepts_bare_array_and_returns_count(self) -> None:
        response = self.client.post("/catalog/ingest", json=CATALOG)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ingested": True, "count": 3})

    def test_ingest_rejects_entry_without_id(self) -> None:
        response = self.client.post("/catalog/ingest", json=[{"text": "no id here"}])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Each entry must have an id")

    def test_ingest_rejects_entry_without_usable_fields(self) -> None:
        response = self.client.post("/catalog/ingest", json=[{"id": "broken"}])
        self.assertEqual(response.status_code, 400)
        self.assertIn("question/answer, title/content, or text", response.json()["error"])

    def test_query_returns_matched_answer(self) -> None:
        self.client.post("/catalog/ingest", json=CATALOG)
        response = self.client.post("/catalog/query", json={"query": "How long is shipping?"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["matched"])
        self.assertIn("3 days", body["answer"])
        self.assertIn("score", body)

    def test_conversational_query_bypasses_catalog(self) -> None:
        response = self.client.post("/catalog/query", json={"query": "hello"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["matched"])
        self.assertEqual(body["score"], 1)

    def test_empty_query_returns_400_error_envelope(self) -> None:
        response = self.client.post("/catalog/query", json={"query": "   "})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "query must be a non-empty string")


class CatalogServiceTest(unittest.TestCase):
    """Unit-level checks on the pure scoring/parsing helpers."""

    def test_meaningful_words_strips_stopwords_and_aliases(self) -> None:
        from src.catalog_service import meaningful_words

        self.assertEqual(meaningful_words("How long does the duration take?"), {"long", "take"})

    def test_lexical_similarity_is_overlap_fraction(self) -> None:
        from src.catalog_service import lexical_similarity

        self.assertEqual(lexical_similarity({"shipping", "long"}, {"shipping", "long", "extra"}), 1.0)
        self.assertEqual(lexical_similarity({"a", "b"}, {"c", "d"}), 0.0)

    def test_parse_price_reads_rupiah(self) -> None:
        from src.catalog_service import parse_price

        self.assertEqual(parse_price("Retail price: Rp 1.500.000"), 1500000)
        self.assertIsNone(parse_price("no price here"))

    def test_parse_budget_query_handles_k_suffix(self) -> None:
        from src.catalog_service import parse_budget_query

        self.assertEqual(parse_budget_query("under 500k"), {"maxPrice": 500000})
        self.assertEqual(parse_budget_query("max Rp 2.000.000"), {"maxPrice": 2000000})
        self.assertIsNone(parse_budget_query("hello there"))

    def test_cosine_similarity_matches_js_behaviour(self) -> None:
        from src.similarity import cosine_similarity

        self.assertEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertEqual(cosine_similarity([], []), 0.0)
        self.assertEqual(cosine_similarity([1.0], [0.0]), 0.0)
        self.assertEqual(cosine_similarity([1.0, 2.0], [1.0]), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
