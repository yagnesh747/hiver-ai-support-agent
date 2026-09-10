import unittest

from src.retrieval import HistoricalRetriever


class TestHistoricalRetriever(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.retriever = HistoricalRetriever()

    def test_returns_results(self):
        results = self.retriever.search(
            "My iPhone battery is draining quickly",
            top_k=3
        )

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_results_have_similarity(self):
        results = self.retriever.search(
            "My WiFi is not working",
            top_k=3
        )

        self.assertIn("similarity", results[0])

    def test_similarity_is_numeric(self):
        results = self.retriever.search(
            "My iPhone will not charge",
            top_k=3
        )

        self.assertIsInstance(
            results[0]["similarity"],
            float
        )


if __name__ == "__main__":
    unittest.main()