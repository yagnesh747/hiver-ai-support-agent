import unittest
from src.response_generator import ResponseGenerator


class TestResponseGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = ResponseGenerator()

    def test_uses_best_historical_response(self):
        historical = [
            {"support_response": "@user DM us your iOS version.", "similarity": 0.5}
        ]
        reply = self.generator.generate("My battery drains fast", "BATTERY", historical)
        self.assertNotIn("@user", reply)
        self.assertIn("DM us your iOS version", reply)

    def test_falls_back_when_no_historical_response(self):
        reply = self.generator.generate("My battery drains fast", "BATTERY", [])
        self.assertIn("Apple Support", reply)

    def test_ignores_results_without_support_response(self):
        historical = [{"similarity": 0.9}]  # no "support_response" key
        reply = self.generator.generate("My battery drains fast", "BATTERY", historical)
        self.assertIn("Apple Support", reply)


if __name__ == "__main__":
    unittest.main()
