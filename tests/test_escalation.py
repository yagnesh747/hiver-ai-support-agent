import unittest
from src.escalation import EscalationDecision


class TestEscalationDecision(unittest.TestCase):

    def setUp(self):
        self.engine = EscalationDecision()

    def test_sensitive_keyword_escalates(self):
        result = self.engine.decide(
            "I was charged twice for my order, this is fraud",
            "OTHER",
            [{"similarity": 0.9}],
        )
        self.assertEqual(result["decision"], "ESCALATE")

    def test_no_historical_evidence_escalates(self):
        result = self.engine.decide(
            "My iPhone battery drains fast",
            "BATTERY",
            [],
        )
        self.assertEqual(result["decision"], "ESCALATE")

    def test_low_similarity_escalates(self):
        result = self.engine.decide(
            "My iPhone battery drains fast",
            "BATTERY",
            [{"similarity": 0.05}],
        )
        self.assertEqual(result["decision"], "ESCALATE")

    def test_technical_intent_troubleshoots(self):
        result = self.engine.decide(
            "My iPhone battery drains fast",
            "BATTERY",
            [{"similarity": 0.5}],
        )
        self.assertEqual(result["decision"], "TROUBLESHOOT")

    def test_non_technical_intent_answers(self):
        result = self.engine.decide(
            "How do I check my Apple ID?",
            "APPLE_ID_ACCOUNT",
            [{"similarity": 0.5}],
        )
        self.assertEqual(result["decision"], "ANSWER")


if __name__ == "__main__":
    unittest.main()
