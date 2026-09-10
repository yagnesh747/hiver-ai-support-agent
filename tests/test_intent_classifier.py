import unittest

from src.intent_classifier import IntentClassifier


class TestIntentClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.classifier = IntentClassifier()

    def test_battery(self):
        result = self.classifier.predict(
            "My iPhone battery is draining very quickly"
        )
        self.assertEqual(result, "BATTERY")

    def test_wifi(self):
        result = self.classifier.predict(
            "My WiFi is not working"
        )
        self.assertEqual(result, "CONNECTIVITY")

    def test_apple_id(self):
        result = self.classifier.predict(
            "I forgot my Apple ID password"
        )
        self.assertEqual(result, "APPLE_ID_ACCOUNT")

    def test_charging(self):
        result = self.classifier.predict(
            "My iPhone won't charge"
        )
        self.assertEqual(result, "HARDWARE_CHARGING")

    def test_ios_update(self):
        result = self.classifier.predict(
            "How do I update iOS?"
        )
        self.assertEqual(result, "IOS_SOFTWARE")


if __name__ == "__main__":
    unittest.main()