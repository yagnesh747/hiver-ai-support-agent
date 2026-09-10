import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


TRAIN_PATH = "data/processed/apple_support_dev_20000_labeled.csv"
GOLDEN_PATH = "data/golden/apple_support_golden_200_annotation.csv"


class IntentClassifier:

    def __init__(self):

        print("Loading intent classification data...")

        # Load training data
        train_df = pd.read_csv(TRAIN_PATH)

        train_df = train_df.dropna(
            subset=["text", "intent"]
        )

        self.X_train = train_df["text"].astype(str)
        self.y_train = train_df["intent"].astype(str)

        # Create model
        self.model = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50000
                )
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced"
                )
            )
        ])

        # IMPORTANT:
        # Train the model before prediction
        print("Training intent classifier...")

        self.model.fit(
            self.X_train,
            self.y_train
        )

        print("Intent classifier ready.")

    def predict(self, text):

        if not text:
            return "OTHER"

        prediction = self.model.predict(
            [str(text)]
        )

        return prediction[0]

    def evaluate(self):

        print("\nLoading golden evaluation set...")

        golden_df = pd.read_csv(GOLDEN_PATH)

        golden_df = golden_df.dropna(
            subset=["text", "intent"]
        )

        X_test = golden_df["text"].astype(str)
        y_test = golden_df["intent"].astype(str)

        predictions = self.model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print(
            f"\nGolden Accuracy: {accuracy:.4f}"
        )

        print("\nClassification Report:")

        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0
            )
        )


if __name__ == "__main__":

    classifier = IntentClassifier()

    # Evaluate classifier
    classifier.evaluate()

    # Test predictions
    test_messages = [
        "My iPhone battery is draining very quickly",
        "My WiFi is not working",
        "I forgot my Apple ID password",
        "My iPhone won't charge",
        "How do I update iOS?"
    ]

    print("\nSample Predictions:\n")

    for message in test_messages:

        prediction = classifier.predict(
            message
        )

        print(
            f"Message: {message}"
        )

        print(
            f"Intent: {prediction}"
        )

        print()