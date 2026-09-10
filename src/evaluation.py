import os
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report

from src.intent_classifier import IntentClassifier
from src.retrieval import HistoricalRetriever
from src.response_generator import ResponseGenerator
from src.escalation import EscalationDecision


GOLDEN_PATH = "data/golden/apple_support_golden_200_annotation.csv"
OUTPUT_PATH = "outputs/evaluation_results.csv"


class EvaluationHarness:

    def __init__(self):

        print("Loading golden evaluation set...")

        self.golden = pd.read_csv(GOLDEN_PATH)

        self.golden = self.golden.dropna(
            subset=["text", "intent", "expected_action"]
        ).reset_index(drop=True)

        print(
            f"Golden examples: {len(self.golden)}"
        )

        print("\nLoading system components...")

        self.classifier = IntentClassifier()
        self.retriever = HistoricalRetriever()
        self.generator = ResponseGenerator()
        self.escalation = EscalationDecision()

        print("\nEvaluation system ready.")

    # --------------------------------------------------
    # Intent Evaluation
    # --------------------------------------------------

    def evaluate_intent(self):

        print("\n" + "=" * 60)
        print("INTENT CLASSIFICATION EVALUATION")
        print("=" * 60)

        texts = self.golden["text"].astype(str)

        true_labels = self.golden["intent"].astype(str)

        predictions = []

        for text in texts:

            prediction = self.classifier.predict(text)

            predictions.append(prediction)

        accuracy = accuracy_score(
            true_labels,
            predictions
        )

        print(
            f"\nIntent Accuracy: {accuracy:.4f}"
        )

        print("\nClassification Report:")

        print(
            classification_report(
                true_labels,
                predictions,
                zero_division=0
            )
        )

        return predictions, accuracy

    # --------------------------------------------------
    # Leakage-Safe Retrieval Evaluation
    # --------------------------------------------------

    def evaluate_retrieval(self):

        print("\n" + "=" * 60)
        print("LEAKAGE-SAFE RETRIEVAL EVALUATION")
        print("=" * 60)

        print(
            "\nEvaluating historical evidence..."
        )

        similarities = []

        rows = []

        # We identify Golden messages by their text.
        golden_texts = set(
            self.golden["text"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        # Historical retrieval corpus
        historical_data = self.retriever.data.copy()

        # Remove Golden examples from the retrieval corpus.
        clean_text_lower = (
            historical_data["clean_text"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        evaluation_corpus = historical_data[
            ~clean_text_lower.isin(golden_texts)
        ].copy()

        print(
            f"Historical examples available after "
            f"Golden-set exclusion: "
            f"{len(evaluation_corpus):,}"
        )

        # Build a separate TF-IDF index
        # only on the leakage-safe corpus.
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000
        )

        matrix = vectorizer.fit_transform(
            evaluation_corpus["clean_text"]
            .astype(str)
        )

        for _, row in self.golden.iterrows():

            query = str(row["text"])

            query_vector = vectorizer.transform(
                [query]
            )

            scores = cosine_similarity(
                query_vector,
                matrix
            ).flatten()

            if len(scores) == 0:

                continue

            best_index = scores.argmax()

            best_score = float(
                scores[best_index]
            )

            best_customer = evaluation_corpus.iloc[
                best_index
            ]["clean_text"]

            similarities.append(
                best_score
            )

            rows.append({
                "tweet_id": row["tweet_id"],
                "query": query,
                "best_match": best_customer,
                "similarity": best_score
            })

        if similarities:

            average_similarity = sum(
                similarities
            ) / len(similarities)

            median_similarity = float(
                pd.Series(similarities).median()
            )

        else:

            average_similarity = 0
            median_similarity = 0

        print(
            f"\nEvaluated examples: "
            f"{len(similarities)}"
        )

        print(
            f"Average Top-1 Similarity: "
            f"{average_similarity:.4f}"
        )

        print(
            f"Median Top-1 Similarity: "
            f"{median_similarity:.4f}"
        )

        # Save retrieval examples
        os.makedirs(
            "outputs",
            exist_ok=True
        )

        retrieval_path = (
            "outputs/"
            "leakage_safe_retrieval_results.csv"
        )

        pd.DataFrame(rows).to_csv(
            retrieval_path,
            index=False
        )

        print(
            f"\nRetrieval results saved to:"
        )

        print(retrieval_path)

        return {
            "average_similarity":
                average_similarity,

            "median_similarity":
                median_similarity
        }

    # --------------------------------------------------
    # Escalation Evaluation
    # --------------------------------------------------

    def evaluate_escalation(self):

        print("\n" + "=" * 60)
        print("ESCALATION / ACTION EVALUATION")
        print("=" * 60)

        true_actions = (
            self.golden["expected_action"]
            .astype(str)
        )

        predictions = []

        for _, row in self.golden.iterrows():

            text = str(row["text"])

            intent = self.classifier.predict(
                text
            )

            historical_results = (
                self.retriever.search(
                    text,
                    top_k=5
                )
            )

            result = self.escalation.decide(
                text,
                intent,
                historical_results
            )

            predictions.append(
                result["decision"]
            )

        accuracy = accuracy_score(
            true_actions,
            predictions
        )

        print(
            f"\nAction Accuracy: "
            f"{accuracy:.4f}"
        )

        print("\nClassification Report:")

        print(
            classification_report(
                true_actions,
                predictions,
                zero_division=0
            )
        )

        return predictions, accuracy

    # --------------------------------------------------
    # Response Generation Evaluation
    # --------------------------------------------------

    def evaluate_responses(
        self,
        intent_predictions
    ):

        print("\n" + "=" * 60)
        print("RESPONSE GENERATION EVALUATION")
        print("=" * 60)

        results = []

        for index, row in self.golden.iterrows():

            text = str(row["text"])

            intent = intent_predictions[index]

            historical_results = (
                self.retriever.search(
                    text,
                    top_k=5
                )
            )

            response = self.generator.generate(
                text,
                intent,
                historical_results
            )

            best_similarity = 0

            if historical_results:

                best_similarity = (
                    historical_results[0]
                    .get("similarity", 0)
                )

            results.append({
                "tweet_id": row["tweet_id"],
                "customer_message": text,
                "intent": intent,
                "expected_action":
                    row["expected_action"],
                "generated_response": response,
                "best_retrieval_similarity":
                    best_similarity
            })

        response_path = (
            "outputs/"
            "response_evaluation.csv"
        )

        os.makedirs(
            "outputs",
            exist_ok=True
        )

        pd.DataFrame(results).to_csv(
            response_path,
            index=False
        )

        print(
            f"\nGenerated responses: "
            f"{len(results)}"
        )

        print(
            f"Response results saved to:"
        )

        print(response_path)

        return results

    # --------------------------------------------------
    # Run Everything
    # --------------------------------------------------

    def run(self):

        print("\n")
        print("#" * 60)
        print("#       APPLESUPPORT EVALUATION HARNESS")
        print("#" * 60)

        # Intent
        intent_predictions, intent_accuracy = (
            self.evaluate_intent()
        )

        # Leakage-safe retrieval
        retrieval_results = (
            self.evaluate_retrieval()
        )

        # Actions
        action_predictions, action_accuracy = (
            self.evaluate_escalation()
        )

        # Responses
        response_results = (
            self.evaluate_responses(
                intent_predictions
            )
        )

        # --------------------------------------------------
        # Save complete evaluation table
        # --------------------------------------------------

        final_results = self.golden.copy()

        final_results[
            "predicted_intent"
        ] = intent_predictions

        final_results[
            "predicted_action"
        ] = action_predictions

        final_results[
            "generated_response"
        ] = [
            result["generated_response"]
            for result in response_results
        ]

        final_results[
            "retrieval_similarity"
        ] = [
            result[
                "best_retrieval_similarity"
            ]
            for result in response_results
        ]

        os.makedirs(
            "outputs",
            exist_ok=True
        )

        final_results.to_csv(
            OUTPUT_PATH,
            index=False
        )

        # --------------------------------------------------
        # Final Summary
        # --------------------------------------------------

        print("\n" + "=" * 60)
        print("FINAL EVALUATION SUMMARY")
        print("=" * 60)

        print(
            f"\nIntent Accuracy: "
            f"{intent_accuracy:.4f}"
        )

        print(
            f"Leakage-Safe Average "
            f"Retrieval Similarity: "
            f"{retrieval_results['average_similarity']:.4f}"
        )

        print(
            f"Leakage-Safe Median "
            f"Retrieval Similarity: "
            f"{retrieval_results['median_similarity']:.4f}"
        )

        print(
            f"Action Accuracy: "
            f"{action_accuracy:.4f}"
        )

        print(
            f"\nComplete results saved to:"
        )

        print(OUTPUT_PATH)

        print("\nEvaluation complete.")


if __name__ == "__main__":

    evaluator = EvaluationHarness()

    evaluator.run()