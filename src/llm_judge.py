import os
import pandas as pd


INPUT_PATH = "outputs/evaluation_results.csv"

OUTPUT_PATH = (
    "outputs/human_response_evaluation.csv"
)

SAMPLE_SIZE = 50


def main():

    print("=" * 60)
    print("HUMAN RESPONSE QUALITY EVALUATION")
    print("=" * 60)

    # --------------------------------------------------
    # Load evaluation results
    # --------------------------------------------------

    if not os.path.exists(INPUT_PATH):

        print(
            f"\nERROR: {INPUT_PATH} not found."
        )

        print(
            "\nRun this first:"
        )

        print(
            "python -m src.evaluation"
        )

        return

    df = pd.read_csv(INPUT_PATH)

    print(
        f"\nAvailable examples: {len(df)}"
    )

    # --------------------------------------------------
    # Reproducible sample
    # --------------------------------------------------

    sample_size = min(
        SAMPLE_SIZE,
        len(df)
    )

    sample = df.sample(
        n=sample_size,
        random_state=42
    ).reset_index(drop=True)

    print(
        f"Examples selected: {len(sample)}"
    )

    # --------------------------------------------------
    # Create human evaluation columns
    # --------------------------------------------------

    output = pd.DataFrame({

        "tweet_id":
            sample["tweet_id"],

        "customer_message":
            sample["text"],

        "expected_intent":
            sample["intent"],

        "predicted_intent":
            sample["predicted_intent"],

        "expected_action":
            sample["expected_action"],

        "predicted_action":
            sample["predicted_action"],

        "generated_response":
            sample["generated_response"],

        "retrieval_similarity":
            sample["retrieval_similarity"],

        # Human scoring fields
        "relevance_1_5": "",

        "helpfulness_1_5": "",

        "grounding_1_5": "",

        "action_appropriateness_1_5": "",

        "overall_1_5": "",

        "reviewer_notes": ""
    })

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("HUMAN EVALUATION FILE CREATED")
    print("=" * 60)

    print(
        f"\nSaved to:"
    )

    print(OUTPUT_PATH)

    print(
        "\nNumber of examples:"
        f" {len(output)}"
    )

    print("\nScoring rubric:")

    print(
        "\nRelevance (1-5):"
        "\n1 = Does not address the issue"
        "\n3 = Partially addresses the issue"
        "\n5 = Directly addresses the issue"
    )

    print(
        "\nHelpfulness (1-5):"
        "\n1 = Not useful"
        "\n3 = Some useful guidance"
        "\n5 = Clear and actionable"
    )

    print(
        "\nGrounding (1-5):"
        "\n1 = Unsupported by historical evidence"
        "\n3 = Partially grounded"
        "\n5 = Clearly grounded in historical support"
    )

    print(
        "\nAction Appropriateness (1-5):"
        "\n1 = Clearly inappropriate"
        "\n3 = Reasonable"
        "\n5 = Appropriate decision"
    )

    print(
        "\nOverall (1-5):"
        "\n1 = Very poor"
        "\n3 = Acceptable"
        "\n5 = Excellent"
    )


if __name__ == "__main__":

    main()