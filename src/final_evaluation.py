import os
import pandas as pd

GOLDEN_PATH = "data/golden/apple_support_golden_200_annotation.csv"
EVAL_PATH = "outputs/evaluation_results.csv"
LEAKAGE_SAFE_RETRIEVAL_PATH = "outputs/leakage_safe_retrieval_results.csv"
HUMAN_PATH = "outputs/human_response_evaluation.csv"

print("=" * 70)
print("FINAL SYSTEM EVALUATION")
print("=" * 70)

# --------------------------------------------------
# 1. Golden set
# --------------------------------------------------

golden = pd.read_csv(GOLDEN_PATH)

print("\n1. GOLDEN SET")
print("-" * 70)
print(f"Golden examples: {len(golden)}")

# --------------------------------------------------
# 2. Automated metrics
# --------------------------------------------------

evaluation = pd.read_csv(EVAL_PATH)

print("\n2. AUTOMATED METRICS")
print("-" * 70)

print("Evaluation columns:")
print(list(evaluation.columns))

# Intent accuracy
if "intent" in evaluation.columns and "predicted_intent" in evaluation.columns:
    intent_accuracy = (
        evaluation["intent"].astype(str)
        == evaluation["predicted_intent"].astype(str)
    ).mean()

    print(f"Intent accuracy: {intent_accuracy:.4f}")

# Action accuracy
if "expected_action" in evaluation.columns and "predicted_action" in evaluation.columns:
    action_accuracy = (
        evaluation["expected_action"].astype(str)
        == evaluation["predicted_action"].astype(str)
    ).mean()

    print(f"Action accuracy: {action_accuracy:.4f}")

# --------------------------------------------------
# 3. Leakage-safe retrieval
# --------------------------------------------------

print("\n3. LEAKAGE-SAFE RETRIEVAL")
print("-" * 70)

if os.path.exists(LEAKAGE_SAFE_RETRIEVAL_PATH):

    leakage_safe = pd.read_csv(
        LEAKAGE_SAFE_RETRIEVAL_PATH
    )

    if "similarity" in leakage_safe.columns:

        average_similarity = leakage_safe["similarity"].mean()
        median_similarity = leakage_safe["similarity"].median()

        print(
            f"Average retrieval similarity (leakage-safe): "
            f"{average_similarity:.4f}"
        )

        print(
            f"Median retrieval similarity (leakage-safe): "
            f"{median_similarity:.4f}"
        )

    else:
        print(
            "ERROR: 'similarity' column not found in "
            "leakage-safe retrieval file."
        )

else:
    print(
        "WARNING: leakage-safe retrieval file not found."
    )
    print(
        "Run: python -m src.evaluation"
    )

# --------------------------------------------------
# 4. AI-assisted response quality audit
# --------------------------------------------------

human = pd.read_csv(HUMAN_PATH)

print("\n4. AI-ASSISTED RESPONSE QUALITY AUDIT")
print("-" * 70)
print(f"Examples evaluated: {len(human)}")

metrics = [
    "relevance_1_5",
    "helpfulness_1_5",
    "grounding_1_5",
    "action_appropriateness_1_5",
    "overall_1_5"
]

for metric in metrics:

    if metric in human.columns:

        print(
            f"{metric}: "
            f"{human[metric].mean():.2f}/5"
        )

# --------------------------------------------------
# 5. Response quality by action
# --------------------------------------------------

print("\n5. RESPONSE QUALITY BY ACTION")
print("-" * 70)

if (
    "predicted_action" in human.columns
    and "overall_1_5" in human.columns
):

    print(
        human.groupby("predicted_action")["overall_1_5"]
        .agg(["count", "mean"])
        .round(2)
    )

# --------------------------------------------------
# 6. Response quality by intent
# --------------------------------------------------

print("\n6. RESPONSE QUALITY BY INTENT")
print("-" * 70)

if (
    "predicted_intent" in human.columns
    and "overall_1_5" in human.columns
):

    print(
        human.groupby("predicted_intent")["overall_1_5"]
        .agg(["count", "mean"])
        .round(2)
    )

# --------------------------------------------------
# 7. Error analysis
# --------------------------------------------------

print("\n7. ERROR COUNTS")
print("-" * 70)

# Use the evaluation file because it contains the actual
# golden expected intent/action columns.

if (
    "intent" in evaluation.columns
    and "predicted_intent" in evaluation.columns
):

    intent_errors = (
        evaluation["intent"].astype(str)
        != evaluation["predicted_intent"].astype(str)
    ).sum()

    print(f"Intent errors: {intent_errors}")

if (
    "expected_action" in evaluation.columns
    and "predicted_action" in evaluation.columns
):

    action_errors = (
        evaluation["expected_action"].astype(str)
        != evaluation["predicted_action"].astype(str)
    ).sum()

    print(f"Action errors: {action_errors}")

# --------------------------------------------------
# 8. Response quality summary
# --------------------------------------------------

print("\n8. RESPONSE QUALITY SUMMARY")
print("-" * 70)

if "overall_1_5" in human.columns:

    scores = pd.to_numeric(
        human["overall_1_5"],
        errors="coerce"
    ).dropna()

    if len(scores) > 0:

        excellent = (
            scores >= 4.0
        ).mean()

        weak = (
            scores < 3.0
        ).mean()

        print(
            f"Responses >= 4.0/5: "
            f"{excellent:.1%}"
        )

        print(
            f"Responses < 3.0/5: "
            f"{weak:.1%}"
        )

# --------------------------------------------------
# Done
# --------------------------------------------------

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)