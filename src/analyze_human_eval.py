import pandas as pd

PATH = "outputs/human_response_evaluation.csv"

df = pd.read_csv(PATH)

score_columns = [
    "relevance_1_5",
    "helpfulness_1_5",
    "grounding_1_5",
    "action_appropriateness_1_5",
    "overall_1_5"
]

print("=" * 60)
print("HUMAN RESPONSE EVALUATION")
print("=" * 60)

print(f"\nExamples evaluated: {len(df)}")

print("\nAverage scores:")
for col in score_columns:
    print(f"{col}: {df[col].mean():.2f} / 5")

print("\nScore distribution for overall quality:")
print(df["overall_1_5"].value_counts().sort_index())

print("\nResponse quality by predicted action:")
print(
    df.groupby("predicted_action")["overall_1_5"]
    .mean()
    .round(2)
)

print("\nResponse quality by intent:")
print(
    df.groupby("predicted_intent")["overall_1_5"]
    .mean()
    .round(2)
)

print("\n" + "=" * 60)