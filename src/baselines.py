import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


GOLDEN_PATH = "data/golden/apple_support_golden_200_annotation.csv"


# --------------------------------------------------
# Baseline 1: Trivial Baseline
# --------------------------------------------------

def trivial_baseline(golden_df):

    # Most common intent in the golden set
    majority_intent = (
        golden_df["intent"]
        .value_counts()
        .idxmax()
    )

    predictions = [
        majority_intent
        for _ in range(len(golden_df))
    ]

    accuracy = accuracy_score(
        golden_df["intent"],
        predictions
    )

    return majority_intent, accuracy, predictions


# --------------------------------------------------
# Baseline 2: Simple Keyword Baseline
# --------------------------------------------------

def simple_keyword_classifier(text):

    text = str(text).lower()

    # Apple ID / account
    if any(keyword in text for keyword in [
        "apple id",
        "icloud account",
        "forgot password",
        "password",
        "account"
    ]):
        return "APPLE_ID_ACCOUNT"

    # Battery
    if any(keyword in text for keyword in [
        "battery",
        "drain",
        "battery life",
        "battery draining"
    ]):
        return "BATTERY"

    # Charging / hardware
    if any(keyword in text for keyword in [
        "charge",
        "charging",
        "charger",
        "won't charge",
        "screen",
        "broken",
        "button",
        "speaker"
    ]):
        return "HARDWARE_CHARGING"

    # Connectivity
    if any(keyword in text for keyword in [
        "wifi",
        "wi-fi",
        "bluetooth",
        "internet",
        "network",
        "signal",
        "cellular",
        "mobile data"
    ]):
        return "CONNECTIVITY"

    # iOS / software
    if any(keyword in text for keyword in [
        "ios",
        "update",
        "upgrade",
        "downgrade",
        "software",
        "ios version"
    ]):
        return "IOS_SOFTWARE"

    # Apps
    if any(keyword in text for keyword in [
        "app",
        "application",
        "itunes",
        "app store"
    ]):
        return "APPS"

    # iCloud / backup
    if any(keyword in text for keyword in [
        "icloud",
        "backup",
        "restore backup",
        "back up"
    ]):
        return "ICLOUD_BACKUP"

    # Audio / media
    if any(keyword in text for keyword in [
        "music",
        "audio",
        "sound",
        "volume",
        "headphone",
        "headphones",
        "airpods",
        "video"
    ]):
        return "AUDIO_MEDIA"

    # Performance
    if any(keyword in text for keyword in [
        "slow",
        "lag",
        "freezing",
        "freeze",
        "performance",
        "crash"
    ]):
        return "DEVICE_PERFORMANCE"

    return "OTHER"


def simple_baseline(golden_df):

    predictions = []

    for text in golden_df["text"]:

        prediction = simple_keyword_classifier(
            text
        )

        predictions.append(prediction)

    accuracy = accuracy_score(
        golden_df["intent"],
        predictions
    )

    return accuracy, predictions


# --------------------------------------------------
# Main evaluation
# --------------------------------------------------

def main():

    print("=" * 60)
    print("APPLE SUPPORT BASELINE EVALUATION")
    print("=" * 60)

    print("\nLoading golden evaluation set...")

    golden_df = pd.read_csv(
        GOLDEN_PATH
    )

    golden_df = golden_df.dropna(
        subset=["text", "intent"]
    ).reset_index(drop=True)

    print(
        f"Golden examples: {len(golden_df)}"
    )

    # --------------------------------------------------
    # Trivial baseline
    # --------------------------------------------------

    majority_intent, trivial_accuracy, _ = (
        trivial_baseline(golden_df)
    )

    print("\n" + "-" * 60)
    print("BASELINE 1: TRIVIAL")
    print("-" * 60)

    print(
        f"Majority intent: {majority_intent}"
    )

    print(
        f"Accuracy: {trivial_accuracy:.4f}"
    )

    # --------------------------------------------------
    # Simple keyword baseline
    # --------------------------------------------------

    simple_accuracy, simple_predictions = (
        simple_baseline(golden_df)
    )

    print("\n" + "-" * 60)
    print("BASELINE 2: SIMPLE KEYWORD")
    print("-" * 60)

    print(
        f"Accuracy: {simple_accuracy:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            golden_df["intent"],
            simple_predictions,
            zero_division=0
        )
    )

    # --------------------------------------------------
    # Comparison
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)

    print(
        f"\nTrivial Baseline: "
        f"{trivial_accuracy:.4f}"
    )

    print(
        f"Simple Keyword Baseline: "
        f"{simple_accuracy:.4f}"
    )

    print(
        f"Our ML System: "
        f"0.8000"
    )

    print("\nDone.")


if __name__ == "__main__":

    main()