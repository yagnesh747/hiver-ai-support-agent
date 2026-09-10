import os
import re
import pandas as pd


# =========================
# Configuration
# =========================

RAW_DATA_PATH = "data/raw/twcs.csv"
PROCESSED_DIR = "data/processed"

APPLE_SUPPORT_OUTPUT = os.path.join(
    PROCESSED_DIR,
    "apple_support_clean.csv"
)


# =========================
# Text cleaning
# =========================

def clean_text(text):
    """
    Basic cleaning of tweet text while keeping the original meaning.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Decode common HTML entities
    text = (
        text.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
    )

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================
# Load AppleSupport data
# =========================

def load_apple_support_data(input_path=RAW_DATA_PATH, chunksize=200_000):
    if not os.path.isdir(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
    collected = []
    print("Reading dataset...")

    for chunk_number, chunk in enumerate(
        pd.read_csv(input_path, chunksize=chunksize)
    ):
        # AppleSupport's official support account
        brand_tweets = chunk[
            chunk["author_id"].astype(str).eq("AppleSupport")
        ]

        # Customer tweets directly mentioning AppleSupport
        customer_tweets = chunk[
            chunk["text"]
            .astype(str)
            .str.contains("@AppleSupport", case=False, na=False)
        ]

        selected = pd.concat(
            [brand_tweets, customer_tweets]
        ).drop_duplicates(subset=["tweet_id"])

        if not selected.empty:
            collected.append(selected)

        print(
            f"Processed chunk {chunk_number + 1} | "
            f"selected {len(selected):,} rows"
        )

    if not collected:
        raise ValueError("No AppleSupport data was found.")

    data = pd.concat(
        collected,
        ignore_index=True
    )

    return data


# =========================
# Prepare data
# =========================

def prepare_data(data):
    """
    Clean and organize AppleSupport conversations.
    """

    data = data.copy()

    # Clean tweet text
    data["clean_text"] = data["text"].apply(clean_text)

    # Remove empty messages
    data = data[
        data["clean_text"].str.len() > 0
    ].copy()

    # Convert timestamp
    data["created_at"] = pd.to_datetime(
        data["created_at"],
        errors="coerce",
        utc=True
    )

    # Identify speaker
    data["speaker"] = data["author_id"].apply(
        lambda x: (
            "support"
            if str(x) == "AppleSupport"
            else "customer"
        )
    )

    # Sort chronologically
    data = data.sort_values(
        ["created_at", "tweet_id"]
    ).reset_index(drop=True)

    # Keep useful columns
    columns = [
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "clean_text",
        "response_tweet_id",
        "in_response_to_tweet_id",
        "speaker"
    ]

    return data[columns]


# =========================
# Main pipeline
# =========================

def main():

    print("=" * 60)
    print("AppleSupport Data Processing")
    print("=" * 60)

    data = load_apple_support_data()

    print(
        f"\nAppleSupport-related rows found: {len(data):,}"
    )

    processed = prepare_data(data)

    print(
        f"Rows after cleaning: {len(processed):,}"
    )

    print("\nSpeaker distribution:")
    print(
        processed["speaker"].value_counts()
    )

    processed.to_csv(
        APPLE_SUPPORT_OUTPUT,
        index=False
    )

    print(
        f"\nSaved processed dataset to:\n"
        f"{APPLE_SUPPORT_OUTPUT}"
    )

    print("\nProcessing complete.")


if __name__ == "__main__":
    main()