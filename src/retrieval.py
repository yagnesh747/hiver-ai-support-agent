import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = "data/processed/apple_support_clean.csv"


class HistoricalRetriever:

    def __init__(self, data_path=DATA_PATH):

        print("Loading AppleSupport historical data...")

        # Load complete processed dataset
        df = pd.read_csv(data_path)

        # Keep complete dataset for response lookup
        self.all_data = df.copy()

        # --------------------------------------------------
        # CUSTOMER MESSAGES
        # --------------------------------------------------

        customers = df[
            (df["speaker"] == "customer") &
            (df["response_tweet_id"].notna())
        ].copy()

        customers = customers[
            customers["clean_text"].notna()
        ].copy()

        customers = customers[
            customers["clean_text"].astype(str).str.strip() != ""
        ].copy()

        self.data = customers.reset_index(drop=True)

        print(
            f"Historical customer messages: "
            f"{len(self.data):,}"
        )

        # --------------------------------------------------
        # SUPPORT RESPONSE LOOKUP
        # --------------------------------------------------

        support_data = df[
            df["speaker"] == "support"
        ].copy()

        # Make IDs strings consistently
        support_data["tweet_id"] = (
            support_data["tweet_id"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
        )

        self.response_lookup = {}

        for _, row in support_data.iterrows():

            tweet_id = str(row["tweet_id"]).strip()

            response_text = row["clean_text"]

            if pd.notna(response_text):

                self.response_lookup[tweet_id] = str(
                    response_text
                )

        print(
            f"Historical support responses available: "
            f"{len(self.response_lookup):,}"
        )

        # --------------------------------------------------
        # TF-IDF INDEX
        # --------------------------------------------------

        print("Building TF-IDF index...")

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000
        )

        self.matrix = self.vectorizer.fit_transform(
            self.data["clean_text"].astype(str)
        )

        print("Retrieval index ready.")

    # ------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------

    def search(self, query, top_k=5):

        query = str(query)

        query_vector = self.vectorizer.transform(
            [query]
        )

        similarities = cosine_similarity(
            query_vector,
            self.matrix
        ).flatten()

        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []

        for index in top_indices:

            row = self.data.iloc[index]

            # ----------------------------------------------
            # Handle one or multiple response tweet IDs
            # ----------------------------------------------

            raw_response_ids = str(
                row["response_tweet_id"]
            )

            response_ids = raw_response_ids.split(",")

            response_texts = []

            for response_id in response_ids:

                response_id = response_id.strip()

                # Remove accidental .0 from numeric IDs
                response_id = reformat_id(response_id)

                if response_id in self.response_lookup:

                    response_texts.append(
                        self.response_lookup[response_id]
                    )

            # ----------------------------------------------
            # Combine historical support responses
            # ----------------------------------------------

            if response_texts:

                response_text = " ".join(
                    response_texts
                )

            else:

                response_text = None

            # ----------------------------------------------
            # Store result
            # ----------------------------------------------

            results.append({

                "customer_text":
                    row["clean_text"],

                "customer_tweet_id":
                    row["tweet_id"],

                "response_tweet_id":
                    row["response_tweet_id"],

                "support_response":
                    response_text,

                "similarity":
                    float(similarities[index])
            })

        return results


# ----------------------------------------------------------
# ID CLEANING
# ----------------------------------------------------------

def reformat_id(value):

    value = str(value).strip()

    try:

        number = float(value)

        if number.is_integer():

            return str(int(number))

    except (ValueError, TypeError):

        pass

    return value


# ----------------------------------------------------------
# TEST
# ----------------------------------------------------------

if __name__ == "__main__":

    retriever = HistoricalRetriever()

    test_query = (
        "My iPhone battery is draining "
        "very quickly"
    )

    results = retriever.search(
        test_query,
        top_k=5
    )

    print("\nTop historical matches:\n")

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"--- Match {i} ---"
        )

        print(
            f"Customer: "
            f"{result['customer_text']}"
        )

        print(
            f"Similarity: "
            f"{result['similarity']:.4f}"
        )

        print(
            f"Response Tweet ID: "
            f"{result['response_tweet_id']}"
        )

        print(
            f"AppleSupport Response: "
            f"{result['support_response']}"
        )

        print()