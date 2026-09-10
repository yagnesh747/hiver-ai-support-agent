from src.intent_classifier import IntentClassifier
from src.retrieval import HistoricalRetriever
from src.response_generator import ResponseGenerator
from src.escalation import EscalationDecision


def main():

    print("=" * 60)
    print("       AppleSupport AI Support Agent")
    print("=" * 60)

    # --------------------------------------------------
    # Load all components
    # --------------------------------------------------

    print("\nLoading AI support system...\n")

    classifier = IntentClassifier()
    retriever = HistoricalRetriever()
    generator = ResponseGenerator()
    escalation_engine = EscalationDecision()

    print("\nSystem ready!")

    # --------------------------------------------------
    # Get customer message
    # --------------------------------------------------

    customer_message = input(
        "\nEnter customer message: "
    ).strip()

    if not customer_message:

        print("Please enter a customer message.")

        return

    # --------------------------------------------------
    # Step 1: Intent Classification
    # --------------------------------------------------

    intent = classifier.predict(customer_message)

    # --------------------------------------------------
    # Step 2: Historical Retrieval
    # --------------------------------------------------

    historical_results = retriever.search(
        customer_message,
        top_k=5
    )

    # --------------------------------------------------
    # Step 3: Generate Response
    # --------------------------------------------------

    response = generator.generate(
        customer_message,
        intent,
        historical_results
    )

    # --------------------------------------------------
    # Step 4: Escalation Decision
    # --------------------------------------------------

    escalation = escalation_engine.decide(
        customer_message,
        intent,
        historical_results
    )

    # --------------------------------------------------
    # Display Results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("                 RESULT")
    print("=" * 60)

    print("\nCustomer Message:")
    print(customer_message)

    print("\nDetected Intent:")
    print(intent)

    print("\nDecision:")
    print(escalation["decision"])

    print("\nReason:")
    print(escalation["reason"])

    print("\nDraft Response:")
    print(response)

    # --------------------------------------------------
    # Display Historical Evidence
    # --------------------------------------------------

    print("\n" + "-" * 60)
    print("Historical Evidence")
    print("-" * 60)

    useful_results = [
        result
        for result in historical_results
        if result.get("support_response")
    ]

    if not useful_results:

        print("No historical support response found.")

    else:

        for i, result in enumerate(
            useful_results[:3],
            start=1
        ):

            print(f"\nExample {i}")

            print(
                f"Similarity: "
                f"{result['similarity']:.4f}"
            )

            print(
                "Customer:",
                result["customer_text"]
            )

            print(
                "Historical Response:",
                result["support_response"]
            )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()