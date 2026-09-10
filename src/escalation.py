class EscalationDecision:

    def decide(self, customer_message, intent, historical_results):

        text = str(customer_message).lower()

        # --------------------------------------------------
        # 1. Sensitive issues -> ESCALATE
        # --------------------------------------------------

        escalation_keywords = [
            "refund",
            "charged twice",
            "double charged",
            "double billed",
            "billing dispute",
            "payment dispute",
            "unauthorized charge",
            "fraud",
            "stolen",
            "lost phone",
            "account compromised",
            "security issue",
            "legal",
            "lawsuit"
        ]

        for keyword in escalation_keywords:

            if keyword in text:

                return {
                    "decision": "ESCALATE",
                    "reason": (
                        "Message contains a sensitive issue "
                        f"requiring human support: {keyword}"
                    )
                }

        # --------------------------------------------------
        # 2. No historical evidence -> ESCALATE
        # --------------------------------------------------

        if not historical_results:

            return {
                "decision": "ESCALATE",
                "reason": (
                    "No relevant historical support "
                    "examples were found."
                )
            }

        # --------------------------------------------------
        # 3. Check retrieval confidence
        # --------------------------------------------------

        best_similarity = historical_results[0].get(
            "similarity",
            0
        )

        if best_similarity < 0.15:

            return {
                "decision": "ESCALATE",
                "reason": (
                    "The closest historical examples "
                    "are not sufficiently similar."
                )
            }

        # --------------------------------------------------
        # 4. Troubleshooting intents
        # --------------------------------------------------

        troubleshooting_intents = [
            "BATTERY",
            "CONNECTIVITY",
            "DEVICE_PERFORMANCE",
            "HARDWARE_CHARGING",
            "AUDIO_MEDIA",
            "IOS_SOFTWARE",
            "APPS",
            "ICLOUD_BACKUP"
        ]

        if intent in troubleshooting_intents:

            return {
                "decision": "TROUBLESHOOT",
                "reason": (
                    "The issue is a technical problem "
                    "where troubleshooting guidance can "
                    "be provided using historical support cases."
                )
            }

        # --------------------------------------------------
        # 5. General/account questions -> ANSWER
        # --------------------------------------------------

        return {
            "decision": "ANSWER",
            "reason": (
                "The issue matches historical AppleSupport "
                "cases and can be answered using historical "
                "support guidance."
            )
        }


if __name__ == "__main__":

    decision_engine = EscalationDecision()

    test_cases = [
        (
            "My iPhone battery is draining very quickly",
            "BATTERY"
        ),
        (
            "I was charged twice for my purchase",
            "OTHER"
        ),
        (
            "How do I check my Apple ID?",
            "APPLE_ID_ACCOUNT"
        )
    ]

    for message, intent in test_cases:

        result = decision_engine.decide(
            customer_message=message,
            intent=intent,
            historical_results=[
                {"similarity": 0.39}
            ]
        )

        print("\nCustomer:")
        print(message)

        print("\nDecision:")
        print(result["decision"])

        print("\nReason:")
        print(result["reason"])

        print("\n" + "-" * 50)