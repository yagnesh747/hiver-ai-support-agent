import re


class ResponseGenerator:

    def __init__(self):
        pass

    def generate(self, customer_message, intent, historical_results):

        # Keep only results that actually contain a historical response
        useful_results = [
            result
            for result in historical_results
            if result.get("support_response")
        ]

        if not useful_results:

            return (
                "Thanks for reaching out. "
                "We'd like to look into this with you. "
                "Please contact Apple Support so we can help further."
            )

        # Use the strongest historical response
        best_response = useful_results[0]["support_response"]

        # Remove Twitter mentions
        response = re.sub(
            r"@\w+",
            "",
            best_response
        )

        # Remove extra whitespace
        response = re.sub(
            r"\s+",
            " ",
            response
        ).strip()

        # Make sure the response is not empty
        if not response:

            return (
                "Thanks for reaching out. "
                "We'd be happy to look into this with you."
            )

        return response


if __name__ == "__main__":

    generator = ResponseGenerator()

    historical_results = [
        {
            "support_response":
                "Let's take a look at this together. "
                "DM us your iOS version to get started.",
            "similarity": 0.39
        }
    ]

    reply = generator.generate(
        customer_message=
            "My battery is draining very quickly",
        intent="BATTERY",
        historical_results=historical_results
    )

    print("Generated reply:")
    print(reply)