# Decision Log

Non-obvious decisions made while building the AppleSupport AI Support Agent, and why.

1. **Brand: AppleSupport.**
   AppleSupport is one of the highest-volume support accounts in the TWCS dataset (~106,860 outbound tweets), giving enough data for both weak-label training and a meaningful retrieval corpus, while still fitting on a laptop without a database.

2. **10 intents, not more.**
   A larger taxonomy (e.g. 20-30 intents) would fragment the training data further and make weak labeling and manual golden-set annotation much slower, without clearly improving usefulness of the agent's routing decision. 10 intents keep each class large enough to learn from.

3. **`OTHER` is a deliberate catch-all, not a defect.**
   Twitter support threads contain many non-technical messages (thanks, follow-up questions, off-topic mentions). Forcing these into a technical intent would corrupt the other classes, so `OTHER` absorbs them. Its size (9,250/20,000 in dev) is reported honestly rather than hidden.

4. **20,000-example development set, not the full ~106,860.**
   TF-IDF + Logistic Regression saturates quickly on this problem; a 20k sample keeps iteration fast (fit in seconds) while still giving thousands of examples per common intent.

5. **Weak/rule-based labels are used only for training, never for evaluation.**
   Using the same heuristics to label both training and evaluation data would make accuracy numbers trivially high and meaningless (the classifier would just be learning to imitate the rules on data drawn from those same rules). The golden set is kept separate specifically to avoid this.

6. **Golden set size: 200 examples.**
   150-250 was the assignment's target range. 200 gives enough support per intent (even the smallest classes reach 3-4 examples) while remaining a size a single person can plausibly review carefully.

7. **TF-IDF + Logistic Regression over a neural/transformer classifier.**
   The dataset is short, informal Twitter text where lexical cues (e.g. "battery", "won't charge", "forgot password") are already highly predictive. A linear model on TF-IDF is fast, fully explainable in an interview, requires no GPU, and reaches 80% golden accuracy — more complex models were judged not worth the added opacity for this problem size.

8. **`class_weight="balanced"` on the classifier.**
   Without it, the model would be biased toward the dominant `OTHER`/`IOS_SOFTWARE` classes and would rarely predict small classes like `DEVICE_PERFORMANCE`. Balancing trades a little overall accuracy for much better recall on minority intents.

9. **TF-IDF cosine similarity for retrieval, not embeddings.**
   Embedding models would require either a paid API or downloading a large local model. TF-IDF retrieval is free, deterministic, and reproducible without a GPU, matching the "no paid API" constraint, at the cost of missing some paraphrases.

10. **Golden examples are excluded from the retrieval corpus before evaluating retrieval quality.**
    Without this exclusion, golden customer messages can match themselves verbatim inside the historical corpus, producing a falsely high similarity score (0.9336 avg / 1.00 median). Excluding them yields the true, leakage-safe number (0.5234 avg / 0.3924 median), which is the one reported as the headline retrieval metric.

11. **Response generation copies and cleans the best historical reply instead of generating free text.**
    A generative model (local or paid) could hallucinate incorrect troubleshooting steps for a support product. Copying a real historical AppleSupport response and stripping handles/whitespace keeps every response grounded in something the brand actually said, at the cost of occasionally reusing a lexically-similar-but-wrong reply.

12. **Sensitive-keyword escalation rules were added on top of similarity.**
    Billing disputes, fraud, and security issues are high-stakes and should never be silently auto-answered even if a similar historical case exists. A short explicit keyword list acts as a safety net independent of the similarity score.

13. **A similarity threshold (0.15) triggers escalation for low-confidence retrieval.**
    When no historical case is close to the customer's message, guessing a canned response is riskier than escalating. The threshold was chosen empirically as a conservative cutoff below which retrieved responses were observed to be unreliable.

14. **No paid LLM API is used anywhere in the pipeline (classification, retrieval, generation, or judging).**
    The project must run for free and reproducibly on any machine. This ruled out both an API-based generator and an API-based LLM-as-judge; both constraints are documented explicitly rather than worked around with a smaller trial credit.

15. **The response-quality audit is labeled "AI-assisted," not "human-evaluated."**
    The 50-example audit in `outputs/human_response_evaluation.csv` was scored by an AI assistant reviewing each example, not by an independent human rater. Calling it "human evaluation" or claiming "judge-human agreement" would misrepresent the assignment's evaluation-integrity requirement, so the report explicitly flags this as a limitation instead.

16. **Escalation's low recall is reported honestly rather than tuned away with more rules.**
    ESCALATE recall is only 0.07 on the golden set. It would be easy to inflate this by adding many more keywords tuned to the golden set itself, but that would overfit to 200 examples and stop generalizing. The weakness is instead documented as a top failure mode with a concrete next step (broader phrase/sentiment-based escalation signals).
