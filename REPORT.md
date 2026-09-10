# AppleSupport AI Support Agent — Evaluation Report

## 1. Executive Summary

This project builds an AI support agent for **AppleSupport** using the Customer Support on Twitter (TWCS) dataset.

The system performs four steps:

1. Classifies a customer message into one of 10 support intents.
2. Retrieves historically similar AppleSupport customer/support interactions.
3. Drafts a response using the retrieved historical support response.
4. Decides whether the issue should be answered, troubleshot, or escalated.

### Headline results

| Metric | Result |
|---|---:|
| Golden evaluation examples | 200 |
| Intent accuracy | **80.0%** |
| Trivial majority baseline | 38.0% |
| Simple keyword baseline | 36.0% |
| Leakage-safe average Top-1 retrieval similarity | **0.5234** |
| Leakage-safe median Top-1 similarity | **0.3924** |
| Action accuracy | **63.0%** |
| Response-quality audit, overall | **3.79 / 5** |
| Response-quality audit, grounding | **4.34 / 5** |
| Response-quality audit, helpfulness | **3.30 / 5** |

The intent classifier substantially outperforms both simple baselines. The main weakness is downstream decision and response quality: retrieving a lexically similar historical response does not always mean that the response is semantically appropriate for the current problem.

> **Evaluation note:** The 50-example response-quality audit currently in `outputs/human_response_evaluation.csv` was AI-assisted. It should not be described as independent human evaluation unless it is manually reviewed by a human evaluator. Therefore, this report does **not** claim human–judge agreement.

---

## 2. Problem Framing

The goal is not to build a general-purpose chatbot. The goal is to build a support agent that behaves conservatively using evidence from historical AppleSupport interactions.

A useful system should:

- identify the customer's actual issue,
- find similar historical cases,
- produce a response consistent with historical support behavior,
- avoid inventing unsupported troubleshooting steps,
- escalate sensitive or uncertain cases,
- provide an understandable reason for the escalation decision.

The prototype does **not** perform real account changes, refunds, payments, device repairs, or other actions in Apple's systems. It only produces a support decision and draft response.

---

## 3. Dataset and Sampling

The source dataset is the Kaggle **Customer Support on Twitter (TWCS)** dataset.

The selected brand is **AppleSupport**.

The processed AppleSupport data contains historical customer messages and AppleSupport responses. The retrieval corpus contains approximately:

- 28,332 customer messages with linked responses
- 32,945 support responses available for lookup

A development subset of 20,000 customer messages was used for model development and weak-label training. These labels were generated using deterministic keyword/rule heuristics and are **not** treated as the final evaluation labels.

### Intent taxonomy

The system uses 10 intents:

1. `IOS_SOFTWARE`
2. `BATTERY`
3. `DEVICE_PERFORMANCE`
4. `APPS`
5. `APPLE_ID_ACCOUNT`
6. `ICLOUD_BACKUP`
7. `CONNECTIVITY`
8. `AUDIO_MEDIA`
9. `HARDWARE_CHARGING`
10. `OTHER`

`OTHER` is a catch-all category for messages that do not fit the defined support categories.

### Golden evaluation set

The golden set contains 200 examples.

The sample was constructed reproducibly using message-length stratification across 10 bins, taking 20 examples per bin and then shuffling the resulting set.

The golden file contains intent and expected-action labels. Because the current labels were initially created with AI assistance, they should be independently reviewed before being described as fully hand-labelled in the final submission.

---

## 4. System Architecture

```text
Customer message
       |
       v
+---------------------+
| Intent Classifier   |
| TF-IDF + Logistic   |
| Regression          |
+----------+----------+
           |
           v
+---------------------+
| Historical Retrieval|
| TF-IDF + Cosine     |
| Similarity           |
+----------+----------+
           |
           +--------------------+
           |                    |
           v                    v
+------------------+   +----------------------+
| Response         |   | Escalation Decision  |
| Generator        |   | Rules + Similarity  |
+--------+---------+   +----------+-----------+
         |                        |
         +------------+-----------+
                      |
                      v
              Final support output
```

### Intent classifier

The classifier uses:

- TF-IDF features
- unigram and bigram features
- up to 50,000 features
- Logistic Regression
- `class_weight="balanced"`

This was chosen because it is fast, reproducible, inexpensive, and interpretable enough for a take-home assignment.

### Historical retrieval

Historical customer messages are indexed with TF-IDF. The system calculates cosine similarity between a new customer message and historical customer messages.

The top historical matches are then used as evidence for the response generator.

### Response generation

The current prototype uses a conservative copy-and-clean strategy:

- select the best historical result that has a support response,
- remove Twitter handles,
- normalize whitespace,
- return the historical support response.

This avoids hallucinating new support instructions, but it also creates one of the system's biggest weaknesses: a retrieved response can be lexically similar while still being inappropriate for the current customer.

### Escalation

The escalation component combines:

- sensitive-issue keyword detection,
- retrieval confidence,
- predicted intent.

Sensitive terms such as refund disputes, unauthorized charges, stolen/lost devices, account compromise, and legal issues are escalated.

Technical intents generally receive `TROUBLESHOOT`, while other supported issues receive `ANSWER`.

---

## 5. Intent Classification Results

### Overall

**Golden accuracy: 80.0%**

The classification report is:

| Intent | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| APPLE_ID_ACCOUNT | 0.80 | 1.00 | 0.89 | 4 |
| APPS | 0.57 | 0.40 | 0.47 | 10 |
| AUDIO_MEDIA | 0.81 | 0.68 | 0.74 | 19 |
| BATTERY | 0.88 | 0.93 | 0.90 | 15 |
| CONNECTIVITY | 1.00 | 0.69 | 0.82 | 13 |
| DEVICE_PERFORMANCE | 0.00 | 0.00 | 0.00 | 3 |
| HARDWARE_CHARGING | 0.92 | 0.69 | 0.79 | 16 |
| ICLOUD_BACKUP | 0.88 | 1.00 | 0.93 | 7 |
| IOS_SOFTWARE | 0.79 | 0.73 | 0.76 | 37 |
| OTHER | 0.82 | 0.93 | 0.87 | 76 |
| **Overall** | | | **0.80 accuracy** | **200** |

The strongest categories are Battery, iCloud/backup, Apple ID, and connectivity.

The weakest category is Device Performance. However, it has only three golden examples, so its zero F1 should not be interpreted as a stable estimate of real-world performance.

---

## 6. Baseline Comparison

Two deliberately simple baselines were implemented.

### Baseline 1 — Trivial majority classifier

Always predict the most frequent golden-set intent.

Result:

**38.0% accuracy**

### Baseline 2 — Simple keyword classifier

Predict an intent using manually defined keyword rules.

Result:

**36.0% accuracy**

### Proposed ML classifier

Result:

**80.0% accuracy**

| System | Accuracy |
|---|---:|
| Trivial majority | 38.0% |
| Simple keyword | 36.0% |
| **TF-IDF + Logistic Regression** | **80.0%** |

The proposed classifier improves accuracy by:

- **42 percentage points** over the trivial baseline
- **44 percentage points** over the keyword baseline

This shows that the model is learning useful lexical patterns beyond the small manually written keyword rules.

---

## 7. Retrieval Evaluation

A major evaluation issue was identified during development.

An initial retrieval evaluation produced an average similarity of approximately **0.9336**. Investigation showed that the retrieval corpus could contain the same examples used in the golden evaluation set.

That is data leakage.

The final retrieval evaluation therefore excludes exact golden customer messages from the historical retrieval corpus.

### Leakage-safe result

- Historical examples after golden exclusion: **28,207**
- Evaluated examples: **200**
- Average Top-1 similarity: **0.5234**
- Median Top-1 similarity: **0.3924**

The leakage-safe number is the metric that should be reported.

---

## 8. Action / Escalation Evaluation

The action classifier predicts:

- `ANSWER`
- `TROUBLESHOOT`
- `ESCALATE`

Overall action accuracy is:

**63.0%**

| Action | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ANSWER | 0.21 | 0.95 | 0.35 | 20 |
| ESCALATE | 0.50 | 0.07 | 0.12 | 15 |
| TROUBLESHOOT | 0.98 | 0.64 | 0.78 | 165 |
| **Overall accuracy** | | | | **63.0%** |

The model is very strong at identifying cases it should troubleshoot, but it is poor at recognizing all cases that should be escalated.

This is important operationally: an escalation system should prioritize recall for genuinely sensitive or risky cases rather than optimizing only for overall accuracy.

---

## 9. Response-Quality Audit

A 50-example response audit was generated from the evaluation output.

The current scores are:

| Dimension | Average |
|---|---:|
| Relevance | 3.78 / 5 |
| Helpfulness | 3.30 / 5 |
| Grounding | 4.34 / 5 |
| Action appropriateness | 3.74 / 5 |
| Overall quality | **3.79 / 5** |

Additional observations:

- 52% of audited responses scored at least 4.0/5 overall.
- 26% scored below 3.0/5.
- `TROUBLESHOOT` responses averaged 4.29/5.
- `ANSWER` responses averaged 3.29/5.

The high grounding score is expected because the prototype directly reuses historical support responses rather than generating unsupported instructions.

However, the lower helpfulness score shows the trade-off: being grounded is not enough if the retrieved response does not actually address the customer's current issue.

**Important:** these scores are currently AI-assisted and should be independently reviewed before presenting them as human ratings.

---

## 10. Top Five Failure Modes

### 1. Lexical retrieval can select the wrong support response

Example:

> Customer message: "Battery life on my iPhone is decreasing."

A retrieved historical response can be a generic conversational response such as a thank-you or unrelated troubleshooting message.

**Why it happens:** TF-IDF rewards shared words but does not understand whether two problems have the same underlying meaning.

**Hypothesis:** semantic embeddings plus intent-aware retrieval should reduce these mismatches.

---

### 2. Software and performance issues are easily confused

Examples include messages where the customer reports a device becoming slow or an app/system behaving poorly after an update.

The classifier can confuse:

- `IOS_SOFTWARE`
- `DEVICE_PERFORMANCE`

**Why it happens:** both categories share terms such as iPhone, update, slow, issue, and problem.

**Hypothesis:** hierarchical classification could first distinguish broad issue type and then classify the specific subtype.

---

### 3. `OTHER` is overloaded

`OTHER` represents 76 of the 200 golden examples.

This gives the classifier a strong incentive to predict the catch-all class.

Some messages that are genuinely actionable therefore receive a generic intent.

**Hypothesis:** splitting `OTHER` into a few additional high-frequency categories, or using a confidence threshold that routes uncertain messages to review, would improve precision.

---

### 4. Escalation recall is too low

The action evaluation shows only **7% recall for ESCALATE**.

Examples such as billing disputes or account/security concerns can be missed if the exact escalation keyword is absent.

**Why it happens:** the current escalation system is primarily rule-based.

**Hypothesis:** a supervised escalation classifier trained from manually labelled examples would be more robust than keyword matching.

---

### 5. Historical response quality depends heavily on the source pairing

Some TWCS records contain response IDs that are malformed, contain multiple IDs, or do not map cleanly to the expected response.

This can cause:

- missing support responses,
- generic responses,
- semantically mismatched responses.

**Hypothesis:** reconstructing complete conversation threads and pairing each customer message with the correct next support response would improve retrieval evidence substantially.

---

## 11. What Is Misleading About My Headline Number?

The most misleading number would be the initial **0.9336 retrieval similarity**.

It looks excellent, but it was affected by retrieval leakage: evaluation examples could be retrieved directly from the historical corpus.

After removing the golden examples from retrieval, the average Top-1 similarity drops to:

**0.5234**

This is a much more honest estimate of retrieval performance.

There are also two other caveats:

1. The 200-example golden set is relatively small.
2. The development labels are weak/rule-generated, so the model's training data is noisier than a fully human-labelled training set.

Therefore, **80% intent accuracy should be interpreted as a useful prototype result, not proof of production-level accuracy.**

---

## 12. Limitations

The current prototype has several limitations:

- Training labels are weak/rule-generated.
- The golden set needs independent human review before being called fully hand-labelled.
- The retrieval model is lexical rather than semantic.
- Response generation mostly copies historical responses.
- Escalation is rule-based and has low recall.
- The `OTHER` category is large.
- Some historical response IDs are malformed or difficult to pair.
- The response-quality audit is currently AI-assisted rather than independent human evaluation.
- A paid external LLM judge could not be used because the API account had insufficient quota, so no LLM-as-judge/human-agreement result is claimed.

---

## 13. Next Week

If given another week, I would prioritize:

### 1. Improve conversation reconstruction

Build a cleaner thread-level representation so each customer message is paired with the correct AppleSupport response.

### 2. Replace lexical retrieval with semantic retrieval

Use a local/open embedding model to retrieve cases based on meaning rather than only shared words.

### 3. Add intent-aware retrieval

Search within the predicted intent first, then rank historical examples by semantic similarity.

### 4. Train a dedicated escalation model

Create a manually reviewed escalation subset and optimize for high `ESCALATE` recall.

### 5. Improve response generation

Instead of copying one historical response, synthesize a short response from multiple retrieved examples while requiring every troubleshooting claim to be supported by retrieved evidence.

### 6. Add confidence calibration

Use classifier confidence and retrieval confidence jointly:

```text
High intent confidence + high retrieval confidence
        -> answer / troubleshoot

Low confidence or sensitive issue
        -> escalate
```

### 7. Complete independent evaluation

Have a human reviewer independently score the response sample and compare those scores with an LLM judge, reporting agreement rather than presenting AI-generated scores as human labels.

---

## 14. Reproducibility

From the project root:

```powershell
python src\data_processing.py
```

Create the development sample and weak labels as described in the repository.

Run the intent classifier:

```powershell
python src\intent_classifier.py
```

Run the complete evaluation:

```powershell
python src\evaluation.py
```

Run the baseline comparison:

```powershell
python src\baselines.py
```

Run the response-quality audit generation:

```powershell
python src\llm_judge.py
```

Run the audit analysis:

```powershell
python src\analyze_human_eval.py
```

Run an end-to-end example:

```powershell
python main.py
```

The headline evaluation is based on the 200-example golden set and the leakage-safe retrieval evaluation.

---

## 15. Final Takeaway

The prototype demonstrates that a lightweight, reproducible support agent can learn useful intent categories from historical AppleSupport conversations and substantially outperform simple baselines.

The strongest result is **80% intent accuracy versus 36–38% for the two baselines**.

The most important weakness is downstream: **retrieving a similar-looking historical message does not guarantee a correct support response**, and the current escalation rules miss too many cases.

The evaluation therefore supports the conclusion that the prototype is a promising retrieval-based support workflow, while also making clear that better semantic retrieval, cleaner conversation reconstruction, stronger escalation modelling, and independent human evaluation are needed before production use.
