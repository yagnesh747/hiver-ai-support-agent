# AppleSupport AI Support Agent

An AI support agent for **AppleSupport** built on the Kaggle *Customer Support on Twitter* (TWCS) dataset. Given a customer message, the agent classifies its intent, drafts a reply grounded in AppleSupport's own historical responses, and decides whether to auto-handle the case or escalate it, with a reason.

Full results, methodology, and honest limitations are in **[REPORT.md](REPORT.md)**. Non-obvious design decisions are in **[DECISIONS.md](DECISIONS.md)**.

## Architecture

```
Customer message
      |
      v
Intent Classifier (TF-IDF + Logistic Regression)
      |
      v
Historical Retrieval (TF-IDF + cosine similarity over AppleSupport Q&A pairs)
      |
      +-----------------------------+
      v                             v
Response Generator            Escalation Decision
(copies/cleans best              (keyword rules +
 historical reply)              similarity + intent)
      |                             |
      +--------------+--------------+
                     v
        Draft response + decision + reason
```

No paid APIs are used anywhere in the pipeline (classification, retrieval, generation, or evaluation).

## Brand and dataset

- **Brand:** AppleSupport (~106,860 outbound tweets in the raw dataset — the highest-volume, most technical support account available).
- **Raw data:** `data/raw/twcs.csv` (Kaggle `thoughtvector/customer-support-on-twitter`, ~2.8M rows, ~190MB).
- **Processed:** `data/processed/apple_support_clean.csv` — AppleSupport-linked customer/response pairs.
- **Development set:** `data/processed/apple_support_dev_20000_labeled.csv` — 20,000 examples, **weak/rule-labelled** (keyword heuristics), used only for training the classifier. Never used for evaluation.
- **Golden evaluation set:** `data/golden/apple_support_golden_200_annotation.csv` — 200 examples used for all reported metrics.

## Intent taxonomy

`IOS_SOFTWARE`, `BATTERY`, `DEVICE_PERFORMANCE`, `APPS`, `APPLE_ID_ACCOUNT`, `ICLOUD_BACKUP`, `CONNECTIVITY`, `AUDIO_MEDIA`, `HARDWARE_CHARGING`, `OTHER`.

Actions: `ANSWER`, `TROUBLESHOOT`, `ESCALATE`.

## Setup

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

Dependencies: `pandas`, `scikit-learn` only. No GPU, no API key, no internet access required at run time.

## Data preparation (optional — processed files are already included)

The already-generated processed and golden files are committed to the repo, so you can skip straight to **Reproducing results** below without needing the raw dataset at all.

`data/raw/twcs.csv` (~190MB) exceeds GitHub's 100MB file-size limit and is therefore **excluded from the repository** (see `.gitignore`). To regenerate the processed data from scratch, download it yourself from Kaggle (`thoughtvector/customer-support-on-twitter`), place it at `data/raw/twcs.csv`, then run:

```bash
python -m src.data_processing      # raw twcs.csv -> data/processed/apple_support_clean.csv
python -m src.create_dev_labels    # weak-labels the 20k development sample
```

## Reproducing results (~2-3 minutes on a laptop, no GPU)

```bash
python -m src.baselines            # trivial + keyword baselines vs golden set
python -m src.intent_classifier    # trains classifier, evaluates on golden set
python -m src.evaluation           # full pipeline: intent, leakage-safe retrieval, action, response gen
python -m src.final_evaluation     # consolidated headline-metric summary
```

Try a single message interactively:

```bash
python main.py
```

### Headline result and important caveat

Intent classification achieved **80.0% accuracy** on the 200-example evaluation set, compared with **38.0%** for the majority-class baseline and **36.0%** for the keyword baseline.

However, intent accuracy is not the complete system result:

- Action accuracy: **63.0%**
- Leakage-safe average retrieval similarity: **0.5234**
- AI-assisted response-quality audit overall: **3.79/5**
- Responses scoring at least 4/5: **52.0%**
- Responses scoring below 3/5: **26.0%**

Therefore, the 80.0% intent number should not be interpreted as an 80% end-to-end support-agent success rate.

## Known limitations (see REPORT.md for full detail)

- Development labels are **weak/rule-based**, not human-labelled; only the 200 golden examples are used for evaluation.
- The response-quality audit is **AI-assisted**, not an independent human evaluation — no LLM-judge/human-agreement number is claimed, since none was genuinely measured, and no paid API was used.
- Escalation recall is low (0.07 on the golden set) — sensitive cases without an exact keyword match can be missed.
- Retrieval uses TF-IDF, not embeddings, so paraphrased historical matches can be missed.

## Repository structure

```
hiver-ai-support-agent/
├── data/{raw,processed,golden}/   # dataset at each processing stage
├── outputs/                       # evaluation CSVs produced by src/evaluation.py etc.
├── src/
│   ├── data_processing.py         # raw CSV -> AppleSupport-only clean CSV
│   ├── create_dev_labels.py       # weak/rule labels for the 20k dev set
│   ├── intent_classifier.py       # TF-IDF + Logistic Regression classifier
│   ├── retrieval.py                # TF-IDF cosine-similarity historical retrieval
│   ├── response_generator.py      # grounded copy-and-clean response drafting
│   ├── escalation.py              # rule + similarity based escalation decision
│   ├── evaluation.py               # full pipeline evaluation incl. leakage-safe retrieval
│   ├── final_evaluation.py        # consolidated metric summary
│   ├── llm_judge.py                # generates the response-quality audit template
│   └── analyze_human_eval.py      # summarizes the filled-in audit
├── main.py                         # interactive single-message demo
├── config.py                       # shared path/taxonomy constants (reference)
├── REPORT.md                       # full write-up, baselines, failure modes
└── DECISIONS.md                    # non-obvious design decisions
```
### Evaluation-labeling and LLM-judge transparency

The 200-example golden evaluation set contains AI-assisted draft labels and should not be interpreted as fully independently hand-labelled ground truth. Human review of the labels is still required.

For response quality, 50 examples were scored in an AI-assisted quality audit using the rubric columns for relevance, helpfulness, grounding, action appropriateness, and overall quality. A paid external LLM API was not available during development, so this audit is **not presented as an independent LLM-as-judge evaluation**.

Accordingly, no judge-human agreement statistic is claimed. Measuring independent human ratings and agreement with an LLM judge is a planned next step.
