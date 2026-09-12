# AmazonHelp Support Agent

A reproducible customer-support agent built from Customer Support on Twitter. It classifies a message, retrieves historical AmazonHelp evidence, drafts a grounded reply, and returns `AUTO_HANDLE` or `ESCALATE` with a reason.

## Reproduce in under 15 minutes

```powershell
python -m pip install -r requirements.txt
python -m src.build_datasets
python -m pytest -q
python -m evaluation.run
python -m src.pipeline --message "Where is my refund?"
python -m evaluation.judge --prepare-only --rebuild
```

Evaluation writes metrics, predictions, and failure examples under `reports/generated/`. Headline results and limitations are in `reports/report.md`.

## Current result

- Majority baseline: 3.8% macro F1
- TF-IDF/logistic-regression baseline: 8.7% macro F1
- Rule-assisted agent: 33.7% macro F1
- Escalation recall: 98.2%, with 1.3% auto-handle coverage and 0% auto-handle precision
- Independent rater vs `gpt-4o-mini` on 40 replies: Spearman ρ = 0.314

This is a conservative prototype, not evidence of deployment readiness.

## Data and annotation

The checked-in source sample has 200 unique, real AmazonHelp customer/reply pairs. Draft labels use ten intents: `delivery_issue`, `order_status`, `refund_return`, `payment_issue`, `account_access`, `cancellation_change`, `product_question`, `technical_issue`, `complaint`, and `other`.

`src.build_datasets` merges both annotation batches and creates a deterministic 50-row train / 150-row golden split with no conversation leakage. Labels are marked `second_pass_reviewed` after a full second-pass audit of all 200 rows (25 corrections).

To rebuild the source sample from Kaggle:

```powershell
python -m src.prepare_twcs --input data/raw/twcs.csv --output data/processed/amazonhelp_pairs.csv --brand-account-id AmazonHelp
python -m src.sample_golden --input data/processed/amazonhelp_pairs.csv --output data/golden_set_amazonhelp.csv --size 200 --seed 7
```

## System

1. Multilingual high-signal rules plus TF-IDF/logistic regression predict intent.
2. TF-IDF cosine retrieval finds the strongest historical cases.
3. The top historical response is sanitized and used as the grounded draft.
4. Weak evidence, risky language, complaints, and ambiguous requests escalate.
5. Output is structured JSON containing intent, confidence, reply, evidence, decision, and reason.

Runtime paths and thresholds come from `config.yaml`. Rule matches report `intent_source=rule` and a null confidence because a deterministic rule does not produce a calibrated probability.

## Human and LLM reply evaluation

Prepare a fixed 40-example sheet, run the LLM judge, then independently fill the five `human_*` columns:

```powershell
python -m evaluation.judge --prepare-only --rebuild
python -m evaluation.judge
python -m evaluation.judge_agreement
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. The judge loads that file automatically. Do not commit `.env` or paste the key into chat.

The agreement command requires at least 30 complete independent/judge pairs. Current result: n=40, aggregate Spearman ρ=0.314.
