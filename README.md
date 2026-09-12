# BrandSupport AI

A reproducible take-home implementation for a brand-specific customer-support agent. The system classifies an incoming customer message, retrieves similar historical resolutions, drafts a grounded reply, and chooses `AUTO_HANDLE` or `ESCALATE` with a reason.

This repository includes a small checked-in fixture so the pipeline runs without credentials. Replace it with a sampled Customer Support on Twitter export before reporting final results.

## Quickstart

```powershell
C:/Python314/python.exe -m pip install -r requirements.txt
C:/Python314/python.exe -m pytest -q
C:/Python314/python.exe -m evaluation.run
C:/Python314/python.exe -m src.pipeline --message "Where is my refund?"
```

The evaluation command writes `reports/generated/metrics.json`. The sample run is intentionally small and is a pipeline smoke test, not a claim about production quality.

## Data contract

The pipeline expects a CSV with these columns:

`conversation_id, brand, customer_message, agent_reply, intent, resolved`

For the real dataset, reconstruct threads before sampling. Split by conversation/thread ID so messages from one support interaction never appear in both train and test. Keep `data/golden_set.csv` frozen after annotation.

Start with:

```powershell
C:/Python314/python.exe -m src.data_profile --data data/your_sample.csv
```

For the Kaggle `twcs.csv` schema, convert inbound customer tweets with a linked response into the pipeline contract:

```powershell
C:/Python314/python.exe -m src.prepare_twcs --input data/raw/twcs.csv --output data/processed/conversations.csv --brand-account-id BRAND_ACCOUNT_ID
C:/Python314/python.exe -m src.data_profile --data data/processed/conversations.csv
```

The preparation script deliberately leaves `intent` as `unlabelled`; define and assign the brand-specific taxonomy during annotation rather than pretending it came from the raw dataset.

For the included dataset, the selected high-volume account is `AmazonHelp`. The real extraction and annotation commands are:

```powershell
C:/Python314/python.exe -m src.prepare_twcs --input "data/raw/twcs (2).csv" --output data/processed/amazonhelp_pairs.csv --brand-account-id AmazonHelp
C:/Python314/python.exe -m src.sample_golden --input data/processed/amazonhelp_pairs.csv --output data/golden_set_amazonhelp.csv --size 200 --seed 7
```

Open `data/golden_set_amazonhelp.csv` and fill `intent`, `expected_action`, and `expected_reason` manually. Do not run the final evaluation until those labels are complete.

The profiler ranks brands using volume, resolution rate, and intent diversity. The checked-in `Acme Mobile` rows are synthetic fixture data only and must not be presented as real brand evidence.

## System design

1. TF-IDF + logistic regression predicts a small, brand-specific intent taxonomy.
2. TF-IDF cosine retrieval finds the strongest historical customer cases.
3. The top historical agent resolution is used as offline grounded reply evidence.
4. The policy escalates low-confidence/weak-evidence requests, security language, complaints, and high-risk intents.
5. The output is structured JSON with intent, confidence, reply, evidence, decision, and reason.

The deterministic generator is the reproducible baseline. An LLM adapter can be added behind the same output contract, but the report must compare it with the offline system and never hide missing evidence behind fluent text.

## Evaluation deliverables

- `data/golden_set.csv`: hand-labelled examples. Expand this to 150-250 real examples using stratified sampling across intents, ambiguity, message length, and escalation risk.
- `evaluation/run.py`: majority, TF-IDF/logistic regression, and agent intent metrics plus auto-handling metrics.
- `evaluation/judge_agreement.py`: Spearman agreement between human and LLM-judge scores.
- `evaluation/human_scores.csv`: fill with 30-40 examples scored independently by a human and the judge.
- `evaluation/judge_rubric.md`: the rubric and judge prompt contract.
- `reports/report.md`: six-page report outline with required caveats and failure analysis.
- `decision_log.md`: non-obvious engineering decisions and rationale.

## Human and LLM reply evaluation

Score each reply from 1-5 for correctness, groundedness, helpfulness, tone, and safety. Record one aggregate score per example in `evaluation/human_scores.csv`, then run:

```powershell
C:/Python314/python.exe -m evaluation.judge_agreement
```

Do not invent agreement numbers. Report the actual sample size, correlation, and limitations.

## Scope and limitations

This MVP does not claim to resolve authentication, refunds, legal threats, or payments autonomously. It has no live Twitter integration, no customer identity verification, and no policy database beyond historical evidence. The sample fixture is too small for meaningful model conclusions. Final submission numbers must come from a frozen real golden set and conversation-level splits.
