# Data

`sample_conversations.csv` is a synthetic, tiny fixture for local execution. Do not use it as evidence in the submission.

Place the sampled Kaggle export under `data/raw/` and create a processed CSV matching the contract described in the root README. Keep raw data out of git if its license or size requires that. Document the exact sample seed, brand-selection query, preprocessing, and conversation-level split in the final report.

## Checked-in AmazonHelp sample

`golden_set_amazonhelp.csv` contains 200 real customer/brand reply pairs sampled without replacement from the prepared AmazonHelp pairs (`seed=7`, one row per conversation).

The two `annotations_*.csv` files are AI-drafted labels. `python -m src.build_datasets` validates and merges them, then makes a deterministic 50/150 train/golden split (`seed=23`) with no conversation overlap. The generated rows deliberately carry `label_status=ai_draft_requires_human_review`.

Before submission, the candidate must independently review every label against `labeling_guide.md`, correct disagreements, and replace `label_status` with `human_verified`. This step cannot truthfully be automated or claimed on the candidate's behalf.
