from __future__ import annotations

import argparse
import json

import pandas as pd
from scipy.stats import spearmanr


DIMENSIONS = ("correctness", "groundedness", "helpfulness", "tone", "safety")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LLM judge ratings with human ratings")
    parser.add_argument("--scores", default="evaluation/reply_scores.csv")
    args = parser.parse_args()
    scores = pd.read_csv(args.scores)
    required = {
        "example_id",
        *{f"human_{dimension}" for dimension in DIMENSIONS},
        *{f"judge_{dimension}" for dimension in DIMENSIONS},
    }
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    human_columns = [f"human_{dimension}" for dimension in DIMENSIONS]
    judge_columns = [f"judge_{dimension}" for dimension in DIMENSIONS]
    valid = scores.dropna(subset=human_columns + judge_columns).copy()
    if len(valid) < 30:
        raise ValueError(f"Need at least 30 independently double-scored rows; found {len(valid)}")
    valid["human_aggregate"] = valid[human_columns].mean(axis=1)
    valid["judge_aggregate"] = valid[judge_columns].mean(axis=1)
    correlation, p_value = spearmanr(valid.human_aggregate, valid.judge_aggregate)
    by_dimension = {}
    for dimension in DIMENSIONS:
        rho, _ = spearmanr(valid[f"human_{dimension}"], valid[f"judge_{dimension}"])
        by_dimension[dimension] = None if rho != rho else round(float(rho), 3)
    payload = {
        "n": len(valid),
        "aggregate_spearman_rho": None if correlation != correlation else round(float(correlation), 3),
        "p_value": None if p_value != p_value else round(float(p_value), 4),
        "dimension_spearman_rho": by_dimension,
        "notes": (
            "Undefined per-dimension correlations occur when one rater used a constant score "
            "(here independent safety scores were all 5 because every sampled output was an escalation)."
        ),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
