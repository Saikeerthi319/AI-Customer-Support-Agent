from __future__ import annotations

import argparse
import pandas as pd
from scipy.stats import spearmanr


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LLM judge ratings with human ratings")
    parser.add_argument("--scores", default="evaluation/human_scores.csv")
    args = parser.parse_args()
    scores = pd.read_csv(args.scores)
    required = {"example_id", "human_score", "judge_score"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    correlation, p_value = spearmanr(scores.human_score, scores.judge_score)
    print({"n": len(scores), "spearman_rho": round(float(correlation), 3), "p_value": round(float(p_value), 4)})


if __name__ == "__main__":
    main()
