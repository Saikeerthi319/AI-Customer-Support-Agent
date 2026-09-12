from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def sample(input_path: str, output_path: str, size: int, seed: int) -> pd.DataFrame:
    frame = pd.read_csv(input_path)
    frame = frame.drop_duplicates(subset=["conversation_id"])
    sample_size = min(size, len(frame))
    result = frame.sample(sample_size, random_state=seed).copy()
    result.insert(0, "id", range(1, len(result) + 1))
    result["intent"] = ""
    result["expected_action"] = ""
    result["expected_reason"] = ""
    result = result[["id", "conversation_id", "customer_message", "agent_reply", "intent", "expected_action", "expected_reason"]]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a frozen manual-labeling sample")
    parser.add_argument("--input", default="data/processed/amazonhelp_pairs.csv")
    parser.add_argument("--output", default="data/golden_set_amazonhelp.csv")
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = sample(args.input, args.output, args.size, args.seed)
    print(f"Wrote {len(result)} examples to {args.output}; fill intent and expected action manually.")


if __name__ == "__main__":
    main()