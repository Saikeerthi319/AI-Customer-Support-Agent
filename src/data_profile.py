from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def profile(path: str) -> dict:
    frame = pd.read_csv(path)
    by_brand = frame.groupby("brand").agg(
        conversations=("conversation_id", "nunique"),
        resolved=("resolved", "mean"),
        intents=("intent", "nunique"),
    ).reset_index()
    by_brand["score"] = (
        0.5 * (by_brand["conversations"] / by_brand["conversations"].max())
        + 0.3 * by_brand["resolved"]
        + 0.2 * (by_brand["intents"] / by_brand["intents"].max())
    )
    return {"brands": by_brand.sort_values("score", ascending=False).to_dict("records"), "rows": len(frame)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_conversations.csv")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = profile(args.data)
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
