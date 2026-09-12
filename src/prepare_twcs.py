from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd
from ftfy import fix_text


REQUIRED_COLUMNS = {"tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id", "response_tweet_id"}


def _first_response(value: object) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
    else:
        parsed = value
    if isinstance(parsed, (list, tuple)) and parsed:
        return str(parsed[0])
    return str(parsed) if parsed else None


def _repair_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return fix_text(text)


def prepare(input_path: str | Path, output_path: str | Path, brand_account_id: str) -> pd.DataFrame:
    raw = pd.read_csv(input_path)
    missing = REQUIRED_COLUMNS.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing Customer Support on Twitter columns: {sorted(missing)}")
    raw["tweet_id"] = raw["tweet_id"].astype(str)
    raw["in_response_to_tweet_id"] = raw["in_response_to_tweet_id"].fillna("").astype(str)
    raw["response_tweet_id"] = raw["response_tweet_id"].map(_first_response)
    replies = raw.set_index("tweet_id")["text"].to_dict()
    reply_authors = raw.set_index("tweet_id")["author_id"].to_dict()
    customers = raw[(raw["inbound"] == True) & raw["response_tweet_id"].notna()].copy()
    customers = customers[customers["response_tweet_id"].map(reply_authors).eq(brand_account_id)]
    customers["agent_reply"] = customers["response_tweet_id"].map(replies)
    customers = customers.dropna(subset=["text", "agent_reply"])
    customers["brand"] = brand_account_id
    customers["conversation_id"] = customers["tweet_id"]
    customers["customer_message"] = customers["text"].map(_repair_text)
    customers["agent_reply"] = customers["agent_reply"].map(_repair_text)
    customers["intent"] = "unlabelled"
    customers["resolved"] = True
    result = customers[["conversation_id", "brand", "customer_message", "agent_reply", "intent", "resolved", "author_id"]].rename(columns={"author_id": "customer_id"})
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Kaggle twcs.csv into the agent input contract")
    parser.add_argument("--input", required=True, help="Path to Kaggle twcs.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--brand-account-id", required=True, help="Outbound brand account ID selected during profiling")
    args = parser.parse_args()
    result = prepare(args.input, args.output, args.brand_account_id)
    print(f"Wrote {len(result)} customer/response pairs to {args.output}")


if __name__ == "__main__":
    main()
