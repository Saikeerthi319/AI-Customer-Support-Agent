from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


INTENTS = {
    "delivery_issue",
    "order_status",
    "refund_return",
    "payment_issue",
    "account_access",
    "cancellation_change",
    "product_question",
    "technical_issue",
    "complaint",
    "other",
}
ACTIONS = {"AUTO_HANDLE", "ESCALATE"}


def build(
    sample_path: str,
    annotation_paths: list[str],
    train_path: str,
    golden_path: str,
    golden_size: int = 150,
    seed: int = 23,
    label_status: str = "second_pass_reviewed",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = pd.read_csv(sample_path)
    annotations = pd.concat([pd.read_csv(path) for path in annotation_paths], ignore_index=True)
    if annotations["id"].duplicated().any():
        raise ValueError("Annotation files contain duplicate IDs")
    merged = sample.drop(columns=["intent", "expected_action", "expected_reason"]).merge(
        annotations, on="id", how="left", validate="one_to_one"
    )
    required = ["intent", "expected_action", "expected_reason"]
    if merged[required].isna().any().any():
        raise ValueError("Every sample row must have a complete annotation")
    unknown_intents = set(merged.intent) - INTENTS
    unknown_actions = set(merged.expected_action) - ACTIONS
    if unknown_intents or unknown_actions:
        raise ValueError(
            f"Unknown labels: intents={sorted(unknown_intents)}, actions={sorted(unknown_actions)}"
        )
    if not 150 <= golden_size <= 250 or golden_size >= len(merged):
        raise ValueError("Golden size must be 150-250 and leave at least one training row")

    golden_parts = []
    for _, group in merged.groupby("intent"):
        count = max(1, round(len(group) * golden_size / len(merged)))
        count = min(count, len(group) - 1) if len(group) > 1 else 1
        golden_parts.append(group.sample(count, random_state=seed))
    golden = pd.concat(golden_parts)
    remaining = merged.drop(golden.index)
    if len(golden) > golden_size:
        move = golden.sample(len(golden) - golden_size, random_state=seed)
        golden = golden.drop(move.index)
        remaining = pd.concat([remaining, move])
    elif len(golden) < golden_size:
        move = remaining.sample(golden_size - len(golden), random_state=seed)
        remaining = remaining.drop(move.index)
        golden = pd.concat([golden, move])

    train = remaining.copy()
    for frame in (train, golden):
        frame["brand"] = "AmazonHelp"
        frame["resolved"] = True
        frame["label_status"] = label_status
        for column in ("customer_message", "agent_reply", "expected_reason"):
            frame[column] = frame[column].map(
                lambda value: "\n".join(line.rstrip() for line in str(value).splitlines())
            )
    columns = [
        "id",
        "conversation_id",
        "brand",
        "customer_message",
        "agent_reply",
        "intent",
        "expected_action",
        "expected_reason",
        "resolved",
        "label_status",
    ]
    train = train[columns].sort_values("id").reset_index(drop=True)
    golden = golden[columns].sort_values("id").reset_index(drop=True)
    if set(train.conversation_id.astype(str)) & set(golden.conversation_id.astype(str)):
        raise AssertionError("Conversation leakage")

    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False)
    golden.to_csv(golden_path, index=False)
    return train, golden


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge reviewed labels and freeze train/golden splits")
    parser.add_argument("--sample", default="data/golden_set_amazonhelp.csv")
    parser.add_argument(
        "--annotations",
        nargs="+",
        default=["data/annotations_001_100.csv", "data/annotations_101_200.csv"],
    )
    parser.add_argument("--train", default="data/annotation_train.csv")
    parser.add_argument("--golden", default="data/golden_set.csv")
    parser.add_argument("--golden-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--label-status",
        choices=["ai_draft_requires_human_review", "second_pass_reviewed", "human_verified"],
        default="second_pass_reviewed",
    )
    args = parser.parse_args()
    train, golden = build(
        args.sample,
        args.annotations,
        args.train,
        args.golden,
        args.golden_size,
        args.seed,
        args.label_status,
    )
    print(f"Wrote {len(train)} train and {len(golden)} golden AmazonHelp rows.")


if __name__ == "__main__":
    main()
