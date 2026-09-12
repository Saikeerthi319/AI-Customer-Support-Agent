from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


DIMENSIONS = ("correctness", "groundedness", "helpfulness", "tone", "safety")


def _judge(prompt: str, model: str, endpoint: str, api_key: str) -> dict:
    payload = json.dumps(
        {
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict customer-support reply evaluator. Return JSON only. "
                        "Score correctness, groundedness, helpfulness, tone, and safety from 1-5, "
                        "then include a one-sentence rationale and needs_human boolean."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
    ).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(1, 9):
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                body = json.loads(response.read().decode("utf-8"))
            result = json.loads(body["choices"][0]["message"]["content"])
            for dimension in DIMENSIONS:
                score = int(result[dimension])
                if not 1 <= score <= 5:
                    raise ValueError(f"Invalid {dimension} score: {score}")
            if "rationale" not in result:
                raise ValueError("Judge response missing rationale")
            return result
        except Exception as error:
            last_error = error
            time.sleep(min(3 ** attempt, 30))
    raise RuntimeError(f"LLM judge failed after retries: {last_error}") from last_error


def build_sample(predictions_path: str, output_path: str, size: int, seed: int) -> pd.DataFrame:
    predictions = pd.read_csv(predictions_path)
    sample = predictions.sample(min(size, len(predictions)), random_state=seed)
    rows = []
    for _, row in sample.iterrows():
        response = row["reply"]
        if pd.isna(response):
            response = f"[ESCALATED] {row['decision_reason']}"
        rows.append(
            {
                "example_id": row["id"],
                "customer_message": row["customer_message"],
                "reply": response,
                "evidence": row.get("retrieved_evidence", ""),
                "expected_action": row["expected_action"],
                **{f"human_{dimension}": pd.NA for dimension in DIMENSIONS},
                **{f"judge_{dimension}": pd.NA for dimension in DIMENSIONS},
                "judge_rationale": "",
                "judge_needs_human": pd.NA,
            }
        )
    result = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def score_file(path: str, model: str, endpoint: str, api_key: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["judge_rationale"] = frame["judge_rationale"].astype("object")
    frame["judge_needs_human"] = frame["judge_needs_human"].astype("object")
    for index, row in frame.iterrows():
        if pd.notna(row.get("judge_correctness")):
            continue
        prompt = (
            f"Customer message:\n{row.customer_message}\n\n"
            f"Historical evidence:\n{row.evidence}\n\n"
            f"Expected routing action:\n{row.expected_action}\n\n"
            f"System reply or routing outcome:\n{row.reply}"
        )
        result = _judge(prompt, model, endpoint, api_key)
        for dimension in DIMENSIONS:
            frame.at[index, f"judge_{dimension}"] = result[dimension]
        frame.at[index, "judge_rationale"] = result["rationale"]
        frame.at[index, "judge_needs_human"] = result["needs_human"]
        frame.to_csv(path, index=False)
        time.sleep(1.5)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and optionally LLM-score the reply-quality sample")
    parser.add_argument("--predictions", default="reports/generated/predictions.csv")
    parser.add_argument("--output", default="evaluation/reply_scores.csv")
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if args.rebuild or not output.exists():
        build_sample(args.predictions, args.output, args.size, args.seed)
    if args.prepare_only:
        print(f"Prepared {args.output}; add human scores independently.")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY, or pass --prepare-only to create the human scoring sheet.")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    endpoint = os.environ.get("OPENAI_CHAT_ENDPOINT", "https://api.openai.com/v1/chat/completions")
    scored = score_file(args.output, model, endpoint, api_key)
    print(f"LLM-scored {scored['judge_correctness'].notna().sum()} rows in {args.output}.")


if __name__ == "__main__":
    main()
