from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

from scipy.stats import spearmanr

from src.pipeline import SupportAgent, load_conversations
from src.settings import load_settings

DIMENSIONS = ("correctness", "groundedness", "helpfulness", "tone", "safety")


def _safe_divide(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 3) if denominator else 0.0


def _judge_validation(path: str = "evaluation/reply_scores.csv") -> dict:
    scores_path = Path(path)
    if not scores_path.exists():
        return {"status": "missing reply_scores.csv", "minimum_comparison_rows": 30}
    scores = pd.read_csv(scores_path)
    human_columns = [f"human_{dimension}" for dimension in DIMENSIONS]
    judge_columns = [f"judge_{dimension}" for dimension in DIMENSIONS]
    required = {"example_id", *human_columns, *judge_columns}
    if not required.issubset(scores.columns):
        return {"status": "reply_scores.csv missing required columns", "minimum_comparison_rows": 30}
    valid = scores.dropna(subset=human_columns + judge_columns)
    if len(valid) < 30:
        return {
            "status": f"incomplete double-scoring ({len(valid)} rows)",
            "n": int(len(valid)),
            "minimum_comparison_rows": 30,
        }
    correlation, p_value = spearmanr(valid[human_columns].mean(axis=1), valid[judge_columns].mean(axis=1))
    return {
        "status": "independent rater vs gpt-4o-mini",
        "n": int(len(valid)),
        "aggregate_spearman_rho": None if correlation != correlation else round(float(correlation), 3),
        "p_value": None if p_value != p_value else round(float(p_value), 4),
        "minimum_comparison_rows": 30,
    }


def evaluate(
    data_path: str | None = None,
    golden_path: str | None = None,
    output_path: str | None = None,
    config_path: str = "config.yaml",
) -> dict:
    settings = load_settings(config_path)
    train = load_conversations(data_path or settings.conversations_path)
    golden = pd.read_csv(golden_path or settings.golden_path)
    required = {"conversation_id", "customer_message", "intent", "expected_action"}
    missing = required.difference(golden.columns)
    if missing:
        raise ValueError(f"Golden set missing columns: {sorted(missing)}")
    golden = golden.dropna(subset=sorted(required)).reset_index(drop=True)
    overlap = set(train.conversation_id.astype(str)) & set(golden.conversation_id.astype(str))
    if overlap:
        raise ValueError(f"Train/golden conversation leakage detected ({len(overlap)} conversations)")
    if len(golden) < 150:
        raise ValueError(f"Golden set must contain at least 150 complete rows; found {len(golden)}")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    train_x = vectorizer.fit_transform(train.customer_message)
    golden_x = vectorizer.transform(golden.customer_message)
    majority = DummyClassifier(strategy="most_frequent").fit(train_x, train.intent)
    simple = LogisticRegression(max_iter=1000, random_state=7).fit(train_x, train.intent)
    agent = SupportAgent(
        train.reset_index(drop=True),
        min_confidence=settings.min_intent_confidence,
        min_similarity=settings.min_retrieval_similarity,
        top_k=settings.top_k,
    )
    golden_predictions = [agent.predict(message) for message in golden.customer_message]
    predicted_intents = [item.intent for item in golden_predictions]
    predicted_actions = [item.decision for item in golden_predictions]
    expected_actions = golden.expected_action.tolist()
    auto_indices = [index for index, action in enumerate(predicted_actions) if action == "AUTO_HANDLE"]
    expected_escalations = sum(action == "ESCALATE" for action in expected_actions)
    correct_auto = sum(expected_actions[index] == "AUTO_HANDLE" for index in auto_indices)
    unsafe_auto = sum(expected_actions[index] == "ESCALATE" for index in auto_indices)
    caught_escalations = sum(
        predicted == expected == "ESCALATE"
        for predicted, expected in zip(predicted_actions, expected_actions)
    )

    results = {
        "brand": str(train.brand.mode().iloc[0]) if "brand" in train.columns else "AmazonHelp",
        "annotation_status": (
            sorted(golden.label_status.dropna().unique().tolist())
            if "label_status" in golden.columns
            else ["unspecified"]
        ),
        "train_rows": int(len(train)),
        "golden_rows": int(len(golden)),
        "conversation_overlap": 0,
        "intent": {
            "majority_macro_f1": round(float(f1_score(golden.intent, majority.predict(golden_x), average="macro")), 3),
            "tfidf_logistic_regression_macro_f1": round(float(f1_score(golden.intent, simple.predict(golden_x), average="macro")), 3),
            "agent_macro_f1": round(float(f1_score(golden.intent, predicted_intents, average="macro")), 3),
            "agent_report": classification_report(
                golden.intent, predicted_intents, output_dict=True, zero_division=0
            ),
        },
        "trust": {
            "auto_handle_rate": _safe_divide(len(auto_indices), len(golden)),
            "correct_auto_handle_rate": _safe_divide(correct_auto, len(golden)),
            "auto_handle_precision": _safe_divide(correct_auto, len(auto_indices)),
            "unsafe_auto_handle_rate": _safe_divide(unsafe_auto, len(golden)),
            "escalation_recall": _safe_divide(caught_escalations, expected_escalations),
        },
        "judge_validation": _judge_validation(),
    }
    output = Path(output_path or settings.metrics_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    prediction_rows = golden.copy()
    prediction_rows["predicted_intent"] = predicted_intents
    prediction_rows["predicted_action"] = predicted_actions
    prediction_rows["intent_confidence"] = [item.intent_confidence for item in golden_predictions]
    prediction_rows["intent_source"] = [item.intent_source for item in golden_predictions]
    prediction_rows["reply"] = [item.reply for item in golden_predictions]
    prediction_rows["decision_reason"] = [item.reason for item in golden_predictions]
    prediction_rows["top_similarity"] = [
        item.evidence[0]["similarity"] if item.evidence else 0.0 for item in golden_predictions
    ]
    prediction_rows["retrieved_evidence"] = [
        item.evidence[0]["agent_reply"] if item.evidence else "" for item in golden_predictions
    ]
    prediction_rows.to_csv(output.with_name("predictions.csv"), index=False)
    failures = prediction_rows[
        (prediction_rows["intent"] != prediction_rows["predicted_intent"])
        | (prediction_rows["expected_action"] != prediction_rows["predicted_action"])
    ].copy()
    failures["failure_type"] = failures.apply(
        lambda row: (
            "intent and routing"
            if row["intent"] != row["predicted_intent"]
            and row["expected_action"] != row["predicted_action"]
            else "intent"
            if row["intent"] != row["predicted_intent"]
            else "routing"
        ),
        axis=1,
    )
    failures.sort_values(["failure_type", "intent_confidence"]).head(5).to_csv(
        output.with_name("failure_examples.csv"), index=False
    )
    return results


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
