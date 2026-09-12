from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from src.pipeline import SupportAgent, load_conversations


def evaluate(data_path: str = "data/sample_conversations.csv", golden_path: str = "data/golden_set.csv") -> dict:
    data = load_conversations(data_path)
    golden = pd.read_csv(golden_path)
    class_count = data.intent.nunique()
    test_size = max(math.ceil(len(data) * 0.25), class_count)
    train, test = train_test_split(data, test_size=test_size, random_state=7, stratify=data.intent)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    train_x = vectorizer.fit_transform(train.customer_message)
    test_x = vectorizer.transform(test.customer_message)
    majority = DummyClassifier(strategy="most_frequent").fit(train_x, train.intent)
    simple = LogisticRegression(max_iter=1000, random_state=7).fit(train_x, train.intent)
    agent = SupportAgent(train.reset_index(drop=True))
    golden_predictions = [agent.predict(message) for message in golden.customer_message]
    judge_status = "pending human annotations"
    judge_instructions = "Fill evaluation/human_scores.csv and run evaluation/judge_agreement.py."
    scores_path = Path("evaluation/human_scores.csv")
    if scores_path.exists():
        try:
            score_df = pd.read_csv(scores_path)
            required = {"example_id", "human_score", "judge_score"}
            if required.issubset(score_df.columns):
                valid = score_df[["human_score", "judge_score"]].dropna()
                if len(valid) >= 5:
                    judge_status = f"human scores available ({len(valid)} rows)"
                    judge_instructions = "Run evaluation/judge_agreement.py to compute the Spearman correlation."
        except Exception:
            pass

    results = {
        "dataset_rows": int(len(data)),
        "golden_rows": int(len(golden)),
        "intent": {
            "majority_macro_f1": round(float(f1_score(test.intent, majority.predict(test_x), average="macro")), 3),
            "tfidf_logistic_regression_macro_f1": round(float(f1_score(test.intent, simple.predict(test_x), average="macro")), 3),
            "agent_macro_f1_on_golden": round(float(f1_score(golden.intent, [item.intent for item in golden_predictions], average="macro")), 3),
            "agent_report": classification_report(golden.intent, [item.intent for item in golden_predictions], output_dict=True, zero_division=0),
        },
        "trust": {
            "auto_handle_rate": round(sum(item.decision == "AUTO_HANDLE" for item in golden_predictions) / len(golden_predictions), 3),
            "escalation_rate": round(sum(item.decision == "ESCALATE" for item in golden_predictions) / len(golden_predictions), 3),
            "expected_escalation_recall": round(sum(item.decision == expected for item, expected in zip(golden_predictions, golden.expected_action) if expected == "ESCALATE") / max(1, sum(golden.expected_action == "ESCALATE")), 3),
        },
        "judge_validation": {"status": judge_status, "instructions": judge_instructions},
    }
    output = Path("reports/generated/metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
