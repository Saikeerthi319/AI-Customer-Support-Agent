import pandas as pd

from evaluation.judge import build_sample
from evaluation.run import evaluate
from src.settings import load_settings


def test_evaluation_writes_metrics():
    metrics = evaluate()
    assert metrics["brand"] == "AmazonHelp"
    assert metrics["golden_rows"] == 150
    assert metrics["conversation_overlap"] == 0
    assert 0 <= metrics["trust"]["auto_handle_rate"] <= 1
    assert "tfidf_logistic_regression_macro_f1" in metrics["intent"]


def test_evaluation_rejects_fixture_sized_golden_set():
    try:
        evaluate(
            data_path="data/sample_conversations.csv",
            golden_path="data/golden_set_amazonhelp.csv",
        )
    except ValueError as error:
        assert "complete rows" in str(error)
    else:
        raise AssertionError("Incomplete golden set should not be accepted")


def test_config_is_wired():
    settings = load_settings()
    assert settings.brand == "AmazonHelp"
    assert settings.min_retrieval_similarity == 0.45
    assert settings.conversations_path == "data/annotation_train.csv"


def test_judge_sample_uses_retrieved_evidence(tmp_path):
    predictions = pd.DataFrame(
        [
            {
                "id": 1,
                "customer_message": "Where is my order?",
                "reply": "I can help.",
                "decision_reason": "safe",
                "expected_action": "AUTO_HANDLE",
                "retrieved_evidence": "The retrieved historical reply.",
            }
        ]
    )
    predictions_path = tmp_path / "predictions.csv"
    scores_path = tmp_path / "scores.csv"
    predictions.to_csv(predictions_path, index=False)
    sample = build_sample(str(predictions_path), str(scores_path), size=1, seed=17)
    assert sample.loc[0, "evidence"] == "The retrieved historical reply."
