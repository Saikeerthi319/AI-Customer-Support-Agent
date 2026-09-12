from evaluation.run import evaluate


def test_evaluation_writes_metrics():
    metrics = evaluate()
    assert metrics["dataset_rows"] == 24
    assert 0 <= metrics["trust"]["auto_handle_rate"] <= 1
    assert "tfidf_logistic_regression_macro_f1" in metrics["intent"]
