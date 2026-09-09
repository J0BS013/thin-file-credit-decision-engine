from __future__ import annotations

from credit_engine.features import build_decision_features, build_matured_training_dataset
from credit_engine.generator import generate_synthetic_data
from credit_engine.modeling import evaluate_binary_model, fit_logistic_scorecard, split_out_of_time


def _training_data():
    data = generate_synthetic_data("smoke", seed=31)
    features = build_decision_features(
        data["applications"], data["bureau_snapshots"], data["cashflow_transactions"]
    )
    return build_matured_training_dataset(
        features, data["loan_outcomes"], data["fraud_outcomes"], "2027-01-01"
    )


def test_out_of_time_split_has_no_temporal_overlap() -> None:
    split = split_out_of_time(_training_data())
    assert split.train["application_timestamp"].max() < split.test["application_timestamp"].min()
    assert len(split.train) + len(split.test) == 1_000


def test_logistic_scorecard_produces_bounded_held_out_metrics() -> None:
    split = split_out_of_time(_training_data())
    model = fit_logistic_scorecard(split.train)
    metrics = evaluate_binary_model(model, split.test)
    assert set(metrics) == {"roc_auc", "pr_auc", "brier_score"}
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["pr_auc"] <= 1
    assert 0 <= metrics["brier_score"] <= 1
