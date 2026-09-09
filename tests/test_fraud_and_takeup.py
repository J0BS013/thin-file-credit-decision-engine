from __future__ import annotations

from credit_engine.features import (
    build_decision_features,
    build_matured_take_up_dataset,
    build_matured_training_dataset,
)
from credit_engine.generator import generate_synthetic_data
from credit_engine.modeling import evaluate_binary_model, fit_logistic_scorecard, split_out_of_time
from credit_engine.takeup import TAKE_UP_FEATURES, fit_take_up_model


def _data():
    data = generate_synthetic_data("smoke", seed=59)
    features = build_decision_features(
        data["applications"], data["bureau_snapshots"], data["cashflow_transactions"]
    )
    return data, features


def test_fraud_model_is_trained_separately_from_default() -> None:
    data, features = _data()
    training = build_matured_training_dataset(
        features, data["loan_outcomes"], data["fraud_outcomes"], "2027-01-01"
    )
    split = split_out_of_time(training)
    fraud_model = fit_logistic_scorecard(split.train, target="fraud_confirmed")
    metrics = evaluate_binary_model(fraud_model, split.test, target="fraud_confirmed")
    assert 0 <= metrics["brier_score"] <= 1


def test_take_up_model_uses_matured_labels_and_friction_features() -> None:
    data, features = _data()
    training = build_matured_take_up_dataset(features, data["take_up_outcomes"], "2027-01-01")
    model = fit_take_up_model(training)
    probabilities = model.predict_proba(training[TAKE_UP_FEATURES])[:, 1]
    assert len(training) == len(features)
    assert all(0 <= probability <= 1 for probability in probabilities)
