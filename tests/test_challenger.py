from __future__ import annotations

from credit_engine.challenger import fit_calibrated_xgboost
from credit_engine.features import build_decision_features, build_matured_training_dataset
from credit_engine.generator import generate_synthetic_data
from credit_engine.modeling import evaluate_probabilities, split_out_of_time


def _training_data():
    data = generate_synthetic_data("smoke", seed=47)
    features = build_decision_features(
        data["applications"], data["bureau_snapshots"], data["cashflow_transactions"]
    )
    return build_matured_training_dataset(
        features, data["loan_outcomes"], data["fraud_outcomes"], "2027-01-01"
    )


def test_challenger_calibration_precedes_final_out_of_time_holdout() -> None:
    training = _training_data()
    final_holdout = split_out_of_time(training)
    challenger = fit_calibrated_xgboost(final_holdout.train)
    assert challenger.calibration_cutoff < final_holdout.test["application_timestamp"].min()


def test_calibrated_challenger_returns_valid_held_out_probabilities_and_metrics() -> None:
    training = _training_data()
    final_holdout = split_out_of_time(training)
    challenger = fit_calibrated_xgboost(final_holdout.train)
    probabilities = challenger.predict_proba(final_holdout.test)
    metrics = evaluate_probabilities(final_holdout.test["defaulted"], probabilities)
    assert len(probabilities) == len(final_holdout.test)
    assert all(0 <= probability <= 1 for probability in probabilities)
    assert 0 <= metrics["brier_score"] <= 1
