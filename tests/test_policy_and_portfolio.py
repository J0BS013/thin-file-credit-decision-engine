from __future__ import annotations

from credit_engine.contracts import PointInTimeViolation
from credit_engine.features import build_decision_features, build_matured_training_dataset
from credit_engine.generator import generate_synthetic_data
from credit_engine.policy import apply_rule_based_policy
from credit_engine.portfolio import backtest_policy, vintage_summary


def _features_and_matured_training_data():
    data = generate_synthetic_data("smoke", seed=19)
    features = build_decision_features(
        data["applications"], data["bureau_snapshots"], data["cashflow_transactions"]
    )
    training = build_matured_training_dataset(
        features, data["loan_outcomes"], data["fraud_outcomes"], "2027-01-01"
    )
    return features, training


def test_rule_policy_is_deterministic_and_uses_allowed_loan_amounts() -> None:
    features, _ = _features_and_matured_training_data()
    decisions = apply_rule_based_policy(features)
    assert decisions["application_id"].is_unique
    assert set(decisions["decision"]).issubset({"approve", "decline"})
    assert set(decisions["approved_amount"]).issubset({0, 100, 200, 300})
    assert decisions.loc[decisions["decision"] == "decline", "approved_amount"].eq(0).all()


def test_vintage_summary_reconciles_to_approved_originations() -> None:
    features, training = _features_and_matured_training_data()
    decisions = apply_rule_based_policy(features)
    originations = backtest_policy(training, decisions)
    summary = vintage_summary(originations)
    assert summary["approved_applications"].sum() == len(originations)
    assert summary["defaults"].le(summary["approved_applications"]).all()
    assert summary["fraud_cases"].le(summary["approved_applications"]).all()
    assert summary["observed_default_rate"].between(0, 1).all()


def test_policy_rejects_training_labels_as_inputs() -> None:
    _, training = _features_and_matured_training_data()
    try:
        apply_rule_based_policy(training)
    except PointInTimeViolation:
        pass
    else:
        raise AssertionError("The policy must not accept outcome labels as decision-time inputs.")
