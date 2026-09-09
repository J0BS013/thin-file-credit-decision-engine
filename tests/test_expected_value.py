from __future__ import annotations

import pandas as pd

from credit_engine.expected_value import calculate_expected_value, select_expected_value_policy
from credit_engine.features import build_decision_features, build_matured_take_up_dataset, build_matured_training_dataset
from credit_engine.generator import generate_synthetic_data
from credit_engine.modeling import fit_logistic_scorecard, split_out_of_time
from credit_engine.takeup import fit_take_up_model


def _models_and_features():
    data = generate_synthetic_data("smoke", seed=71)
    features = build_decision_features(
        data["applications"], data["bureau_snapshots"], data["cashflow_transactions"]
    )
    risk_data = build_matured_training_dataset(
        features, data["loan_outcomes"], data["fraud_outcomes"], "2027-01-01"
    )
    take_up_data = build_matured_take_up_dataset(features, data["take_up_outcomes"], "2027-01-01")
    risk_train = split_out_of_time(risk_data).train
    default_model = fit_logistic_scorecard(risk_train, target="defaulted")
    fraud_model = fit_logistic_scorecard(risk_train, target="fraud_confirmed")
    take_up_model = fit_take_up_model(take_up_data)
    return features, default_model, fraud_model, take_up_model


def test_expected_value_penalizes_higher_default_and_fraud_risk() -> None:
    safe = calculate_expected_value(0.05, 0.01, 0.8, 200, 2)
    risky = calculate_expected_value(0.35, 0.12, 0.8, 200, 2)
    assert safe > risky


def test_policy_selects_exactly_one_safe_action_per_application() -> None:
    features, default_model, fraud_model, take_up_model = _models_and_features()
    decisions = select_expected_value_policy(features, default_model, fraud_model, take_up_model)
    assert len(decisions) == len(features)
    assert decisions["application_id"].is_unique
    assert set(decisions["approved_amount"]).issubset({0, 50, 100, 200, 300})
    assert decisions["decision"].eq("approve").any()
    assert decisions.loc[decisions["decision"] == "approve", "expected_value"].gt(0).all()
    assert decisions.loc[decisions["decision"] == "decline", "approved_amount"].eq(0).all()
