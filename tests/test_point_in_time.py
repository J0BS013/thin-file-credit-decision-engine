from __future__ import annotations

import pandas as pd
import pytest

from credit_engine.contracts import PointInTimeViolation
from credit_engine.features import build_decision_features, build_matured_training_dataset
from credit_engine.generator import generate_synthetic_data


def test_decision_features_only_use_records_available_at_application_time() -> None:
    data = generate_synthetic_data("smoke", seed=11)
    features = build_decision_features(
        data["applications"], data["bureau_snapshots"], data["cashflow_transactions"]
    )
    assert len(features) == len(data["applications"])
    assert features["application_id"].is_unique


def test_future_bureau_record_is_rejected_as_leakage() -> None:
    data = generate_synthetic_data("smoke", seed=11)
    future_bureau = data["bureau_snapshots"].copy()
    future_bureau.loc[0, "available_at"] = (
        data["applications"].loc[0, "application_timestamp"] + pd.Timedelta(days=1)
    )
    with pytest.raises(PointInTimeViolation, match="available at decision time"):
        build_decision_features(data["applications"], future_bureau, data["cashflow_transactions"])


def test_future_event_is_rejected_even_if_availability_is_malformed() -> None:
    data = generate_synthetic_data("smoke", seed=11)
    future_bureau = data["bureau_snapshots"].copy()
    future_bureau.loc[0, "event_timestamp"] = (
        data["applications"].loc[0, "application_timestamp"] + pd.Timedelta(days=1)
    )
    future_bureau.loc[0, "available_at"] = data["applications"].loc[0, "application_timestamp"]
    with pytest.raises(PointInTimeViolation, match="happen on or before"):
        build_decision_features(data["applications"], future_bureau, data["cashflow_transactions"])


def test_unmatured_outcomes_are_excluded_from_training_data() -> None:
    data = generate_synthetic_data("smoke", seed=11)
    features = build_decision_features(
        data["applications"], data["bureau_snapshots"], data["cashflow_transactions"]
    )
    cutoff = data["applications"]["application_timestamp"].max() + pd.Timedelta(days=60)
    training = build_matured_training_dataset(
        features, data["loan_outcomes"], data["fraud_outcomes"], cutoff
    )
    assert 0 < len(training) < len(features)
    assert {"defaulted", "fraud_confirmed"}.issubset(training.columns)


def test_feature_data_cannot_already_contain_labels() -> None:
    data = generate_synthetic_data("smoke", seed=11)
    features_with_label = data["applications"].assign(defaulted=0)
    with pytest.raises(PointInTimeViolation, match="cannot contain outcome columns"):
        build_matured_training_dataset(
            features_with_label,
            data["loan_outcomes"],
            data["fraud_outcomes"],
            "2027-01-01",
        )
