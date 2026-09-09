from __future__ import annotations

import pandas as pd
import pytest

from credit_engine.contracts import PointInTimeViolation
from credit_engine.features import build_decision_features
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
