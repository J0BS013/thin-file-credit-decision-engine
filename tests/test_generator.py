from __future__ import annotations

import pandas as pd

from credit_engine.generator import generate_synthetic_data


def test_smoke_generation_is_reproducible() -> None:
    first = generate_synthetic_data(profile="smoke", seed=7)
    second = generate_synthetic_data(profile="smoke", seed=7)
    for table_name in first:
        pd.testing.assert_frame_equal(first[table_name], second[table_name])


def test_profiles_have_expected_application_counts() -> None:
    assert len(generate_synthetic_data("smoke")["applications"]) == 1_000
    assert len(generate_synthetic_data("full")["applications"]) == 50_000


def test_source_tables_have_expected_grain_and_timing_columns() -> None:
    data = generate_synthetic_data("smoke")
    assert data["applications"]["application_id"].is_unique
    assert data["applicants"]["applicant_id"].is_unique
    assert data["loan_outcomes"]["outcome_available_at"].gt(
        data["applications"]["application_timestamp"]
    ).all()
    assert {"event_timestamp", "available_at"}.issubset(data["cashflow_transactions"].columns)
