"""Build a minimal application-level feature set under the timing contract."""

from __future__ import annotations

import pandas as pd

from credit_engine.contracts import assert_available_at_decision_time


def build_decision_features(
    applications: pd.DataFrame,
    bureau_snapshots: pd.DataFrame,
    cashflow_transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate only records available when each application was evaluated."""
    application_columns = ["application_id", "applicant_id", "application_timestamp", "requested_amount"]
    decisions = applications[application_columns].copy()

    bureau = bureau_snapshots.merge(
        decisions[["application_id", "application_timestamp"]], on="application_id", how="inner"
    )
    assert_available_at_decision_time(bureau)

    cashflow = cashflow_transactions.merge(
        decisions[["application_id", "applicant_id", "application_timestamp"]], on="applicant_id", how="inner"
    )
    cashflow = cashflow[cashflow["event_timestamp"] <= cashflow["application_timestamp"]].copy()
    assert_available_at_decision_time(cashflow)

    bureau_features = bureau.groupby("application_id", as_index=False).agg(
        bureau_score=("bureau_score", "max"),
        bureau_missing=("bureau_score", lambda value: value.isna().all()),
    )
    cashflow_features = cashflow.groupby("application_id", as_index=False).agg(
        trailing_cashflow_amount=("amount", "sum"),
        cashflow_transaction_count=("transaction_id", "nunique"),
    )
    return decisions.merge(bureau_features, on="application_id", how="left").merge(
        cashflow_features, on="application_id", how="left"
    )
