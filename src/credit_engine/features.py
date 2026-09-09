"""Build a minimal application-level feature set under the timing contract."""

from __future__ import annotations

import pandas as pd

from credit_engine.contracts import (
    PointInTimeViolation,
    assert_available_at_decision_time,
    assert_event_happened_before_decision,
    assert_no_label_columns,
)


def build_decision_features(
    applications: pd.DataFrame,
    bureau_snapshots: pd.DataFrame,
    cashflow_transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate only records available when each application was evaluated."""
    application_columns = [
        "application_id",
        "applicant_id",
        "application_timestamp",
        "requested_amount",
        "country_code",
    ]
    decisions = applications[application_columns].copy()

    bureau = bureau_snapshots.merge(
        decisions[["application_id", "application_timestamp"]], on="application_id", how="inner"
    )
    assert_available_at_decision_time(bureau)
    assert_event_happened_before_decision(bureau)

    cashflow = cashflow_transactions.merge(
        decisions[["application_id", "applicant_id", "application_timestamp"]], on="applicant_id", how="inner"
    )
    cashflow = cashflow[cashflow["event_timestamp"] <= cashflow["application_timestamp"]].copy()
    assert_available_at_decision_time(cashflow)
    assert_event_happened_before_decision(cashflow)

    bureau_features = bureau.groupby("application_id", as_index=False).agg(
        bureau_score=("bureau_score", "max"),
        bureau_missing=("bureau_score", lambda value: value.isna().all()),
    )
    cashflow_features = cashflow.groupby("application_id", as_index=False).agg(
        trailing_cashflow_amount=("amount", "sum"),
        cashflow_transaction_count=("transaction_id", "nunique"),
    )
    features = decisions.merge(bureau_features, on="application_id", how="left").merge(
        cashflow_features, on="application_id", how="left"
    )
    assert_no_label_columns(features)
    return features


def build_matured_training_dataset(
    decision_features: pd.DataFrame,
    loan_outcomes: pd.DataFrame,
    fraud_outcomes: pd.DataFrame,
    as_of_timestamp: str | pd.Timestamp,
) -> pd.DataFrame:
    """Attach only labels that were mature by a declared training cutoff."""
    assert_no_label_columns(decision_features)
    cutoff = pd.Timestamp(as_of_timestamp)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")

    labels = loan_outcomes.merge(fraud_outcomes, on="application_id", suffixes=("_loan", "_fraud"))
    mature = labels[
        (pd.to_datetime(labels["outcome_available_at_loan"], utc=True) <= cutoff)
        & (pd.to_datetime(labels["outcome_available_at_fraud"], utc=True) <= cutoff)
    ].copy()
    training = decision_features.merge(
        mature[["application_id", "defaulted", "fraud_confirmed"]], on="application_id", how="inner"
    )
    if training.empty:
        raise PointInTimeViolation("No mature outcomes are available for the requested training cutoff.")
    return training
