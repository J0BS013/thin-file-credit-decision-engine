"""Portfolio and vintage reporting for policy backtests with matured outcomes."""

from __future__ import annotations

import pandas as pd


def backtest_policy(training_data: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    """Return originations with observed matured outcomes; no causal claim is made."""
    required = {"application_id", "application_timestamp", "country_code", "defaulted", "fraud_confirmed"}
    missing = required.difference(training_data.columns)
    if missing:
        raise ValueError(f"Training data missing backtest fields: {sorted(missing)}")
    merged = training_data.merge(decisions, on="application_id", how="inner", validate="one_to_one")
    approved = merged.loc[merged["decision"] == "approve"].copy()
    approved["origination_month"] = (
        pd.to_datetime(approved["application_timestamp"], utc=True).dt.tz_localize(None).dt.to_period("M").astype(str)
    )
    return approved


def vintage_summary(originations: pd.DataFrame) -> pd.DataFrame:
    """Summarize approved applications by vintage and country with reconciled counts."""
    if originations.empty:
        return pd.DataFrame(
            columns=[
                "origination_month",
                "country_code",
                "approved_applications",
                "approved_amount",
                "defaults",
                "fraud_cases",
                "observed_default_rate",
                "observed_fraud_rate",
            ]
        )
    result = originations.groupby(["origination_month", "country_code"], as_index=False).agg(
        approved_applications=("application_id", "nunique"),
        approved_amount=("approved_amount", "sum"),
        defaults=("defaulted", "sum"),
        fraud_cases=("fraud_confirmed", "sum"),
    )
    result["observed_default_rate"] = result["defaults"] / result["approved_applications"]
    result["observed_fraud_rate"] = result["fraud_cases"] / result["approved_applications"]
    return result.sort_values(["origination_month", "country_code"], ignore_index=True)
