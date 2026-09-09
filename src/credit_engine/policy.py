"""Interpretable baseline policy used as the champion before ML challengers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_engine.contracts import assert_no_label_columns


def apply_rule_based_policy(features: pd.DataFrame) -> pd.DataFrame:
    """Produce a deterministic first-loan decision from bureau and cash-flow signals.

    This is a deliberately simple benchmark, not a calibrated probability model.
    Later milestones replace its heuristic risk estimate with calibrated models.
    """
    assert_no_label_columns(features)
    required = {"application_id", "requested_amount", "bureau_score", "bureau_missing", "trailing_cashflow_amount"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Missing policy inputs: {sorted(missing)}")

    bureau_score = features["bureau_score"].fillna(600)
    cashflow = features["trailing_cashflow_amount"].fillna(0)
    heuristic_pd = np.full(len(features), 0.12, dtype=float)
    heuristic_pd += features["bureau_missing"].astype(float).to_numpy() * 0.07
    heuristic_pd += (bureau_score < 600).astype(float).to_numpy() * 0.10
    heuristic_pd -= (bureau_score >= 700).astype(float).to_numpy() * 0.04
    heuristic_pd -= (cashflow >= 300).astype(float).to_numpy() * 0.03
    heuristic_pd = np.clip(heuristic_pd, 0.02, 0.40)

    approved = heuristic_pd <= 0.20
    approved_amount = np.select(
        [approved & (heuristic_pd <= 0.07), approved & (heuristic_pd <= 0.13), approved],
        [300, 200, 100],
        default=0,
    )
    reason_code = np.where(
        ~approved,
        "high_heuristic_risk",
        np.where(features["bureau_missing"], "thin_file_with_cashflow", "sufficient_bureau_and_cashflow"),
    )
    return pd.DataFrame(
        {
            "application_id": features["application_id"],
            "policy_version": "rules_v1",
            "decision": np.where(approved, "approve", "decline"),
            "approved_amount": approved_amount.astype(int),
            "heuristic_pd": np.round(heuristic_pd, 4),
            "reason_code": reason_code,
        }
    )
