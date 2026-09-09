"""Expected-value policy selection across loan amounts and verification friction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from credit_engine.contracts import assert_no_label_columns
from credit_engine.modeling import MODEL_FEATURES
from credit_engine.takeup import TAKE_UP_FEATURES


AMOUNTS = (50, 100, 200, 300)
REQUIREMENT_OPTIONS = (
    ("phone_verification", 1, 2.0),
    ("document_verification", 2, 6.0),
)


def calculate_expected_value(
    pd_default,
    p_fraud,
    p_take_up,
    amount,
    requirement_operational_cost,
):
    """Evaluate expected contribution after credit risk, fraud, take-up, and friction.

    Assumptions are deliberately explicit: revenue is 18% of amount, LGD is 90%,
    fraud loss is the full amount, funding cost is 5% of amount, and servicing is
    US$3 per originated loan. Fraud and default are modeled separately and their
    losses are not double-counted.
    """
    amount = np.asarray(amount, dtype=float)
    pd_default = np.asarray(pd_default, dtype=float)
    p_fraud = np.asarray(p_fraud, dtype=float)
    p_take_up = np.asarray(p_take_up, dtype=float)
    revenue = amount * 0.18
    loss_given_default = amount * 0.90
    fraud_loss = amount
    funding_cost = amount * 0.05
    servicing_cost = 3.0
    credit_contribution = (1 - pd_default) * revenue - pd_default * loss_given_default
    expected_originated_value = (1 - p_fraud) * credit_contribution - p_fraud * fraud_loss
    return p_take_up * (expected_originated_value - funding_cost - servicing_cost) - requirement_operational_cost


def select_expected_value_policy(
    features: pd.DataFrame,
    default_model: Pipeline,
    fraud_model: Pipeline,
    take_up_model: Pipeline,
) -> pd.DataFrame:
    """Choose the best candidate offer per application, or decline when all EV is non-positive."""
    assert_no_label_columns(features)
    candidates: list[pd.DataFrame] = []
    base_pd = default_model.predict_proba(features[MODEL_FEATURES])[:, 1]
    base_fraud = fraud_model.predict_proba(features[MODEL_FEATURES])[:, 1]
    for amount in AMOUNTS:
        for requirement, requirement_count, requirement_cost in REQUIREMENT_OPTIONS:
            scenario = features.copy()
            # The synthetic fixture uses requested amount as the offer-amount proxy.
            scenario["requested_amount"] = amount
            scenario["document_requirement_count"] = requirement_count
            scenario["decision_time_minutes"] = 15 * requirement_count
            p_take_up = take_up_model.predict_proba(scenario[TAKE_UP_FEATURES])[:, 1]
            expected_value = calculate_expected_value(
                base_pd, base_fraud, p_take_up, amount, requirement_cost
            )
            candidates.append(
                pd.DataFrame(
                    {
                        "application_id": features["application_id"],
                        "approved_amount": amount,
                        "requirements": requirement,
                        "pd": base_pd,
                        "p_fraud": base_fraud,
                        "p_take_up": p_take_up,
                        "expected_value": expected_value,
                    }
                )
            )
    all_candidates = pd.concat(candidates, ignore_index=True)
    best = all_candidates.loc[all_candidates.groupby("application_id")["expected_value"].idxmax()].copy()
    best["policy_version"] = "expected_value_v1"
    best["decision"] = np.where(best["expected_value"] > 0, "approve", "decline")
    best.loc[best["decision"] == "decline", ["approved_amount", "requirements"]] = [0, "none"]
    best["reason_code"] = np.where(
        best["decision"] == "approve", "positive_expected_value", "non_positive_expected_value"
    )
    return best[
        [
            "application_id",
            "policy_version",
            "decision",
            "approved_amount",
            "requirements",
            "pd",
            "p_fraud",
            "p_take_up",
            "expected_value",
            "reason_code",
        ]
    ].sort_values("application_id", ignore_index=True)
