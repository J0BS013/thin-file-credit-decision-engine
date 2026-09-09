"""Economic comparison of policies on the same matured synthetic population."""

from __future__ import annotations

import pandas as pd


def compare_policies(training_data: pd.DataFrame, policies: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare observed synthetic contribution; this is not a causal policy estimate."""
    summaries = []
    for name, decisions in policies.items():
        merged = training_data.merge(decisions, on="application_id", validate="one_to_one")
        approved = merged.loc[merged["decision"] == "approve"].copy()
        amount = approved["approved_amount"].astype(float)
        requirement_cost = approved["requirements"].map(
            {"phone_verification": 2.0, "document_verification": 6.0}
        ).fillna(0.0)
        realized = (
            (1 - approved["fraud_confirmed"]) * ((1 - approved["defaulted"]) * amount * 0.18 - approved["defaulted"] * amount * 0.90)
            - approved["fraud_confirmed"] * amount
            - amount * 0.05
            - 3.0
            - requirement_cost
        )
        summaries.append(
            {
                "policy": name,
                "eligible_applications": len(merged),
                "approved_applications": len(approved),
                "approval_rate": len(approved) / len(merged),
                "observed_default_rate": approved["defaulted"].mean() if len(approved) else 0.0,
                "observed_fraud_rate": approved["fraud_confirmed"].mean() if len(approved) else 0.0,
                "observed_contribution": realized.sum(),
                "observed_contribution_per_application": realized.sum() / len(merged),
            }
        )
    return pd.DataFrame(summaries).sort_values("policy", ignore_index=True)
