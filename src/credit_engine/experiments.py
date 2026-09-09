"""Intention-to-treat analysis for the synthetic friction experiment."""

from __future__ import annotations

import math

import pandas as pd


def analyze_friction_experiment(assignments: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Report randomized completion differences by assigned arm."""
    data = assignments.merge(outcomes, on="application_id", validate="one_to_one")
    summary = data.groupby("assignment", as_index=False).agg(
        assigned_applications=("application_id", "nunique"), completion_rate=("completed_application", "mean")
    )
    if set(summary["assignment"]) != {"control", "treatment"}:
        raise ValueError("Experiment requires both control and treatment assignments.")
    control = summary.loc[summary["assignment"] == "control", "completion_rate"].iloc[0]
    treatment = summary.loc[summary["assignment"] == "treatment", "completion_rate"].iloc[0]
    summary["itt_completion_difference"] = treatment - control
    return summary


def required_sample_size_per_arm(baseline_rate: float, minimum_detectable_effect: float, alpha: float = 0.05) -> int:
    """Approximate two-sided 80%-power sample size for a two-proportion experiment."""
    if not 0 < baseline_rate < 1 or not 0 < minimum_detectable_effect < 1:
        raise ValueError("Rates must be strictly between zero and one.")
    treatment_rate = min(baseline_rate + minimum_detectable_effect, 0.999)
    pooled_variance = 2 * baseline_rate * (1 - baseline_rate)
    alternative_variance = baseline_rate * (1 - baseline_rate) + treatment_rate * (1 - treatment_rate)
    return math.ceil(((1.96 * math.sqrt(pooled_variance) + 0.84 * math.sqrt(alternative_variance)) / minimum_detectable_effect) ** 2)
