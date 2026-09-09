from __future__ import annotations

import pandas as pd

from credit_engine.backtest import compare_policies
from credit_engine.experiments import analyze_friction_experiment, required_sample_size_per_arm
from credit_engine.generator import generate_synthetic_data


def test_policy_backtest_compares_same_eligible_population() -> None:
    training = pd.DataFrame(
        {"application_id": ["a", "b"], "defaulted": [0, 1], "fraud_confirmed": [0, 0]}
    )
    rules = pd.DataFrame(
        {"application_id": ["a", "b"], "decision": ["approve", "decline"], "approved_amount": [100, 0], "requirements": ["phone_verification", "none"]}
    )
    challenger = rules.assign(decision=["approve", "approve"], approved_amount=[100, 100])
    result = compare_policies(training, {"rules_v1": rules, "expected_value_v1": challenger})
    assert result["eligible_applications"].eq(2).all()
    assert set(result["policy"]) == {"rules_v1", "expected_value_v1"}


def test_randomized_experiment_has_both_arms_and_power_plan() -> None:
    data = generate_synthetic_data("smoke", seed=83)
    summary = analyze_friction_experiment(data["experiment_assignments"], data["experiment_outcomes"])
    assert set(summary["assignment"]) == {"control", "treatment"}
    assert required_sample_size_per_arm(0.60, 0.05) > 0
