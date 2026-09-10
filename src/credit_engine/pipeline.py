"""Runnable smoke pipeline linking generation, modeling, decisioning, and reporting."""

from __future__ import annotations

import json
from pathlib import Path

from credit_engine.backtest import compare_policies
from credit_engine.expected_value import select_expected_value_policy
from credit_engine.experiments import analyze_friction_experiment
from credit_engine.features import build_decision_features, build_matured_take_up_dataset, build_matured_training_dataset
from credit_engine.generator import generate_synthetic_data
from credit_engine.modeling import evaluate_binary_model, fit_logistic_scorecard, split_out_of_time
from credit_engine.policy import apply_rule_based_policy
from credit_engine.takeup import fit_take_up_model


def run_smoke_pipeline(output_dir: str | Path, seed: int = 20260909) -> dict:
    """Run the critical path and persist a compact, JSON-serializable report."""
    data = generate_synthetic_data("smoke", seed=seed)
    features = build_decision_features(data["applications"], data["bureau_snapshots"], data["cashflow_transactions"])
    risk_data = build_matured_training_dataset(features, data["loan_outcomes"], data["fraud_outcomes"], "2027-01-01")
    take_up_data = build_matured_take_up_dataset(features, data["take_up_outcomes"], "2027-01-01")
    split = split_out_of_time(risk_data)
    default_model = fit_logistic_scorecard(split.train, "defaulted")
    fraud_model = fit_logistic_scorecard(split.train, "fraud_confirmed")
    take_up_model = fit_take_up_model(take_up_data)
    rules = apply_rule_based_policy(features)
    expected_value = select_expected_value_policy(features, default_model, fraud_model, take_up_model)
    report = {
        "seed": seed,
        "applications": len(features),
        "default_holdout_metrics": evaluate_binary_model(default_model, split.test, "defaulted"),
        "fraud_holdout_metrics": evaluate_binary_model(fraud_model, split.test, "fraud_confirmed"),
        "policy_backtest": compare_policies(risk_data, {"rules_v1": rules, "expected_value_v1": expected_value}).to_dict("records"),
        "friction_experiment": analyze_friction_experiment(
            data["experiment_assignments"], data["experiment_outcomes"]
        ).to_dict("records"),
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "mvp_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
