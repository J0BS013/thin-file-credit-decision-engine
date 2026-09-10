"""Interactive, synthetic-only demonstration for the credit decision engine."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from credit_engine.backtest import compare_policies
from credit_engine.expected_value import evaluate_expected_value_candidates, select_expected_value_policy
from credit_engine.experiments import analyze_friction_experiment
from credit_engine.features import build_decision_features, build_matured_take_up_dataset, build_matured_training_dataset
from credit_engine.generator import generate_synthetic_data
from credit_engine.modeling import evaluate_binary_model, fit_logistic_scorecard, split_out_of_time
from credit_engine.policy import apply_rule_based_policy
from credit_engine.takeup import fit_take_up_model


COLORS = {"positive": "#24c78b", "negative": "#ff6b6b", "default": "#6ea8fe", "fraud": "#f6bd60", "take_up": "#b99cff"}


@st.cache_resource(show_spinner="Running the synthetic decisioning pipeline...")
def build_demo(seed: int) -> dict[str, object]:
    """Build the same smoke-sized artifacts exercised by CI."""
    data = generate_synthetic_data("smoke", seed=seed)
    features = build_decision_features(data["applications"], data["bureau_snapshots"], data["cashflow_transactions"])
    risk_data = build_matured_training_dataset(features, data["loan_outcomes"], data["fraud_outcomes"], "2027-01-01")
    take_up_data = build_matured_take_up_dataset(features, data["take_up_outcomes"], "2027-01-01")
    split = split_out_of_time(risk_data)
    default_model = fit_logistic_scorecard(split.train, "defaulted")
    fraud_model = fit_logistic_scorecard(split.train, "fraud_confirmed")
    take_up_model = fit_take_up_model(take_up_data)
    candidates = evaluate_expected_value_candidates(features, default_model, fraud_model, take_up_model)
    expected_value = select_expected_value_policy(features, default_model, fraud_model, take_up_model)
    rules = apply_rule_based_policy(features)
    return {
        "features": features,
        "decisions": expected_value,
        "candidates": candidates,
        "backtest": compare_policies(risk_data, {"rules_v1": rules, "expected_value_v1": expected_value}),
        "experiment": analyze_friction_experiment(data["experiment_assignments"], data["experiment_outcomes"]),
        "default_metrics": evaluate_binary_model(default_model, split.test, "defaulted"),
        "fraud_metrics": evaluate_binary_model(fraud_model, split.test, "fraud_confirmed"),
    }


def percent(value: float) -> str:
    return f"{value:.1%}"


def usd(value: float) -> str:
    return f"US${value:,.2f}"


def readable_value(feature: str, value: object) -> str:
    if pd.isna(value):
        return "Unavailable (thin file)"
    if feature == "bureau_missing":
        return "Yes" if bool(value) else "No"
    if feature in {"requested_amount", "trailing_cashflow_amount"}:
        return usd(float(value))
    if feature == "decision_time_minutes":
        return f"{int(value)} minutes"
    return str(value)


st.set_page_config(page_title="Thin-File Credit Decision Demo", page_icon="💳", layout="wide")
st.title("Thin-File Credit Decision Engine")
st.caption("Decision simulator: probability estimates become an auditable economic action.")
st.warning("Synthetic data only. This demo is for methodology review and portfolio presentation; it is not production lending software and does not assess real applicants.")

with st.sidebar:
    st.header("Demo controls")
    seed = st.number_input("Synthetic scenario seed", min_value=1, value=20260909, step=1)
    st.caption("Changing the seed regenerates a deterministic synthetic portfolio.")

demo = build_demo(int(seed))
features = demo["features"]
decisions = demo["decisions"]
candidates = demo["candidates"]
backtest = demo["backtest"]
experiment = demo["experiment"]

decision_view, portfolio_view, validation_view = st.tabs(["Decision simulator", "Portfolio economics", "Validation and experiment"])

with decision_view:
    choice_col, selector_col = st.columns([1, 3])
    with choice_col:
        decision_filter = st.radio("Example type", ["Approved", "Declined"], horizontal=True)
    decision_value = {"Approved": "approve", "Declined": "decline"}[decision_filter]
    filtered = decisions.loc[decisions["decision"] == decision_value, "application_id"].tolist()
    with selector_col:
        application_id = st.selectbox(
            f"Select a synthetic {decision_filter.lower()} application",
            options=filtered,
            key=f"application_{decision_filter.lower()}",
        )

    application = features.loc[features["application_id"] == application_id].iloc[0]
    decision = decisions.loc[decisions["application_id"] == application_id].iloc[0]
    scenario = candidates.loc[candidates["application_id"] == application_id].copy()
    scenario["label"] = scenario["approved_amount"].map(lambda amount: f"US${amount}") + " · " + scenario["requirements"].str.replace("_", " ")
    scenario["result"] = scenario["expected_value"].map(lambda value: "positive" if value > 0 else "negative")

    st.subheader("Recommended policy action")
    metric_one, metric_two, metric_three, metric_four = st.columns(4)
    metric_one.metric("Decision", str(decision["decision"]).upper())
    metric_two.metric("Recommended amount", usd(float(decision["approved_amount"])))
    metric_three.metric("Expected value", usd(float(decision["expected_value"])))
    metric_four.metric("Verification", str(decision["requirements"]).replace("_", " ").title())

    risk_col, fraud_col, takeup_col, reason_col = st.columns(4)
    risk_col.metric("Predicted default risk", percent(float(decision["pd"])))
    fraud_col.metric("Predicted fraud risk", percent(float(decision["p_fraud"])))
    takeup_col.metric("Predicted take-up", percent(float(decision["p_take_up"])))
    reason_col.metric("Policy reason", str(decision["reason_code"]).replace("_", " "))

    left, right = st.columns(2)
    with left:
        st.subheader("Why this action wins")
        candidate_chart = px.bar(scenario, x="label", y="expected_value", color="result", color_discrete_map=COLORS, text=scenario["expected_value"].map(usd), labels={"label": "Offer and verification", "expected_value": "Expected value (USD)"})
        candidate_chart.add_hline(y=0, line_color="#9aa0a6", line_width=1)
        candidate_chart.update_layout(showlegend=False, xaxis_tickangle=-28, margin=dict(t=25, b=80, l=5, r=5))
        st.plotly_chart(candidate_chart, width="stretch")
        st.caption("The policy evaluates all amount and verification combinations, then approves only the best positive-value scenario.")
    with right:
        st.subheader("Probability estimates are not the decision")
        probabilities = pd.DataFrame({"signal": ["Default risk", "Fraud risk", "Take-up"], "probability": [decision["pd"], decision["p_fraud"], decision["p_take_up"]], "color": ["default", "fraud", "take_up"]})
        probability_chart = px.bar(probabilities, x="signal", y="probability", color="color", color_discrete_map=COLORS, text="probability", labels={"signal": "", "probability": "Predicted probability"})
        probability_chart.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        probability_chart.update_layout(showlegend=False, yaxis_tickformat=".0%", yaxis_range=[0, 1], margin=dict(t=25, b=5, l=5, r=5))
        st.plotly_chart(probability_chart, width="stretch")
        st.caption("Expected value combines separate risk estimates with amount, loss, funding, servicing, and verification costs.")

    st.subheader("Decision-time evidence")
    feature_labels = {"country_code": "Country", "requested_amount": "Requested amount", "bureau_score": "Bureau score", "bureau_missing": "Bureau unavailable", "trailing_cashflow_amount": "Trailing cash flow", "cashflow_transaction_count": "Cash-flow transactions", "document_requirement_count": "Original document requirements", "decision_time_minutes": "Original decision time"}
    input_frame = pd.DataFrame([{"Input": feature_labels[feature], "Value": readable_value(feature, application[feature])} for feature in feature_labels])
    st.dataframe(input_frame, hide_index=True, width="stretch")

with portfolio_view:
    st.subheader("Economic comparison on the same matured synthetic population")
    comparison = backtest.copy()
    comparison["policy_label"] = comparison["policy"].map({"rules_v1": "Rule-based baseline", "expected_value_v1": "Expected-value policy"})
    economics_chart = px.bar(comparison, x="policy_label", y="observed_contribution_per_application", color="policy_label", color_discrete_sequence=["#ff8c69", "#24c78b"], text=comparison["observed_contribution_per_application"].map(usd), labels={"policy_label": "", "observed_contribution_per_application": "Observed contribution per eligible application (USD)"})
    economics_chart.add_hline(y=0, line_color="#9aa0a6", line_width=1)
    economics_chart.update_layout(showlegend=False, margin=dict(t=25, b=5, l=5, r=5))
    st.plotly_chart(economics_chart, width="stretch")
    display = comparison[["policy_label", "approved_applications", "approval_rate", "observed_default_rate", "observed_fraud_rate", "observed_contribution"]].copy()
    for column in ["approval_rate", "observed_default_rate", "observed_fraud_rate"]:
        display[column] = display[column].map(percent)
    display["observed_contribution"] = display["observed_contribution"].map(usd)
    display.columns = ["Policy", "Approved applications", "Approval rate", "Observed default rate", "Observed fraud rate", "Observed contribution"]
    st.dataframe(display, hide_index=True, width="stretch")
    st.caption("This is a backtest on synthetic matured outcomes, not a causal policy estimate or a real-world profitability claim.")

with validation_view:
    left, right = st.columns(2)
    with left:
        st.subheader("Out-of-time model checks")
        model_metrics = pd.DataFrame([{"model": "Default scorecard", **demo["default_metrics"]}, {"model": "Fraud scorecard", **demo["fraud_metrics"]}]).round(3)
        st.dataframe(model_metrics, hide_index=True, width="stretch")
        st.caption("Models are evaluated separately on a later holdout period; probability quality is tracked with Brier score.")
    with right:
        st.subheader("Randomized friction experiment")
        experiment_chart = px.bar(experiment, x="assignment", y="completion_rate", color="assignment", color_discrete_sequence=["#6ea8fe", "#24c78b"], text="completion_rate", labels={"assignment": "Experiment assignment", "completion_rate": "Completion rate"})
        experiment_chart.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        experiment_chart.update_layout(showlegend=False, yaxis_tickformat=".0%", yaxis_range=[0, 1], margin=dict(t=25, b=5, l=5, r=5))
        st.plotly_chart(experiment_chart, width="stretch")
        experiment_delta = float(experiment["itt_completion_difference"].iloc[0])
        st.metric("Intention-to-treat completion difference", percent(experiment_delta))
        st.caption("This separate randomized experiment is how reduced-friction hypotheses should be evaluated.")
