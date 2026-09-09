# Thin-File Credit Decision Engine

A reproducible Decision Science project that simulates credit decisions for applicants with limited traditional credit history. It is designed to show the full path from point-in-time data to an economically grounded policy decision.

> **Status:** Milestone 1 — synthetic data and point-in-time foundations.

## Business problem

A fintech must decide whether to approve an applicant, request additional verification, and choose an initial loan amount without relying on a complete bureau score. The later milestones will compare policies on approval, fraud, repayment, customer friction, and expected value.

## Current scope

- Deterministic synthetic data generator with `smoke` (1,000 applications) and `full` (50,000 applications) profiles.
- Three countries and currencies, incomplete bureau data, alternative cash-flow signals, device events, and free-text employment statements.
- Explicit `event_timestamp`, `available_at`, and `application_timestamp` fields.
- Point-in-time feature builder that rejects events or records unavailable when the application was made.
- Matured-label training dataset builder with an explicit `as_of_timestamp` cutoff.
- Future repayment, fraud, and take-up outcomes kept separate from decision-time features.
- An interpretable `rules_v1` champion policy and approved-loan vintage reporting.
- Logistic-regression scorecard baseline with held-out out-of-time evaluation.
- XGBoost challenger with calibration isolated from the final out-of-time holdout.
- Separate fraud and take-up models; take-up captures the predictive effect of offer friction.

## Architecture

```text
Synthetic source generator
  -> applicants / applications / bureau / cash flow / device signals
  -> point-in-time feature contract
  -> eligible decision dataset
  -> [next] risk, fraud and take-up models
  -> [next] expected-value policy engine and backtest
```

## Data model and timing contract

| Dataset | Grain | Decision-time use |
|---|---|---|
| `applicants` | one row per applicant | identity and country context |
| `applications` | one row per application | decision timestamp and requested amount |
| `bureau_snapshots` | one snapshot per application | only if `available_at <= application_timestamp` |
| `cashflow_transactions` | one transaction | only historical, available transactions |
| `device_events` | one event | only historical, available events |
| `employment_statements` | one statement per application | only if available at decision time |
| `loan_outcomes` | one matured loan outcome | labels only; never a feature source |
| `fraud_outcomes` | one matured fraud outcome | labels only; never a feature source |

## Quick start

```bash
python -m pip install -r requirements.txt
python -m credit_engine --profile smoke --output-dir data/generated/smoke
python -m pytest -q
```

The generated data is synthetic and contains no personal or corporate data.

## Validation

The test suite verifies deterministic generation, dataset cardinalities, smoke/full profile sizes, point-in-time availability, future-event leakage, outcome-column leakage, and label maturity. A feature with `available_at` after the application timestamp raises an error rather than leaking future information into a model. See the [data dictionary](docs/data_dictionary.md) for the contract.

## Roadmap

1. Expected-value policy engine, first-loan sizing, and champion/challenger backtest.
2. Randomized friction experiment, model card, decision memo, and MVP release.

## Limitations

This first milestone generates realistic-looking fixtures for engineering and methodology tests; it does not claim real-world predictive performance, causal lift, or loan profitability. Those claims will only be made after the corresponding models, backtest, and experiment are implemented.
