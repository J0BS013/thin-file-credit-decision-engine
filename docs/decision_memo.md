# Decision Memo — Synthetic First-Loan Policy MVP

## Recommendation

Use `expected_value_v1` only as the synthetic champion candidate for further backtesting. It selects amount and verification friction from separate default, fraud, and take-up probabilities instead of optimizing a single technical metric.

## Why

The engine declines non-positive expected-value scenarios and exposes its assumptions. It is therefore auditable and comparable with `rules_v1` on the same matured population.

## Risks

The data is synthetic; probability calibration and take-up relationships may not transfer; fairness is not assessed; and randomized evidence is still required before reducing verification requirements.

## Monitoring plan

Monitor approval, take-up, FPD30, fraud, contribution per application, calibration, and drift by country, thin-file status, amount, and policy version. Roll back if guardrails deteriorate beyond pre-specified limits.
