# Model Card — Synthetic Thin-File Credit Decisioning

## Intended use

Educational, synthetic demonstration of point-in-time risk modeling and policy selection. It must not be used for real lending, eligibility, pricing, or fraud decisions.

## Models

- Logistic scorecard: interpretable default baseline.
- Calibrated XGBoost: default challenger, calibrated before the final temporal holdout.
- Logistic fraud model: separate fraud probability.
- Logistic take-up model: historical acceptance prediction under observed friction.

## Evaluation

Default models use an out-of-time holdout and report ROC-AUC, PR-AUC, and Brier score. The project does not claim production performance, fairness, or economic gains from synthetic results.

## Risks and limitations

Synthetic relationships can be wrong; missing bureau data may proxy for protected or socioeconomic characteristics; take-up/friction estimates are observational; and economic assumptions are scenario parameters. Real deployment would require governance, fairness analysis, calibration monitoring, human review, and legal/compliance approval.
