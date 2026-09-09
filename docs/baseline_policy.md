# Baseline Policy and Backtest Scope

`rules_v1` is the initial champion: an interpretable heuristic based on bureau availability, bureau score, and trailing cash-flow amount. It approves applicants with a heuristic risk estimate at or below 20% and assigns US$100, US$200, or US$300 according to that estimate.

The rule is intentionally not calibrated. It exists to establish a transparent benchmark before scorecard and gradient-boosting challengers are introduced.

The backtest reports only matured observed outcomes for approved applications, grouped by origination month and country. Approval rate, observed default rate, and observed fraud rate are descriptive results of this historical synthetic fixture; they are not causal policy effects or profit claims.
