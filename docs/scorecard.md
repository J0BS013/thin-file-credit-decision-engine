# Logistic Scorecard Baseline

The first model is a regularized logistic regression for `defaulted`. It uses only bureau score/missingness, trailing cash-flow activity, and country available at application time. Missing bureau information remains an explicit feature rather than being silently removed.

Evaluation uses an out-of-time split: every validation application occurs on or after the first validation timestamp, while all training applications occur before it. Reported ROC-AUC, PR-AUC, and Brier score are held-out synthetic-fixture diagnostics, not evidence of real-world credit performance.

Probability calibration, stability analysis, and a gradient-boosting challenger are deliberately deferred to the next milestones.
