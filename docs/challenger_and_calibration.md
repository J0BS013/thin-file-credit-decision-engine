# XGBoost Challenger and Calibration

The challenger is an `XGBClassifier` trained on the same point-in-time feature set and eligible population as the logistic scorecard. Country is one-hot encoded; missing bureau scores are represented explicitly, preserving thin-file missingness.

The final out-of-time holdout is never used for fitting or calibration. Inside the training period, an earlier partition fits XGBoost and a later partition fits a logistic probability calibrator. The held-out period then reports ROC-AUC, PR-AUC, and Brier score for the calibrated challenger.

This milestone establishes the methodology only. The synthetic fixture is not a claim that XGBoost improves real credit decisions or expected value. Champion/challenger policy selection will wait for calibration diagnostics, fraud/take-up models, and economic backtesting.
