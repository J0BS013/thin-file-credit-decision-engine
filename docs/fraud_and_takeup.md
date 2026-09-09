# Fraud and Take-Up Models

Fraud is trained as a separate binary target from repayment default. It uses the same decision-time feature contract and is evaluated independently; the policy engine will combine the two risks explicitly rather than treating them as one event.

The take-up model predicts historical offer acceptance from requested amount, document requirement count, decision time, cash-flow activity, bureau missingness, and country. Its relationship with friction is observational in this milestone. A randomized experiment is required before claiming that reducing requirements causes higher take-up or contribution margin.
