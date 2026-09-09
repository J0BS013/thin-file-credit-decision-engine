# Data Dictionary and Timing Rules

## Decision-time sources

| Dataset | Primary key | Timing rule | Missing-data policy |
|---|---|---|---|
| `applications` | `application_id` | `application_timestamp` defines the decision boundary | not allowed |
| `bureau_snapshots` | `application_id` | `event_timestamp` and `available_at` must be on or before the decision boundary | missing bureau is retained as an informative thin-file signal |
| `cashflow_transactions` | `transaction_id` | transaction and availability timestamps must be on or before the decision boundary | no records aggregate to zero activity |
| `device_events` | `event_id` | event and availability timestamps must be on or before the decision boundary | missing device data is an explicit missingness feature in a later milestone |
| `employment_statements` | `application_id` | statement must be available at the decision boundary | missing text is retained as missing, never imputed from outcomes |

## Outcome-only sources

`loan_outcomes`, `fraud_outcomes`, and `take_up_outcomes` are labels, not feature sources. They may join a training dataset only when their outcome timestamps are on or before a declared `as_of_timestamp`.

## Contract

Every decision-time feature record must satisfy:

```text
event_timestamp <= application_timestamp
available_at <= application_timestamp
```

Features cannot contain `defaulted`, `fraud_confirmed`, `taken_up`, or `outcome_available_at`. Violations raise `PointInTimeViolation` and fail the test suite.
