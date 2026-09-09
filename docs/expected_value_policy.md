# Expected-Value Policy Engine

For every application, `expected_value_v1` evaluates US$50, US$100, US$200, and US$300 with either phone or document verification. It selects the highest expected-value candidate; if every candidate has non-positive value, it declines.

```text
EV = P(take_up) × [
       (1 - P(fraud)) × ((1 - PD) × revenue - PD × LGD)
       - P(fraud) × fraud_loss
       - funding_cost
       - servicing_cost
     ] - requirement_operational_cost
```

The fixture assumptions are 18% revenue, 90% LGD, full principal fraud loss, 5% funding cost, US$3 servicing cost, and US$2/US$6 for phone/document verification. They are scenario parameters, not real lending economics.

Default and fraud remain separate probabilities. The current take-up model uses the requested-amount field as a synthetic offer-amount proxy; a future experiment will test whether friction changes take-up causally.
