# Decision threshold

The API no longer treats the 8 per cent population default rate as an approval rule. It uses a
small expected-value policy instead. For a requested exposure `EAD`, the demonstration assumes a
performing-loan margin, cost to administer the loan, capital cost, and loss given default (LGD):

`EAD × ((1 - PD) × margin - PD × LGD - cost to administer - capital cost)`

With a 12 per cent margin, 2 per cent administration cost, 2 per cent capital cost, and 45 per cent
LGD, approval breaks even at a PD of 0.140351. The service approves below that point and declines
at or above it.

Those inputs are illustrative. A lender would estimate pricing, cost, capital, LGD, prepayment,
and collections from product and portfolio data, then approve the policy through its credit-risk
governance process. This project has none of that data, so the policy is deliberately exposed in
`src/config.py` rather than presented as an observed business rule.

| performing margin | LGD 35% | LGD 45% | LGD 55% |
| ---: | ---: | ---: | ---: |
| 8% | 9.30% | 7.55% | 6.35% |
| 12% | 17.02% | 14.04% | 11.94% |
| 16% | 23.53% | 19.67% | 16.90% |

The table makes the limitation visible: a higher assumed margin raises the cut-off, while a higher
LGD lowers it. The PD model ranks default risk; the decision policy converts that risk into a
commercial choice. They should be reviewed separately.
