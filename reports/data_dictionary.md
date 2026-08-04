# Data dictionary — application_train.csv

307,511 rows, 122 columns. One row is one loan application; `TARGET` is 1 if the client had a payment more than X days late on at least one installment (Home Credit's definition of default), 0 otherwise.

Target balance: 91.9% repaid, 8.1% defaulted — a 11.4:1 imbalance. Accuracy is meaningless here: a model that predicts 'repaid' for every applicant scores 91.9% accuracy while catching zero defaults.

## Missing values (67 of 122 columns affected)

| column | missing % |
|---|---|
| COMMONAREA_MEDI | 69.87% |
| COMMONAREA_AVG | 69.87% |
| COMMONAREA_MODE | 69.87% |
| NONLIVINGAPARTMENTS_MEDI | 69.43% |
| NONLIVINGAPARTMENTS_MODE | 69.43% |
| NONLIVINGAPARTMENTS_AVG | 69.43% |
| FONDKAPREMONT_MODE | 68.39% |
| LIVINGAPARTMENTS_MODE | 68.35% |
| LIVINGAPARTMENTS_MEDI | 68.35% |
| LIVINGAPARTMENTS_AVG | 68.35% |
| FLOORSMIN_MODE | 67.85% |
| FLOORSMIN_MEDI | 67.85% |
| FLOORSMIN_AVG | 67.85% |
| YEARS_BUILD_MODE | 66.5% |
| YEARS_BUILD_MEDI | 66.5% |
| YEARS_BUILD_AVG | 66.5% |
| OWN_CAR_AGE | 65.99% |
| LANDAREA_AVG | 59.38% |
| LANDAREA_MEDI | 59.38% |
| LANDAREA_MODE | 59.38% |
| BASEMENTAREA_MEDI | 58.52% |
| BASEMENTAREA_AVG | 58.52% |
| BASEMENTAREA_MODE | 58.52% |
| EXT_SOURCE_1 | 56.38% |
| NONLIVINGAREA_MEDI | 55.18% |
| NONLIVINGAREA_MODE | 55.18% |
| NONLIVINGAREA_AVG | 55.18% |
| ELEVATORS_MEDI | 53.3% |
| ELEVATORS_MODE | 53.3% |
| ELEVATORS_AVG | 53.3% |
| ... 37 more columns with missing values | |
