# Credit scorecard

WoE/IV binning via optbinning, scaled to points with PDO (points-to-double-odds): 600 points at 50:1 good:bad odds, every 20 points doubles the odds of being good. Base points (from the model intercept alone): 557.0.

## Information value by feature

IV < 0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong, above 0.5 suspiciously strong (usually a leak worth double-checking).

| feature | IV |
|---|---|
| EXT_SOURCE_3 | 0.3289 |
| EXT_SOURCE_2 | 0.3218 |
| EXT_SOURCE_1 | 0.1459 |
| years_employed | 0.1172 |
| age_years | 0.0871 |
| AMT_CREDIT | 0.0567 |
| NAME_EDUCATION_TYPE | 0.0471 |
| CODE_GENDER | 0.0380 |
| REGION_POPULATION_RELATIVE | 0.0367 |
| AMT_ANNUITY | 0.0302 |
| AMT_INCOME_TOTAL | 0.0119 |
| FLAG_OWN_CAR | 0.0064 |
| CNT_CHILDREN | 0.0057 |

## Points table

| feature | bin | WoE | points |
|---|---|---|---|
| EXT_SOURCE_1 | (-inf, 0.24) | -0.7720 | -10.6 |
| EXT_SOURCE_1 | [0.24, 0.40) | -0.2532 | -3.5 |
| EXT_SOURCE_1 | [0.40, 0.53) | 0.1782 | +2.5 |
| EXT_SOURCE_1 | [0.53, 0.64) | 0.4241 | +5.8 |
| EXT_SOURCE_1 | [0.64, 0.72) | 0.7285 | +10.0 |
| EXT_SOURCE_1 | [0.72, inf) | 1.0734 | +14.8 |
| EXT_SOURCE_1 | Special | 0.0000 | +0.0 |
| EXT_SOURCE_1 | Missing | -0.0597 | -0.8 |
| EXT_SOURCE_2 | (-inf, 0.14) | -1.1576 | -26.1 |
| EXT_SOURCE_2 | [0.14, 0.26) | -0.6258 | -14.1 |
| EXT_SOURCE_2 | [0.26, 0.35) | -0.4022 | -9.1 |
| EXT_SOURCE_2 | [0.35, 0.41) | -0.2410 | -5.4 |
| EXT_SOURCE_2 | [0.41, 0.49) | -0.0834 | -1.9 |
| EXT_SOURCE_2 | [0.49, 0.54) | 0.0568 | +1.3 |
| EXT_SOURCE_2 | [0.54, 0.57) | 0.0867 | +2.0 |
| EXT_SOURCE_2 | [0.57, 0.59) | 0.2068 | +4.7 |
| EXT_SOURCE_2 | [0.59, 0.62) | 0.2426 | +5.5 |
| EXT_SOURCE_2 | [0.62, 0.65) | 0.3810 | +8.6 |
| EXT_SOURCE_2 | [0.65, 0.67) | 0.4311 | +9.7 |
| EXT_SOURCE_2 | [0.67, 0.70) | 0.6003 | +13.5 |
| EXT_SOURCE_2 | [0.70, 0.73) | 0.8277 | +18.7 |
| EXT_SOURCE_2 | [0.73, inf) | 1.1401 | +25.7 |
| EXT_SOURCE_2 | Special | 0.0000 | +0.0 |
| EXT_SOURCE_2 | Missing | -0.0213 | -0.5 |
| EXT_SOURCE_3 | (-inf, 0.19) | -1.1139 | -27.2 |
| EXT_SOURCE_3 | [0.19, 0.32) | -0.6089 | -14.9 |
| EXT_SOURCE_3 | [0.32, 0.41) | -0.2285 | -5.6 |
| EXT_SOURCE_3 | [0.41, 0.46) | 0.0533 | +1.3 |
| EXT_SOURCE_3 | [0.46, 0.50) | 0.1482 | +3.6 |
| EXT_SOURCE_3 | [0.50, 0.57) | 0.3402 | +8.3 |
| EXT_SOURCE_3 | [0.57, 0.64) | 0.5468 | +13.3 |
| EXT_SOURCE_3 | [0.64, 0.68) | 0.6836 | +16.7 |
| EXT_SOURCE_3 | [0.68, 0.71) | 0.8545 | +20.8 |
| EXT_SOURCE_3 | [0.71, 0.76) | 0.8879 | +21.7 |
| EXT_SOURCE_3 | [0.76, inf) | 0.9714 | +23.7 |
| EXT_SOURCE_3 | Special | 0.0000 | +0.0 |
| EXT_SOURCE_3 | Missing | -0.1607 | -3.9 |
| AMT_INCOME_TOTAL | (-inf, 76374.00) | 0.0128 | -0.0 |
| AMT_INCOME_TOTAL | [76374.00, 111777.75) | -0.0383 | +0.0 |
| AMT_INCOME_TOTAL | [111777.75, 127649.25) | -0.1031 | +0.1 |
| AMT_INCOME_TOTAL | [127649.25, 180117.00) | -0.0512 | +0.1 |
| AMT_INCOME_TOTAL | [180117.00, 205274.25) | -0.0033 | +0.0 |
| AMT_INCOME_TOTAL | [205274.25, 252225.00) | 0.0675 | -0.1 |
| AMT_INCOME_TOTAL | [252225.00, 297832.50) | 0.1857 | -0.2 |
| AMT_INCOME_TOTAL | [297832.50, inf) | 0.3309 | -0.4 |
| AMT_INCOME_TOTAL | Special | 0.0000 | -0.0 |
| AMT_INCOME_TOTAL | Missing | 0.0000 | -0.0 |
| AMT_CREDIT | (-inf, 158031.00) | 0.2984 | +3.5 |
| AMT_CREDIT | [158031.00, 281884.50) | 0.0435 | +0.5 |
| AMT_CREDIT | [281884.50, 337630.50) | -0.1534 | -1.8 |
| AMT_CREDIT | [337630.50, 407477.25) | -0.2381 | -2.8 |
| AMT_CREDIT | [407477.25, 453440.25) | -0.4246 | -5.0 |
| AMT_CREDIT | [453440.25, 672320.25) | -0.2075 | -2.4 |
| AMT_CREDIT | [672320.25, 898380.00) | 0.0704 | +0.8 |
| AMT_CREDIT | [898380.00, 1101435.75) | 0.1783 | +2.1 |
| AMT_CREDIT | [1101435.75, 1349955.00) | 0.2765 | +3.3 |
| AMT_CREDIT | [1349955.00, inf) | 0.6551 | +7.7 |
| AMT_CREDIT | Special | 0.0000 | +0.0 |
| AMT_CREDIT | Missing | 0.0000 | +0.0 |
| AMT_ANNUITY | (-inf, 12728.25) | 0.1764 | +2.8 |
| AMT_ANNUITY | [12728.25, 16404.75) | 0.1003 | +1.6 |
| AMT_ANNUITY | [16404.75, 26507.25) | -0.0798 | -1.3 |
| AMT_ANNUITY | [26507.25, 29837.25) | -0.1769 | -2.8 |
| AMT_ANNUITY | [29837.25, 31637.25) | -0.3432 | -5.4 |
| AMT_ANNUITY | [31637.25, 35961.75) | -0.0786 | -1.2 |
| AMT_ANNUITY | [35961.75, 43724.25) | 0.0018 | +0.0 |
| AMT_ANNUITY | [43724.25, 52481.25) | 0.2575 | +4.1 |
| AMT_ANNUITY | [52481.25, inf) | 0.4452 | +7.1 |
| AMT_ANNUITY | Special | 0.0000 | +0.0 |
| AMT_ANNUITY | Missing | 0.0000 | +0.0 |
| age_years | (-inf, 25.73) | -0.4271 | -2.2 |
| age_years | [25.73, 28.25) | -0.3987 | -2.1 |
| age_years | [28.25, 31.68) | -0.3213 | -1.7 |
| age_years | [31.68, 34.82) | -0.2469 | -1.3 |
| age_years | [34.82, 38.19) | -0.1414 | -0.7 |
| age_years | [38.19, 42.88) | -0.0076 | -0.0 |
| age_years | [42.88, 44.80) | 0.0280 | +0.1 |
| age_years | [44.80, 47.10) | 0.0561 | +0.3 |
| age_years | [47.10, 50.16) | 0.1263 | +0.7 |
| age_years | [50.16, 54.67) | 0.1885 | +1.0 |
| age_years | [54.67, 59.55) | 0.4051 | +2.1 |
| age_years | [59.55, 63.47) | 0.4573 | +2.4 |
| age_years | [63.47, inf) | 0.6478 | +3.4 |
| age_years | Special | 0.0000 | +0.0 |
| age_years | Missing | 0.0000 | +0.0 |
| years_employed | (-inf, 1.52) | -0.3869 | -5.9 |
| years_employed | [1.52, 2.54) | -0.3426 | -5.2 |
| years_employed | [2.54, 3.55) | -0.2789 | -4.2 |
| years_employed | [3.55, 4.92) | -0.1370 | -2.1 |
| years_employed | [4.92, 6.64) | 0.0088 | +0.1 |
| years_employed | [6.64, 8.41) | 0.1051 | +1.6 |
| years_employed | [8.41, 10.78) | 0.2282 | +3.5 |
| years_employed | [10.78, 18.12) | 0.4277 | +6.5 |
| years_employed | [18.12, inf) | 0.7433 | +11.3 |
| years_employed | Special | 0.0000 | +0.0 |
| years_employed | Missing | 0.4291 | +6.5 |
| REGION_POPULATION_RELATIVE | (-inf, 0.01) | -0.0393 | -0.3 |
| REGION_POPULATION_RELATIVE | [0.01, 0.02) | -0.0646 | -0.5 |
| REGION_POPULATION_RELATIVE | [0.02, 0.02) | -0.1177 | -0.9 |
| REGION_POPULATION_RELATIVE | [0.02, 0.02) | -0.3082 | -2.3 |
| REGION_POPULATION_RELATIVE | [0.02, 0.03) | 0.0319 | +0.2 |
| REGION_POPULATION_RELATIVE | [0.03, 0.04) | 0.2449 | +1.9 |
| REGION_POPULATION_RELATIVE | [0.04, inf) | 0.6368 | +4.8 |
| REGION_POPULATION_RELATIVE | Special | 0.0000 | +0.0 |
| REGION_POPULATION_RELATIVE | Missing | 0.0000 | +0.0 |
| CNT_CHILDREN | (-inf, 0.50) | 0.0507 | +0.5 |
| CNT_CHILDREN | [0.50, 1.50) | -0.1253 | -1.2 |
| CNT_CHILDREN | [1.50, inf) | -0.0803 | -0.8 |
| CNT_CHILDREN | Special | 0.0000 | +0.0 |
| CNT_CHILDREN | Missing | 0.0000 | +0.0 |
| CODE_GENDER | XNA, F | 0.1532 | +3.5 |
| CODE_GENDER | M | -0.2491 | -5.7 |
| CODE_GENDER | Special | 0.0000 | +0.0 |
| CODE_GENDER | Missing | 0.0000 | +0.0 |
| NAME_EDUCATION_TYPE | Academic degree, Higher education | 0.4294 | +8.3 |
| NAME_EDUCATION_TYPE | Incomplete higher, Secondary / secondary special, Lower secondary | -0.1100 | -2.1 |
| NAME_EDUCATION_TYPE | Special | 0.0000 | +0.0 |
| NAME_EDUCATION_TYPE | Missing | 0.0000 | +0.0 |
| FLAG_OWN_CAR | Y | 0.1154 | +4.8 |
| FLAG_OWN_CAR | N | -0.0553 | -2.3 |
| FLAG_OWN_CAR | Special | 0.0000 | +0.0 |
| FLAG_OWN_CAR | Missing | 0.0000 | +0.0 |
