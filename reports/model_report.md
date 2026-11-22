# Model report — Technology Finance Forecasting & Capital Portfolio Analytics

All data is synthetic. Hold-outs: forecasting Jul–Dec 2025; close Oct–Dec 2025. Seed 42.

## Module 1 — Forecasting (hold-out, 72 rows)

| Model | MAPE | MAE |
|---|---|---|
| naive | 15.28% | $59k |
| linear | 5.11% | $17k |
| rf | 5.88% | $21k |

Linear regression wins on this structure — reported openly. The budget carries ~79% of the random-forest feature importance.

## Variance bridge (Jul–Dec 2025, $k)

| Component | $k |
|---|---|
| rate | 38.2 |
| volume | -45.8 |
| one_off | -204.9 |
| total_variance | -212.5 |

## Module 2 — CER portfolio

| Strategy | Selected | Cost | NPV | Utilization |
|---|---|---|---|---|
| knapsack optimizer | 19 | $20,406,000 | $6,676,391 | 99.5% |
| greedy by NPV/$ | 25 | $20,368,000 | $6,648,017 | 99.4% |
| approve all NPV>0 | 42 | $40,997,000 | $8,645,814 | 200.0% |

CERs: 60 total, 42 NPV-positive. Sensitivity grid: 44 of 45 cells stay NPV-positive across discount rates 7–11% and benefit shocks -20%/+20%.

## Module 3 — Close monitor (hold-out Oct–Dec 2025)

| Model | Precision | Recall | F1 | Accuracy | ROC-AUC |
|---|---|---|---|---|---|
| rule | 0.700 | 0.824 | 0.757 | 0.938 | — |
| rf | 0.944 | 1.000 | 0.971 | 0.993 | 0.997 |

Close health score: full-period mean 93.0; hold-out mean 91.6; worst month 2025-04 at 88.3.

