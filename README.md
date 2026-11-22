# Technology Finance Forecasting & Capital Portfolio Analytics

**ML portfolio project for the Financial Analyst II role at Paramount
(Technology Finance) — via Randstad**

Author: Bhupesh Chandra Dimri · November 2026

**GitHub repo (to be created by the user):**
https://github.com/bhupesh-debug/paramount-financial-analyst-ml

> **ALL DATA IS SYNTHETIC.** No real Paramount, Randstad, or client figures
> are used anywhere. The generator mimics the structure of the job
> description (six Technology Finance cost centers, monthly opex/capex,
> Capital Expenditure Requests, monthly-close journal entries) so the models
> can be built and tested honestly. Magnitudes, seasonality, and anomaly
> rates are design choices, not company facts.

## The problem (from the JD)

A Financial Analyst II in Technology Finance does three repeating jobs:
forecast next month's technology spend, decide which capital projects
deserve a fixed budget, and close the month without a surprise. This project
builds all three as reproducible, tested code — the monthly close pack a
finance team re-runs, not a demo.

## Three modules

| # | Module | What it does | Models |
|---|--------|--------------|--------|
| 1 | Forecasting & variance | Monthly opex/capex forecast per cost center + budget-vs-actual variance bridge (rate / volume / one-off) | Seasonal naive vs Linear Regression vs Random Forest |
| 2 | CER portfolio | NPV/IRR/payback scoring for 60 Capital Expenditure Requests + sensitivity + portfolio optimizer under a capital budget | Exact NPV/IRR + 0/1 knapsack DP vs greedy |
| 3 | Close readiness | Journal-entry anomaly flags + monthly-close health score (0–100) | Rule baseline vs Random Forest |

## How to run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py          # generate -> validate -> features -> models -> charts
python tests/run_tests.py  # test suite (exit 0 = all pass)
python build_pdf.py        # 10-page project report -> ~/workspace/your_files/
python build_deck.py       # 10-slide deck PDF -> ~/workspace/your_files/
```

Deterministic: seed 42 everywhere. Forecasts use a time-based hold-out
(Jul–Dec 2025); close flags use Oct–Dec 2025 — nothing from the hold-out is
used to fit features, scalers, or models.

## Results (hold-out)

See `reports/model_report.md` for the full numbers. Headline results from
the run that produced this README:

- **Module 1:** Linear Regression beats the seasonal-naive baseline on MAPE
  (see `models/forecast_metrics.csv`). The linear model also beats the
  random forest on this structure — reported openly, not hidden.
- **Module 2:** The knapsack optimizer beats greedy-by-NPV/$ on total NPV
  under the same capital budget (see `models/cer_portfolio.csv`).
- **Module 3:** Random Forest beats the rule baseline on close-anomaly F1
  (see `models/close_metrics.csv`).

## Project structure

```
ml-project-paramount-financial-analyst/
├── README.md
├── requirements.txt
├── run_all.py                 # orchestrator
├── build_pdf.py               # 10-page report builder
├── build_deck.py              # 10-slide deck builder (PDF)
├── make_notebooks.py          # generates notebooks/01-04
├── src/
│   ├── data_generation.py   # synthetic spend / CERs / close entries
│   ├── data_pipeline.py     # validation + cleaning
│   ├── feature_engineering.py
│   ├── models.py            # forecasting, knapsack, close classifier
│   └── charts.py            # matplotlib figures
├── tests/run_tests.py
├── data/synthetic/          # generated CSVs (regenerate with run_all.py)
├── data/processed/          # validated copies
├── models/*.csv             # every metric the report cites
├── reports/charts/*.png
├── reports/model_report.md
└── notebooks/01_eda.ipynb … 04_evaluation.ipynb
```

## Limits — read before citing any number

- Synthetic data proves the **pipeline design**, not performance on real
  books. Injected anomalies are deliberately detectable; real errors are
  subtler — expect lower recall on live data and keep a human in the review
  queue.
- The linear forecast win is partly structural: synthetic actuals are
  budget × small drift + rare shocks. On real spend, with regime changes and
  reorgs, the forest may earn its keep — re-test, don't assume.
- The knapsack assumes independent projects. Real CERs interact (shared
  infrastructure, sequencing); the optimizer is a starting point for the
  portfolio conversation, not the decision.
- The candidate does **not** claim TM1, Longview, or CPA candidacy — all are
  [CONFIRM WITH CANDIDATE] items for the role, not facts in this repo.

## Design continuity

This project extends the candidate's existing variance-and-forecasting line
— the Enterprise Predictive FP&A Variance Engine and the Enterprise FP&A
Predictive ML Engine (both live at github.com/bhupesh-debug) — from
enterprise FP&A into Technology Finance capital allocation and close
readiness. Same discipline (hold-outs, baselines, auditable CSVs), new
decision surface.
