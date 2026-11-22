"""Generate the four walkthrough notebooks (01 EDA, 02 features,
03 modeling, 04 evaluation) as .ipynb JSON.

Run from the project root AFTER `python run_all.py`:
    ~/workspace/quant-ipo-engine/.venv/bin/python make_notebooks.py
"""

import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(ROOT, "notebooks")


def nb(cells):
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4, "nbformat_minor": 5}


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(True)}


def write(name, cells):
    with open(os.path.join(NB, name), "w") as fh:
        json.dump(nb(cells), fh, indent=1)
    print(f"  notebook: {name}")


os.makedirs(NB, exist_ok=True)

write("01_eda.ipynb", [
    md("# 01 — Exploratory Data Analysis\n\nAll data is synthetic. "
       "Monthly spend: 432 rows (36 months × 6 cost centers × opex/capex). "
       "CERs: 60 Capital Expenditure Requests. Close entries: 1,675 journal "
       "entries across 36 monthly closes."),
    code("import pandas as pd\n"
         "m = pd.read_csv('data/processed/monthly_spend.csv')\n"
         "c = pd.read_csv('data/processed/cers.csv')\n"
         "j = pd.read_csv('data/processed/close_entries.csv')\n"
         "print(m.shape, c.shape, j.shape)"),
    code("m.groupby('cost_center')[['budget','actual']].sum()"),
    code("m.groupby('month')[['budget','actual']].sum().plot("
         "title='Total monthly spend: budget vs actual')"),
    code("c['npv'].describe(), (c['npv'] > 0).mean()"),
    code("j.groupby('close_month')['is_anomaly'].mean().plot("
         "title='Anomaly rate by close month')"),
])

write("02_feature_engineering.ipynb", [
    md("# 02 — Feature Engineering\n\nForecast features: lag-1 / lag-12 "
       "actuals, 3-month rolling mean, trend, sin/cos month, budget, "
       "headcount, cloud-usage index. Close features: log amount, days after "
       "close, weekend/round/reversal flags, amount z-score within "
       "account+month, duplicate count, poster count. Features are built "
       "from the full series (lags only look backward — no leakage)."),
    code("import sys; sys.path.insert(0, 'src')\n"
         "import pandas as pd, feature_engineering as fe\n"
         "m = pd.read_csv('data/processed/monthly_spend.csv')\n"
         "f = fe.build_forecast_features(m)\n"
         "print(f.shape)\n"
         "print(f.columns.tolist())"),
    code("j = pd.read_csv('data/processed/close_entries.csv')\n"
         "cf = fe.build_close_features(j)\n"
         "cf.groupby('is_anomaly')[['log_amount','days_after_close',"
         "'dup_count','amount_z']].mean()"),
])

write("03_modeling.ipynb", [
    md("# 03 — Modeling\n\nModule 1: seasonal naive vs linear regression "
       "vs random forest (time split Jul–Dec 2025). Module 2: NPV/IRR/"
       "payback scoring + 0/1 knapsack DP vs greedy-by-NPV/$. Module 3: "
       "rule flags vs random-forest classifier (time split Oct–Dec 2025)."),
    code("import sys; sys.path.insert(0, 'src')\n"
         "import pandas as pd, feature_engineering as fe, models as md\n"
         "m = pd.read_csv('data/processed/monthly_spend.csv')\n"
         "f = fe.build_forecast_features(m)\n"
         "metrics, imp, test_out = md.run_forecast(f)\n"
         "print(metrics)"),
    code("c = pd.read_csv('data/processed/cers.csv')\n"
         "portfolio, budget, sel = md.run_cer_portfolio(c)\n"
         "print('budget:', budget)\n"
         "print(portfolio)"),
    code("j = pd.read_csv('data/processed/close_entries.csv')\n"
         "cf = fe.build_close_features(j)\n"
         "cm, cimp, scored = md.run_close(cf)\n"
         "print(cm)"),
])

write("04_evaluation.ipynb", [
    md("# 04 — Evaluation\n\nRead the metric CSVs, check the variance "
       "bridge reconciles, inspect the CER sensitivity grid, and look at the "
       "close health score by month. Everything in the PDF report comes "
       "from these files."),
    code("import pandas as pd\n"
         "print(pd.read_csv('models/forecast_metrics.csv'))\n"
         "print(pd.read_csv('models/variance_bridge.csv'))\n"
         "print(pd.read_csv('models/cer_portfolio.csv'))\n"
         "print(pd.read_csv('models/close_metrics.csv'))"),
    code("h = pd.read_csv('models/close_health.csv')\n"
         "h.plot(x='close_month', y='health_score', "
         "title='Close health score by month')"),
    code("s = pd.read_csv('models/cer_sensitivity.csv')\n"
         "s.pivot_table(index='cer_id', columns='discount_rate', "
         "values='npv', aggfunc='mean')"),
])
