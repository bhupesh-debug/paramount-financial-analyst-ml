"""Orchestrator: generate -> validate -> features -> models -> metrics.

Run from the project root:
    ~/workspace/quant-ipo-engine/.venv/bin/python run_all.py

Deterministic: seed 42 everywhere. Hold-out is Jul–Dec 2025 for the
forecast module and Oct–Dec 2025 for the close module. Every metric in
the PDF and deck comes from models/*.csv written here — nothing is
hard-coded.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import charts as ch  # noqa: E402
import data_generation as dg  # noqa: E402
import data_pipeline as dp  # noqa: E402
import feature_engineering as fe  # noqa: E402
import models as md  # noqa: E402

MODELS = os.path.join(ROOT, "models")
PROC = os.path.join(ROOT, "data", "processed")
os.makedirs(MODELS, exist_ok=True)


def main():
    rng = np.random.default_rng(42)
    print("== 1. generate synthetic data ==")
    dg.main()
    print("== 2. validate + clean ==")
    dp.main()

    monthly = pd.read_csv(os.path.join(PROC, "monthly_spend.csv"))
    cers = pd.read_csv(os.path.join(PROC, "cers.csv"))
    close = pd.read_csv(os.path.join(PROC, "close_entries.csv"))

    print("== 3. module 1: forecasting + variance bridge ==")
    feats = fe.build_forecast_features(monthly)
    metrics, imp, test_out = md.run_forecast(feats)
    pd.DataFrame(metrics).T.round(2).to_csv(
        os.path.join(MODELS, "forecast_metrics.csv"))
    imp.to_csv(os.path.join(MODELS, "forecast_feature_importance.csv"))
    test_out.to_csv(os.path.join(MODELS, "forecast_holdout.csv"), index=False)
    bridge = md.variance_bridge(monthly)
    bridge.to_csv(os.path.join(MODELS, "variance_bridge.csv"), index=False)
    for k, v in metrics.items():
        print(f"  {k}: MAPE {v['mape']:.2f}%  MAE ${v['mae']:,.0f}")
    print("  bridge total variance $"
          f"{bridge.loc[bridge['component'] == 'total_variance', 'usd'].iloc[0]:,.0f}")

    print("== 4. module 2: CER scoring + portfolio optimizer ==")
    sens = md.cer_sensitivity(cers)
    sens.to_csv(os.path.join(MODELS, "cer_sensitivity.csv"), index=False)
    portfolio, budget, sel = md.run_cer_portfolio(cers)
    portfolio.to_csv(os.path.join(MODELS, "cer_portfolio.csv"), index=False)
    pd.DataFrame({"cer_id": sel}).to_csv(
        os.path.join(MODELS, "cer_selected.csv"), index=False)
    print(f"  capital budget ${budget:,.0f}")
    print(portfolio.to_string(index=False))

    print("== 5. module 3: close-readiness monitor ==")
    cfeats = fe.build_close_features(close)
    cmetrics, cimp, scored = md.run_close(cfeats)
    pd.DataFrame(cmetrics).T.round(3).to_csv(
        os.path.join(MODELS, "close_metrics.csv"))
    cimp.to_csv(os.path.join(MODELS, "close_feature_importance.csv"))
    scored.to_csv(os.path.join(MODELS, "close_holdout_scored.csv"),
                  index=False)
    health = md.close_health(cfeats)
    health.to_csv(os.path.join(MODELS, "close_health.csv"), index=False)
    for k, v in cmetrics.items():
        print(f"  {k}: P {v['precision']:.3f} R {v['recall']:.3f} "
              f"F1 {v['f1']:.3f}")
    print(f"  hold-out mean health score "
          f"{health[health['close_month'] >= '2025-10']['health_score'].mean():.1f}")

    print("== 6. charts ==")
    ch.main()
    print("done.")


if __name__ == "__main__":
    main()
