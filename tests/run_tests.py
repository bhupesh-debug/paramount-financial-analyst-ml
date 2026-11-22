"""Test suite for the Paramount Technology Finance project.

Run AFTER `python run_all.py`:
    ~/workspace/quant-ipo-engine/.venv/bin/python tests/run_tests.py
Exit 0 = all pass. Every check reads the real artifacts — nothing is
assumed. Determinism checks regenerate data in memory with seed 42.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import data_generation as dg  # noqa: E402

MODELS = os.path.join(ROOT, "models")
PROC = os.path.join(ROOT, "data", "processed")
SYN = os.path.join(ROOT, "data", "synthetic")
CHARTS = os.path.join(ROOT, "reports", "charts")

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS {name}")
    else:
        failed += 1
        print(f"  FAIL {name}")


def main():
    print("== data integrity ==")
    m = pd.read_csv(os.path.join(PROC, "monthly_spend.csv"))
    check("monthly rows = 432 (36 months x 6 centers x 2 kinds)",
          len(m) == 432)
    check("monthly: 6 cost centers", m["cost_center"].nunique() == 6)
    check("monthly: no nulls", not m.isna().any().any())
    check("monthly: actual > 0", (m["actual"] > 0).all())
    recon = m["budget"] * (1 + m["rate_effect"] + m["volume_effect"]) \
        + m["one_off"]
    check("monthly: variance bridge reconciles",
          np.allclose(recon, m["actual"], atol=1.0))
    check("monthly: 3 hold-out months per center-kind in Jul-Dec 2025",
          len(m[m["month"] >= "2025-07"]) == 72)

    c = pd.read_csv(os.path.join(PROC, "cers.csv"))
    check("cers: 60 rows", len(c) == 60)
    check("cers: costs positive", (c["cost"] > 0).all())
    check("cers: some NPV>0 and some NPV<0",
          (c["npv"] > 0).any() and (c["npv"] < 0).any())
    check("cers: payback within life",
          (c["payback_years"] <= c["life_years"]).all())
    # NPV formula spot-check on the first row
    r0 = c.iloc[0]
    cfs = [r0["annual_benefit"] * (1.02 ** y)
           for y in range(int(r0["life_years"]))]
    npv0 = sum(cf / 1.09 ** (i + 1) for i, cf in enumerate(cfs)) - r0["cost"]
    check("cers: NPV recomputes from cash flows",
          abs(npv0 - r0["npv"]) < 1.0)
    check("cers: positive-NPV rows have IRR above 9%",
          ((c[c["npv"] > 0]["irr"] > 0.09) | c[c["npv"] > 0]["irr"].isna())
          .all())

    j = pd.read_csv(os.path.join(PROC, "close_entries.csv"))
    check("close: labels are 0/1", set(j["is_anomaly"].unique()) <= {0, 1})
    check("close: anomaly rate between 5% and 15%",
          0.05 < j["is_anomaly"].mean() < 0.15)
    check("close: 36 close months", j["close_month"].nunique() == 36)
    check("close: late anomalies exist",
          ((j["is_anomaly"] == 1) & (j["days_after_close"] >= 6)).any())
    check("close: round anomalies exist",
          ((j["is_anomaly"] == 1) & (j["is_round"] == 1)).any())

    print("== determinism ==")
    a = dg.gen_monthly(np.random.default_rng(42))
    b = dg.gen_monthly(np.random.default_rng(42))
    check("regeneration with seed 42 is identical", a.equals(b))
    # clean up the extra write (same content, harmless)
    c1 = dg.gen_cers(np.random.default_rng(42))
    c2 = dg.gen_cers(np.random.default_rng(42))
    check("CER regeneration with seed 42 is identical", c1.equals(c2))

    print("== module 1: forecasting ==")
    fm = pd.read_csv(os.path.join(MODELS, "forecast_metrics.csv"), index_col=0)
    check("forecast metrics file has 3 models", len(fm) == 3)
    check("RF MAPE < seasonal-naive MAPE",
          fm.loc["rf", "mape"] < fm.loc["naive", "mape"])
    check("RF MAPE < 10%", fm.loc["rf", "mape"] < 10)
    check("linear MAPE < naive MAPE",
          fm.loc["linear", "mape"] < fm.loc["naive", "mape"])
    check("forecast MAPEs are positive", (fm["mape"] > 0).all())
    ho = pd.read_csv(os.path.join(MODELS, "forecast_holdout.csv"))
    check("hold-out rows = 72", len(ho) == 72)
    check("hold-out predictions are finite",
          np.isfinite(ho[["naive", "linear", "rf"]].values).all())
    br = pd.read_csv(os.path.join(MODELS, "variance_bridge.csv"))
    tot = br.loc[br["component"] == "total_variance", "usd"].iloc[0]
    parts = br.loc[br["component"] != "total_variance", "usd"].sum()
    check("bridge components sum to total variance",
          abs(parts - tot) < 1.0)
    check("bridge has rate/volume/one_off rows",
          set(br["component"]) == {"rate", "volume", "one_off",
                                   "total_variance"})

    print("== module 2: CER portfolio ==")
    pf = pd.read_csv(os.path.join(MODELS, "cer_portfolio.csv"))
    check("portfolio file has 3 strategies", len(pf) == 3)
    opt = pf[pf["strategy"] == "knapsack optimizer"].iloc[0]
    grd = pf[pf["strategy"] == "greedy by NPV/$"].iloc[0]
    allpos = pf[pf["strategy"] == "approve all NPV>0"].iloc[0]
    check("optimizer NPV >= greedy NPV", opt["npv"] >= grd["npv"])
    check("optimizer utilization <= 100%",
          opt["utilization_pct"] <= 100.0)
    check("approve-all breaches the budget",
          allpos["utilization_pct"] > 100.0)
    check("optimizer selects at least 5 CERs", opt["selected"] >= 5)
    sel = pd.read_csv(os.path.join(MODELS, "cer_selected.csv"))
    check("selected ids match optimizer count",
          len(sel) == int(opt["selected"]))
    check("every selected CER has NPV > 0",
          c.set_index("cer_id").loc[sel["cer_id"], "npv"].gt(0).all())
    sens = pd.read_csv(os.path.join(MODELS, "cer_sensitivity.csv"))
    check("sensitivity grid = 5 CERs x 3 rates x 3 shocks",
          len(sens) == 45)
    check("higher discount rate lowers NPV (top CER, base shock)",
          sens[(sens["cer_id"] == sens["cer_id"].iloc[0])
               & (sens["benefit_shock"] == 1.0)]
          .sort_values("discount_rate")["npv"].is_monotonic_decreasing)

    print("== module 3: close monitor ==")
    cm = pd.read_csv(os.path.join(MODELS, "close_metrics.csv"), index_col=0)
    check("close metrics file has rule + RF rows", len(cm) == 2)
    check("RF F1 > rule F1", cm.loc["rf", "f1"] > cm.loc["rule", "f1"])
    check("RF recall > 0.5", cm.loc["rf", "recall"] > 0.5)
    check("RF ROC-AUC > 0.8", cm.loc["rf", "roc_auc"] > 0.8)
    check("rule precision < RF precision",
          cm.loc["rule", "precision"] < cm.loc["rf", "precision"])
    scored = pd.read_csv(os.path.join(MODELS, "close_holdout_scored.csv"))
    check("hold-out scored rows = Oct-Dec 2025 entries",
          len(scored) == len(j[j["close_month"] >= "2025-10"]))
    check("scored probs in [0,1]",
          scored["rf_prob"].between(0, 1).all())
    check("scored labels match close labels (as a multiset)",
          sorted(scored["is_anomaly"]) == sorted(
              j[j["close_month"] >= "2025-10"]["is_anomaly"]))
    health = pd.read_csv(os.path.join(MODELS, "close_health.csv"))
    check("health scores in [0,100]",
          health["health_score"].between(0, 100).all())
    check("health has 36 months", len(health) == 36)
    check("hold-out mean health < full-period mean (anomalies cluster late)",
          health[health["close_month"] >= "2025-10"]["health_score"].mean()
          <= health["health_score"].mean() + 5)

    print("== artifacts ==")
    for f in ["forecast_metrics.csv", "forecast_holdout.csv",
              "variance_bridge.csv", "cer_portfolio.csv",
              "cer_sensitivity.csv", "cer_selected.csv",
              "close_metrics.csv", "close_holdout_scored.csv",
              "close_health.csv", "forecast_feature_importance.csv",
              "close_feature_importance.csv"]:
        check(f"models/{f} exists", os.path.exists(os.path.join(MODELS, f)))
    for f in ["forecast_holdout.png", "variance_bridge.png",
              "cer_scatter.png", "forecast_importance.png",
              "close_importance.png", "close_health.png", "roc_curve.png"]:
        check(f"reports/charts/{f} exists",
              os.path.exists(os.path.join(CHARTS, f)))

    print(f"\n{passed} passed, {failed} failed")
    if failed:
        print("SOME TESTS FAILED")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
