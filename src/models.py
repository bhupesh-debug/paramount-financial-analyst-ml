"""Models for the three modules.

Module 1 — forecasting: seasonal-naive baseline (same month last year),
linear regression, random forest. Time split: train < 2025-07, hold-out
2025-07..2025-12. Metrics: MAPE / MAE.
Module 2 — CER scoring: NPV/IRR/payback recomputed from cash flows,
sensitivity (discount rate, benefit), 0/1 knapsack portfolio optimizer
under a capital budget (DP over $10k units) vs greedy baselines.
Module 3 — close monitor: rule baseline vs random-forest anomaly
classifier; train on closes < 2025-10, hold-out Oct–Dec 2025. Metrics:
precision / recall / F1 / ROC-AUC. Close health score 0–100 per month.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


# ---------------- Module 1 ----------------

FORECAST_FEATURES = ["lag1", "lag12", "roll3", "trend", "sin_m", "cos_m",
                     "budget", "headcount", "cloud_usage_idx"]


def mape(y, p):
    return float(np.mean(np.abs(y - p) / np.maximum(y, 1.0)) * 100)


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def run_forecast(feats):
    train = feats[feats["month"] < "2025-07"]
    test = feats[feats["month"] >= "2025-07"]
    Xtr, ytr = train[FORECAST_FEATURES], train["target"]
    Xte, yte = test[FORECAST_FEATURES], test["target"]
    out = {}
    # seasonal naive: same month last year
    p_naive = test["lag12"].values
    out["naive"] = {"mape": mape(yte, p_naive), "mae": mae(yte, p_naive)}
    lr = LinearRegression().fit(Xtr, ytr)
    p_lr = lr.predict(Xte)
    out["linear"] = {"mape": mape(yte, p_lr), "mae": mae(yte, p_lr)}
    rf = RandomForestRegressor(n_estimators=300, random_state=42,
                               min_samples_leaf=2, n_jobs=-1)
    rf.fit(Xtr, ytr)
    p_rf = rf.predict(Xte)
    out["rf"] = {"mape": mape(yte, p_rf), "mae": mae(yte, p_rf)}
    imp = pd.Series(rf.feature_importances_, index=FORECAST_FEATURES) \
        .sort_values(ascending=False)
    test_out = test[["month", "cost_center", "kind", "target"]].copy()
    test_out["naive"] = p_naive
    test_out["linear"] = p_lr
    test_out["rf"] = p_rf
    return out, imp, test_out


def variance_bridge(monthly):
    """Decompose hold-out budget-vs-actual variance into rate / volume /
    one-off dollars. Exact by construction (see data_generation)."""
    h = monthly[monthly["month"] >= "2025-07"].copy()
    h["var"] = h["actual"] - h["budget"]
    h["rate_usd"] = h["budget"] * h["rate_effect"]
    h["vol_usd"] = h["budget"] * h["volume_effect"]
    total = h["var"].sum()
    bridge = pd.DataFrame({
        "component": ["rate", "volume", "one_off", "total_variance"],
        "usd": [h["rate_usd"].sum(), h["vol_usd"].sum(), h["one_off"].sum(),
                total],
    })
    bridge["pct_of_variance"] = bridge["usd"] / total * 100
    return bridge


# ---------------- Module 2 ----------------

def npv(rate, cfs):
    return sum(cf / (1 + rate) ** (i + 1) for i, cf in enumerate(cfs))


def cer_sensitivity(cers, rates=(0.07, 0.09, 0.11), shocks=(0.8, 1.0, 1.2)):
    """NPV of the top-5 NPV CERs under discount-rate and benefit shocks."""
    top = cers.nlargest(5, "npv")
    rows = []
    for _, r in top.iterrows():
        base_cfs = [r["annual_benefit"] * (1.02 ** y)
                    for y in range(int(r["life_years"]))]
        for rate in rates:
            for shock in shocks:
                cfs = [b * shock for b in base_cfs]
                rows.append({
                    "cer_id": r["cer_id"], "discount_rate": rate,
                    "benefit_shock": shock,
                    "npv": round(npv(rate, cfs) - r["cost"], 2),
                })
    return pd.DataFrame(rows)


def knapsack(cers, budget):
    """0/1 knapsack DP (budget in $10k units). Maximizes NPV; only NPV>0
    projects are eligible. Returns selected ids, total cost, total NPV."""
    items = cers[cers["npv"] > 0].reset_index(drop=True)
    unit = 1_000  # $1k units: quantization error <= $1k per item
    cap = int(budget // unit)
    # ceiling: DP cost >= true cost, so a DP-feasible set can never breach
    # the real budget (floor division would silently over-spend)
    costs = np.ceil(items["cost"].values / unit).astype(int)
    values = items["npv"].values
    n = len(items)
    dp = np.zeros((n + 1, cap + 1))
    for i in range(1, n + 1):
        c, v = costs[i - 1], values[i - 1]
        for w in range(cap + 1):
            dp[i, w] = dp[i - 1, w]
            if c <= w:
                dp[i, w] = max(dp[i, w], dp[i - 1, w - c] + v)
    w, sel = cap, []
    for i in range(n, 0, -1):
        if dp[i, w] != dp[i - 1, w]:
            sel.append(i - 1)
            w -= costs[i - 1]
    chosen = items.iloc[sel]
    return chosen["cer_id"].tolist(), float(chosen["cost"].sum()), \
        float(chosen["npv"].sum())


def run_cer_portfolio(cers):
    pos = cers[cers["npv"] > 0]
    budget = float(pos["cost"].sum() * 0.5)  # tight year: only half the
    # positive-NPV asks can be funded, so the choice of optimizer matters
    sel, cost, value = knapsack(cers, budget)
    # greedy by NPV per dollar
    g = pos.assign(ratio=pos["npv"] / pos["cost"]).sort_values("ratio",
                                                               ascending=False)
    g_cost, g_value, g_sel = 0.0, 0.0, []
    for _, r in g.iterrows():
        if g_cost + r["cost"] <= budget:
            g_cost += r["cost"]
            g_value += r["npv"]
            g_sel.append(r["cer_id"])
    # approve-all-positive-NPV (infeasible: breaches the budget)
    all_cost, all_value = float(pos["cost"].sum()), float(pos["npv"].sum())
    res = pd.DataFrame([
        {"strategy": "knapsack optimizer", "selected": len(sel),
         "cost": round(cost, 2), "npv": round(value, 2),
         "utilization_pct": round(cost / budget * 100, 1)},
        {"strategy": "greedy by NPV/$", "selected": len(g_sel),
         "cost": round(g_cost, 2), "npv": round(g_value, 2),
         "utilization_pct": round(g_cost / budget * 100, 1)},
        {"strategy": "approve all NPV>0", "selected": len(pos),
         "cost": round(all_cost, 2), "npv": round(all_value, 2),
         "utilization_pct": round(all_cost / budget * 100, 1)},
    ])
    return res, budget, sel


# ---------------- Module 3 ----------------

CLOSE_FEATURES = ["log_amount", "days_after_close", "is_weekend", "is_round",
                  "reversal", "amount_z", "dup_count", "poster_count"]


def rule_flags(df):
    """Rule baseline: flag if late (>=6 days), weekend, round, or duplicate."""
    return ((df["days_after_close"] >= 6) | (df["is_weekend"] == 1)
            | (df["is_round"] == 1) | (df["dup_count"] > 1)).astype(int)


def run_close(feats):
    train = feats[feats["close_month"] < "2025-10"]
    test = feats[feats["close_month"] >= "2025-10"]
    Xtr, ytr = train[CLOSE_FEATURES], train["is_anomaly"]
    Xte, yte = test[CLOSE_FEATURES], test["is_anomaly"]
    out = {}
    p_rule = rule_flags(test)
    out["rule"] = {"precision": precision_score(yte, p_rule, zero_division=0),
                   "recall": recall_score(yte, p_rule, zero_division=0),
                   "f1": f1_score(yte, p_rule, zero_division=0),
                   "accuracy": accuracy_score(yte, p_rule)}
    rf = RandomForestClassifier(n_estimators=300, random_state=42,
                                min_samples_leaf=2, n_jobs=-1,
                                class_weight="balanced")
    rf.fit(Xtr, ytr)
    p_rf = rf.predict(Xte)
    proba = rf.predict_proba(Xte)[:, 1]
    out["rf"] = {"precision": precision_score(yte, p_rf, zero_division=0),
                 "recall": recall_score(yte, p_rf, zero_division=0),
                 "f1": f1_score(yte, p_rf, zero_division=0),
                 "accuracy": accuracy_score(yte, p_rf),
                 "roc_auc": roc_auc_score(yte, proba)}
    imp = pd.Series(rf.feature_importances_, index=CLOSE_FEATURES) \
        .sort_values(ascending=False)
    scored = test[["entry_id", "close_month", "account", "amount",
                   "is_anomaly"]].copy()
    scored["rule_flag"] = p_rule.values
    scored["rf_flag"] = p_rf
    scored["rf_prob"] = np.round(proba, 3)
    return out, imp, scored


def close_health(feats):
    """Health score 0–100 per close month: penalize anomaly rate, late
    postings, and round-number postings."""
    rows = []
    for month, g in feats.groupby("close_month"):
        anomaly_rate = g["is_anomaly"].mean()
        late_rate = (g["days_after_close"] >= 6).mean()
        round_rate = g["is_round"].mean()
        score = 100 - 100 * (0.5 * anomaly_rate + 0.3 * late_rate
                             + 0.2 * round_rate)
        rows.append({"close_month": month, "entries": len(g),
                     "anomaly_rate": round(anomaly_rate, 3),
                     "late_rate": round(late_rate, 3),
                     "health_score": round(max(score, 0.0), 1)})
    return pd.DataFrame(rows)
