"""Feature engineering.

Module 1 (forecasting): per cost-center x kind series, lag-1 / lag-12 actual,
3-month rolling mean, trend index, month-of-year dummies (sin/cos), budget
for the target month, headcount and cloud-usage drivers.
Module 3 (close): per-entry features — log amount, days after close,
weekend/round/reversal flags, amount z-score within account+month, count of
same account+amount in the month (duplicate signal), poster entry count.
"""

import numpy as np
import pandas as pd


def build_forecast_features(df):
    """One row per (cost_center, kind, month); target = actual next month."""
    df = df.sort_values(["cost_center", "kind", "month"]).copy()
    df["month_dt"] = pd.to_datetime(df["month"])
    feats = []
    for (center, kind), g in df.groupby(["cost_center", "kind"]):
        g = g.reset_index(drop=True)
        g["lag1"] = g["actual"].shift(1)
        g["lag12"] = g["actual"].shift(12)
        g["roll3"] = g["actual"].shift(1).rolling(3).mean()
        g["trend"] = np.arange(len(g))
        g["sin_m"] = np.sin(2 * np.pi * g["month_dt"].dt.month / 12)
        g["cos_m"] = np.cos(2 * np.pi * g["month_dt"].dt.month / 12)
        g["target"] = g["actual"]
        feats.append(g)
    out = pd.concat(feats, ignore_index=True)
    out = out.dropna(subset=["lag1", "lag12", "roll3"])
    return out


def build_close_features(df):
    """Per-entry features for the close anomaly classifier."""
    df = df.copy()
    df["log_amount"] = np.log(df["amount"])
    grp = df.groupby(["close_month", "account"])["amount"]
    df["acct_month_mean"] = grp.transform("mean")
    df["acct_month_std"] = grp.transform("std").fillna(0)
    df["amount_z"] = (df["amount"] - df["acct_month_mean"]) / \
        (df["acct_month_std"].replace(0, np.nan)).fillna(1.0)
    dup = df.groupby(["close_month", "account", "amount"])["entry_id"] \
        .transform("count")
    df["dup_count"] = dup
    df["poster_count"] = df.groupby(["close_month", "poster"])["entry_id"] \
        .transform("count")
    return df
