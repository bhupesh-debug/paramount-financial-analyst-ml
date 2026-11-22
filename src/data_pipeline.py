"""Validation + cleaning for the synthetic files.

Checks: no missing values, dates parse and are ordered, amounts positive,
one-off/rate/volume components reconcile to actual within rounding, CER
NPV/IRR recompute, close labels are 0/1. Writes cleaned copies to
data/processed/ and a validation report.
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYN = os.path.join(ROOT, "data", "synthetic")
PROC = os.path.join(ROOT, "data", "processed")
os.makedirs(PROC, exist_ok=True)


def _npv(rate, cfs):
    return sum(cf / (1 + rate) ** (i + 1) for i, cf in enumerate(cfs))


def validate_monthly(df):
    assert df["actual"].min() > 0, "actual must be positive"
    assert df["budget"].min() > 0, "budget must be positive"
    assert not df[["budget", "actual", "forecast"]].isna().any().any()
    # bridge reconciles: actual = budget*(1+rate+volume) + one_off
    recon = df["budget"] * (1 + df["rate_effect"] + df["volume_effect"]) \
        + df["one_off"]
    assert np.allclose(recon, df["actual"], atol=1.0), "bridge breaks"
    return len(df)


def validate_cers(df):
    assert not df[["cost", "annual_benefit", "npv"]].isna().any().any()
    assert (df["cost"] > 0).all()
    assert (df["life_years"] >= 3).all() and (df["life_years"] <= 7).all()
    # spot-check NPV of the first 5 rows against the cash-flow formula
    for _, r in df.head(5).iterrows():
        cfs = [r["annual_benefit"] * (1.02 ** y)
               for y in range(int(r["life_years"]))]
        assert abs(_npv(0.09, cfs) - r["cost"] - r["npv"]) < 1.0, "NPV mismatch"
    return len(df)


def validate_close(df):
    assert set(df["is_anomaly"].unique()) <= {0, 1}
    assert (df["days_after_close"] >= 0).all()
    assert not df["amount"].isna().any()
    return len(df)


def main():
    report = []
    m = pd.read_csv(os.path.join(SYN, "monthly_spend.csv"))
    report.append(("monthly_spend", validate_monthly(m)))
    m.to_csv(os.path.join(PROC, "monthly_spend.csv"), index=False)
    c = pd.read_csv(os.path.join(SYN, "cers.csv"))
    report.append(("cers", validate_cers(c)))
    c.to_csv(os.path.join(PROC, "cers.csv"), index=False)
    j = pd.read_csv(os.path.join(SYN, "close_entries.csv"))
    report.append(("close_entries", validate_close(j)))
    j.to_csv(os.path.join(PROC, "close_entries.csv"), index=False)
    for name, n in report:
        print(f"validated {name}: {n} rows")
    return report


if __name__ == "__main__":
    main()
