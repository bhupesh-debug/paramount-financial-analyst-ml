"""Synthetic data generation — Technology Finance, Paramount (fictional).

ALL DATA IS SYNTHETIC. No real Paramount, Randstad, or client figures are
used anywhere. The generator mimics the JD's structure: six Technology
Finance cost centers with monthly opex/capex, an annual budget, Capital
Expenditure Requests, and monthly-close journal entries. Magnitudes,
seasonality, and anomaly rates are design choices, not company facts.

Deterministic: seed 42 everywhere. Horizon Jan 2023 – Dec 2025 (36 months).
Hold-out is Jul–Dec 2025 (6 months) for the forecasting module and the last
3 close cycles (Oct–Dec 2025) for the close module.
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYN = os.path.join(ROOT, "data", "synthetic")
os.makedirs(SYN, exist_ok=True)

CENTERS = [
    ("Cloud Infrastructure", 420, 310, 0.018),
    ("Data Platforms", 260, 190, 0.012),
    ("Cybersecurity", 180, 120, 0.010),
    ("Workplace IT", 150, 60, 0.006),
    ("Broadcast Engineering", 300, 240, 0.008),
    ("Enterprise Applications", 210, 150, 0.011),
]
# (center, base monthly opex $k, base monthly capex $k, monthly growth)

MONTHS = pd.period_range("2023-01", "2025-12", freq="M")
DRIVERS = {"headcount": 0.35, "cloud_usage": 0.45, "project_load": 0.20}


def gen_monthly(rng):
    """Monthly opex/capex per cost center with an explicit variance bridge.

    actual = budget * (1 + rate_effect + volume_effect) + one_off
    so the budget-vs-actual variance decomposes exactly into rate, volume,
    and one-off components. The prior-month forecast is a lagged, slightly
    stale estimate of actual (what a spreadsheet forecast looks like).
    """
    rows = []
    for center, base_opex, base_capex, growth in CENTERS:
        # annual budget: base * 12, uplifted 4% per year, set each January
        for i, per in enumerate(MONTHS):
            t = i
            year = per.year - 2023
            season = 1.0 + 0.09 * np.sin(2 * np.pi * (per.month - 1) / 12)
            year_end = 1.12 if per.month in (11, 12) else 1.0  # budget flush
            trend = (1 + growth) ** t
            for kind, base in (("opex", base_opex), ("capex", base_capex)):
                budget = base * trend * season * year_end * (1.04 ** year)
                # rate effect: unit-price drift (cloud rates, vendor pricing)
                rate = rng.normal(0.0, 0.025)
                # volume effect: usage / headcount driven
                volume = rng.normal(0.0, 0.035)
                # one-off: rare, large (project pull-forward, true-up)
                one_off = 0.0
                if rng.random() < 0.06:
                    one_off = budget * rng.choice([-1, 1]) * rng.uniform(0.08, 0.22)
                actual = budget * (1 + rate + volume) + one_off
                # prior forecast: last month's actual grown by trend, no shock
                forecast = budget * (1 + rng.normal(0.0, 0.02))
                rows.append({
                    "month": str(per), "month_start": str(per.start_time.date()),
                    "cost_center": center, "kind": kind,
                    "budget": round(budget, 2), "actual": round(max(actual, 1.0), 2),
                    "forecast": round(forecast, 2),
                    "rate_effect": round(rate, 6), "volume_effect": round(volume, 6),
                    "one_off": round(one_off, 2),
                    "headcount": int(120 + 40 * np.sin(2 * np.pi * t / 24)
                                     + rng.integers(-4, 5)),
                    "cloud_usage_idx": round(100 * (1 + 0.02) ** t
                                             * (1 + rng.normal(0, 0.03)), 2),
                })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SYN, "monthly_spend.csv"), index=False)
    return df


def npv(rate, cfs):
    return sum(cf / (1 + rate) ** (i + 1) for i, cf in enumerate(cfs))


def irr(cfs, lo=-0.9, hi=10.0):
    """Bisection IRR for cash flows starting with a negative outflow."""
    def f(r):
        return sum(cf / (1 + r) ** i for i, cf in enumerate(cfs))
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def gen_cers(rng):
    """Capital Expenditure Requests with full NPV/IRR/payback economics."""
    rows = []
    depts = [c[0] for c in CENTERS]
    for i in range(60):
        dept = depts[i % len(depts)]
        cost = float(rng.choice([120, 250, 400, 650, 900, 1400, 2200]) * 1000
                     * rng.uniform(0.85, 1.15))
        cost = round(cost, -3)  # nearest $1k; continuous-ish costs make the
        # knapsack packing problem real (round multiples pack too neatly)
        life = int(rng.integers(3, 8))  # years
        # annual benefit: cost recovered over the life plus a margin —
        # total undiscounted benefits ≈ cost * (1 + margin), so NPV>0
        # when the margin clears the 9% discount drag (about 60% of CERs)
        margin = rng.normal(0.35, 0.20)
        benefit = cost * (1 + margin) / life * rng.uniform(0.9, 1.1)
        cfs = [-cost] + [benefit * (1.02 ** y) for y in range(life)]
        rate = 0.09
        value = npv(rate, cfs[1:]) - cost
        r = irr(cfs)
        cum, payback = 0.0, float(life)
        for y, cf in enumerate(cfs[1:], start=1):
            cum += cf
            if cum >= cost:
                payback = (y - 1) + (cost - (cum - cf)) / cf
                break
        rows.append({
            "cer_id": f"CER-{1000 + i}", "department": dept,
            "cost": round(cost, 2), "life_years": life,
            "annual_benefit": round(benefit, 2),
            "npv": round(value, 2), "irr": round(r, 4) if r == r else None,
            "payback_years": round(payback, 2),
            "strategic_score": int(rng.integers(1, 6)),  # 1-5
            "risk_score": int(rng.integers(1, 6)),       # 1 low - 5 high
            "submitted_month": str(MONTHS[int(rng.integers(0, 30))]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SYN, "cers.csv"), index=False)
    return df


def gen_close(rng):
    """Monthly-close journal entries with injected anomalies.

    Labels: 1 = anomaly (late posting, round-number, duplicate, weekend,
    out-of-pattern amount). Normal entries post within 3 business days of
    month-end, during business hours, with non-round amounts.
    """
    rows = []
    accounts = ["Accrued Expenses", "Prepaid Expenses", "Accrued Revenue",
                "Depreciation", "Payroll Accrual", "Vendor Payables",
                "Intercompany", "Cloud Usage Accrual"]
    entry_id = 0
    for per in MONTHS:
        n = int(rng.integers(38, 55))
        month_rows = []
        for _ in range(n):
            entry_id += 1
            anomaly = rng.random() < 0.10
            kind = None
            acct = accounts[int(rng.integers(0, len(accounts)))]
            amount = float(abs(rng.normal(85, 60)) * 1000)
            days_after = int(rng.integers(0, 4))
            weekend = 0
            round_num = 0
            if anomaly:
                kind = rng.choice(["late", "round", "duplicate", "weekend",
                                   "spike"])
                if kind == "late":
                    days_after = int(rng.integers(6, 12))
                elif kind == "round":
                    amount = float(rng.choice([50000, 100000, 250000, 500000]))
                    round_num = 1
                elif kind == "weekend":
                    weekend = 1
                elif kind == "spike":
                    amount *= rng.uniform(6, 12)
                elif kind == "duplicate" and month_rows:
                    # copy an earlier entry's account+amount in this month
                    src = month_rows[int(rng.integers(0, len(month_rows)))]
                    acct, amount = src["account"], src["amount"]
            else:
                if rng.random() < 0.04:
                    round_num = 1  # benign round numbers exist too
            rec = {
                "entry_id": f"JE-{entry_id:05d}",
                "close_month": str(per), "account": acct,
                "amount": round(amount, 2),
                "days_after_close": days_after,
                "is_weekend": weekend, "is_round": round_num,
                "reversal": int(rng.random() < 0.08),
                "poster": f"user_{int(rng.integers(1, 9))}",
                "is_anomaly": int(anomaly),
                "anomaly_kind": kind if anomaly else "",
            }
            month_rows.append(rec)
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SYN, "close_entries.csv"), index=False)
    return df


def main():
    rng = np.random.default_rng(42)
    m = gen_monthly(rng)
    c = gen_cers(rng)
    j = gen_close(rng)
    print(f"monthly_spend: {len(m)} rows, {m['cost_center'].nunique()} centers")
    print(f"cers: {len(c)} rows, NPV>0: {(c['npv'] > 0).sum()}")
    print(f"close_entries: {len(j)} rows, anomalies: {j['is_anomaly'].sum()}")


if __name__ == "__main__":
    main()
