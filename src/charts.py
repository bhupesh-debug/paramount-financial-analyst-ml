"""Charts for the report and deck (matplotlib, deterministic)."""

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTS = os.path.join(ROOT, "reports", "charts")
os.makedirs(CHARTS, exist_ok=True)

NAVY = "#1f4e79"
TEAL = "#2e9e6b"
GREY = "#595959"
RED = "#b23a48"
LIGHT = "#dce6f1"


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, name), dpi=150)
    plt.close(fig)


def forecast_chart(test_out):
    """Hold-out actual vs RF / linear / naive for total spend by month."""
    agg = test_out.groupby("month")[["target", "naive", "linear", "rf"]].sum()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(agg.index, agg["target"] / 1e3, "o-", color=NAVY, lw=2,
            label="Actual")
    ax.plot(agg.index, agg["rf"] / 1e3, "s--", color=TEAL, lw=1.8,
            label="Random forest")
    ax.plot(agg.index, agg["linear"] / 1e3, "^-.", color=GREY, lw=1.4,
            label="Linear regression")
    ax.plot(agg.index, agg["naive"] / 1e3, "x:", color=RED, lw=1.4,
            label="Seasonal naive")
    ax.set_ylabel("$k per month (all centers)")
    ax.set_title("Hold-out forecast vs actual — total monthly spend, Jul–Dec 2025")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=4, frameon=False)
    ax.grid(alpha=0.25)
    _save(fig, "forecast_holdout.png")


def bridge_chart(bridge):
    b = bridge[bridge["component"] != "total_variance"]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    colors = [RED if v < 0 else TEAL for v in b["usd"]]
    ax.bar(b["component"], b["usd"] / 1e3, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("$k, Jul–Dec 2025")
    ax.set_title("Variance bridge: actual − budget, decomposed")
    for i, v in enumerate(b["usd"] / 1e3):
        ax.text(i, v + (8 if v >= 0 else -14), f"{v:,.0f}", ha="center",
                fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "variance_bridge.png")


def cer_scatter(cers, selected):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    cers = cers.copy()
    cers["selected"] = cers["cer_id"].isin(selected)
    ax.scatter(cers[~cers["selected"]]["cost"] / 1e3,
               cers[~cers["selected"]]["npv"] / 1e3, c=GREY, s=28,
               label="Not selected", alpha=0.7)
    ax.scatter(cers[cers["selected"]]["cost"] / 1e3,
               cers[cers["selected"]]["npv"] / 1e3, c=TEAL, s=55,
               label="Knapsack-selected", edgecolor="black", linewidths=0.5)
    ax.axhline(0, color=RED, lw=1, ls="--", label="NPV = 0")
    ax.set_xlabel("Project cost ($k)")
    ax.set_ylabel("NPV ($k) at 9% discount")
    ax.set_title("CER portfolio: NPV vs cost — optimizer picks")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    _save(fig, "cer_scatter.png")


def feature_importance(imp, name, title):
    top = imp.head(8)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(top.index[::-1], top.values[::-1], color=NAVY)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.25)
    _save(fig, name)


def health_chart(health):
    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.plot(health["close_month"], health["health_score"], "o-", color=NAVY,
            lw=2)
    ax.axhline(80, color=TEAL, lw=1, ls="--", label="Healthy (>= 80)")
    ax.fill_between(health["close_month"], 80, 100, color=TEAL, alpha=0.08)
    ax.set_ylabel("Close health score (0–100)")
    ax.set_title("Monthly-close health score, Jan 2023 – Dec 2025")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(55, 102)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    _save(fig, "close_health.png")


def roc_chart(y_true, proba):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, color=NAVY, lw=2, label="Random forest")
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Close anomaly detector — ROC, Oct–Dec 2025 hold-out")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    _save(fig, "roc_curve.png")


def main():
    models = os.path.join(ROOT, "models")
    proc = os.path.join(ROOT, "data", "processed")
    test_out = pd.read_csv(os.path.join(models, "forecast_holdout.csv"))
    forecast_chart(test_out)
    bridge = pd.read_csv(os.path.join(models, "variance_bridge.csv"))
    bridge_chart(bridge)
    cers = pd.read_csv(os.path.join(proc, "cers.csv"))
    portfolio = pd.read_csv(os.path.join(models, "cer_portfolio.csv"))
    # selected ids are stored in the knapsack result file
    sel = pd.read_csv(os.path.join(models, "cer_selected.csv"))
    cer_scatter(cers, sel["cer_id"].tolist())
    f_imp = pd.read_csv(os.path.join(models, "forecast_feature_importance.csv"),
                        index_col=0).iloc[:, 0]
    feature_importance(f_imp, "forecast_importance.png",
                       "Forecast model — feature importance (RF)")
    c_imp = pd.read_csv(os.path.join(models, "close_feature_importance.csv"),
                        index_col=0).iloc[:, 0]
    feature_importance(c_imp, "close_importance.png",
                       "Close anomaly detector — feature importance (RF)")
    health = pd.read_csv(os.path.join(models, "close_health.csv"))
    health_chart(health)
    scored = pd.read_csv(os.path.join(models, "close_holdout_scored.csv"))
    roc_chart(scored["is_anomaly"], scored["rf_prob"])
    print("charts written:", sorted(os.listdir(CHARTS)))


if __name__ == "__main__":
    main()
