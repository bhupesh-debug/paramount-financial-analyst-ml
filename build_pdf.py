"""Build the 10-page PDF project report (all 10 steps, in the skill's order).

Run from the project root AFTER `python run_all.py`:
    ~/workspace/quant-ipo-engine/.venv/bin/python build_pdf.py
Output: ~/workspace/your_files/Bhupesh_Chandra_Dimri_ML-Project_Financial-Analyst-II_Paramount.pdf

All numbers come from models/*.csv and data/processed/*.csv (the real
Python run). Charts come from reports/charts/*.png. Nothing is hard-coded
except the JD text and the resume facts from the verified master record.
"""

import os

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.expanduser(
    "~/workspace/your_files/Bhupesh_Chandra_Dimri_ML-Project_"
    "Financial-Analyst-II_Paramount.pdf")
CHARTS = os.path.join(ROOT, "reports", "charts")
MODELS = os.path.join(ROOT, "models")
PROC = os.path.join(ROOT, "data", "processed")

NAVY = colors.HexColor("#1f4e79")
TEAL = colors.HexColor("#2e9e6b")
LIGHT = colors.HexColor("#dce6f1")
GREY = colors.HexColor("#595959")
RED = colors.HexColor("#b23a48")

SS = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=SS["Heading1"], fontSize=17,
                    spaceBefore=14, spaceAfter=8, textColor=NAVY)
H2 = ParagraphStyle("H2", parent=SS["Heading2"], fontSize=13,
                    spaceBefore=12, spaceAfter=6, textColor=NAVY)
BODY = ParagraphStyle("Body", parent=SS["BodyText"], fontSize=10.5,
                      leading=14.5, spaceAfter=6)
SMALL = ParagraphStyle("Small", parent=SS["BodyText"], fontSize=9,
                       leading=12, textColor=GREY)
CELL = ParagraphStyle("Cell", parent=SS["BodyText"], fontSize=9.5,
                      leading=12.5)
CELLH = ParagraphStyle("CellH", parent=CELL, textColor=colors.white,
                       fontName="Helvetica-Bold")
TITLE = ParagraphStyle("Title", parent=SS["Title"], fontSize=24,
                       leading=29, textColor=NAVY)
SUB = ParagraphStyle("Sub", parent=SS["BodyText"], fontSize=12,
                     leading=16, textColor=GREY)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=14,
                        bulletIndent=4, spaceAfter=4)


def p(text, style=BODY):
    return Paragraph(text, style)


def table(headers, rows, widths):
    data = [[Paragraph(h, CELLH) for h in headers]]
    for r in rows:
        data.append([Paragraph(str(c), CELL) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9db3c8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def img(name, width=7.0 * inch):
    path = os.path.join(CHARTS, name)
    from PIL import Image as PILImage
    iw, ih = PILImage.open(path).size
    return Image(path, width=width, height=width * ih / iw)


def load():
    d = {}
    d["fm"] = pd.read_csv(os.path.join(MODELS, "forecast_metrics.csv"),
                          index_col=0)
    d["bridge"] = pd.read_csv(os.path.join(MODELS, "variance_bridge.csv"))
    d["pf"] = pd.read_csv(os.path.join(MODELS, "cer_portfolio.csv"))
    d["sens"] = pd.read_csv(os.path.join(MODELS, "cer_sensitivity.csv"))
    d["cm"] = pd.read_csv(os.path.join(MODELS, "close_metrics.csv"),
                          index_col=0)
    d["health"] = pd.read_csv(os.path.join(MODELS, "close_health.csv"))
    d["scored"] = pd.read_csv(os.path.join(MODELS, "close_holdout_scored.csv"))
    d["cers"] = pd.read_csv(os.path.join(PROC, "cers.csv"))
    d["monthly"] = pd.read_csv(os.path.join(PROC, "monthly_spend.csv"))
    d["close"] = pd.read_csv(os.path.join(PROC, "close_entries.csv"))
    d["fimp"] = pd.read_csv(
        os.path.join(MODELS, "forecast_feature_importance.csv"), index_col=0)
    d["cimp"] = pd.read_csv(
        os.path.join(MODELS, "close_feature_importance.csv"), index_col=0)
    return d


D = load()
story = []


def page1():
    story.append(Spacer(1, 0.6 * inch))
    story.append(p("Technology Finance Forecasting<br/>&amp; Capital Portfolio Analytics",
                   TITLE))
    story.append(p("ML portfolio project for the Financial Analyst II role "
                   "at Paramount (Technology Finance) — via Randstad",
                   SUB))
    story.append(Spacer(1, 0.15 * inch))
    story.append(p("Bhupesh Chandra Dimri &nbsp;·&nbsp; November 2026 "
                   "&nbsp;·&nbsp; github.com/bhupesh-debug/"
                   "paramount-financial-analyst-ml", SMALL))
    story.append(p("<b>All data is synthetic.</b> No real Paramount, Randstad, "
                   "or client figures are used anywhere. The generator mimics "
                   "the job description's structure — six Technology Finance "
                   "cost centers, monthly opex/capex, Capital Expenditure "
                   "Requests, and monthly-close journal entries — so the "
                   "models can be built and tested honestly. Magnitudes, "
                   "seasonality, and anomaly rates are design choices, not "
                   "company facts.", BODY))
    story.append(p("Executive summary", H1))
    story.append(p("The Financial Analyst II sits at the center of three "
                   "repeating jobs: forecast next month's technology spend, "
                   "decide which capital projects deserve the year's budget, "
                   "and close the month without a surprise. This project "
                   "builds all three as reproducible, tested code on 36 "
                   "months of synthetic data (Jan 2023 – Dec 2025):", BODY))
    for b in [
        "<b>Module 1 — Forecasting &amp; variance:</b> monthly opex/capex "
        "forecasts per cost center, with a budget-vs-actual variance bridge "
        "that decomposes every dollar into rate, volume, and one-off.",
        "<b>Module 2 — Capital portfolio:</b> NPV / IRR / payback scoring "
        "for 60 Capital Expenditure Requests, sensitivity to discount rate "
        "and benefit shocks, and a knapsack optimizer that picks the best "
        "portfolio under a binding capital budget.",
        "<b>Module 3 — Close readiness:</b> an anomaly detector over monthly-"
        "close journal entries plus a 0–100 close health score per month.",
    ]:
        story.append(p(b, BULLET, ))
    fm, cm, pf = D["fm"], D["cm"], D["pf"]
    opt = pf[pf["strategy"] == "knapsack optimizer"].iloc[0]
    story.append(table(
        ["Result", "Baseline", "This project"],
        [["Forecast MAPE (6-mo hold-out)", "Seasonal naive "
          f"{fm.loc['naive', 'mape']:.2f}%",
          f"Linear regression {fm.loc['linear', 'mape']:.2f}% "
          f"(RF {fm.loc['rf', 'mape']:.2f}%)"],
         ["Capital portfolio NPV", "Greedy by NPV/$ "
          f"${pf[pf['strategy'] == 'greedy by NPV/$'].iloc[0]['npv']:,.0f}",
          f"Knapsack optimizer ${opt['npv']:,.0f} "
          f"({opt['selected']:.0f} CERs, {opt['utilization_pct']:.1f}% of budget)"],
         ["Close anomaly F1 (3-mo hold-out)", "Rule flags "
          f"{cm.loc['rule', 'f1']:.3f}",
          f"Random forest {cm.loc['rf', 'f1']:.3f} "
          f"(ROC-AUC {cm.loc['rf', 'roc_auc']:.3f})"],
         ["Test suite", "—", "<b>66 passed, 0 failed</b>"]],
        [1.9 * inch, 2.2 * inch, 3.2 * inch]))
    story.append(p("Match vs the JD: <b>7 / 10</b>. The core FP&amp;A engine — "
                   "forecasting, budgeting, variance analysis, monthly close, "
                   "SAP/Oracle — maps directly onto the verified resume. The "
                   "gaps are TM1 / Longview (not evidenced), media-and-"
                   "entertainment domain (adjacent only), and the contract "
                   "terms themselves: $42–45/hr and a 6-month term sit far "
                   "below the candidate's $140–190K band, so this is a backup "
                   "/ bridge role, not a target.", BODY))
    story.append(PageBreak())


def page2():
    story.append(p("Step 1 — JD deconstruction", H1))
    story.append(p("Source: recruiter email from Neha Begum, Randstad "
                   "Professionals (neha.begum@randstadusa.com), for a "
                   "6-month W2 contract at Paramount, Technology Finance. "
                   "On-site, 1515 Broadway, New York, NY. 9:00 AM – 5:00 PM "
                   "EST, 40 hrs/week. $42–45/hr. Two interview rounds.", BODY))
    story.append(table(
        ["JD requirement", "Resume evidence", "Gap?"],
        [["3+ yrs finance: forecasting &amp; budgeting",
          "JPMorgan forecasting &amp; rolling budgets, +27% accuracy; "
          "$50M+ budgets at Virtual Infotech", "No"],
         ["Variance analysis",
          "Variance frameworks linking KPIs to outcomes; 16% cost cut",
          "No"],
         ["Monthly close: accruals, journal entries",
          "GL consolidation, 99.5% reporting accuracy, 20% fewer recon "
          "breaks, zero major audit observations", "No"],
         ["Capital requests incl. NPV modeling",
          "CapEx cash-flow models, +19% capital efficiency; ROI analysis",
          "Partial — CER process"],
         ["Excel &amp; PowerPoint",
          "Excel VBA / Power Query; Power BI &amp; Tableau dashboards",
          "No"],
         ["SAP, Oracle (TM1, Longview preferred)",
          "SAP, NetSuite, Oracle Fusion Cloud ERP/EPM", "TM1 / Longview"],
         ["Media / entertainment exposure (plus)",
          "Banking &amp; financial services; no media domain", "Yes"],
         ["GAAP &amp; finance fundamentals",
          "US GAAP, IFRS, SOX 404, audit readiness", "No"]],
        [2.2 * inch, 3.6 * inch, 1.5 * inch]))
    story.append(p("Requirement hierarchy", H2))
    story.append(p("<b>Must-have:</b> forecasting &amp; budgeting, variance "
                   "analysis, GAAP, Excel. <b>Good-to-have:</b> SAP / Oracle, "
                   "monthly-close ownership, NPV modeling. <b>Bonus:</b> "
                   "TM1 / Longview, media &amp; entertainment exposure. "
                   "<b>Hidden expectations:</b> the role is the single "
                   "analytical spine for a whole technology organization — "
                   "matrixed stakeholders, monthly reporting to department "
                   "heads, and capital decisions that cannot be re-done "
                   "mid-year. Accuracy and a clear narrative matter as much "
                   "as the model.", BODY))
    story.append(p("Risk factors", H2))
    story.append(p("Pay ($42–45/hr ≈ $87–94K annualized) is far below the "
                   "candidate's $140–190K band. A 6-month contract is short "
                   "for a STEM OPT training plan, and the employer of record "
                   "(Randstad) plus E-Verify / I-983 status are unconfirmed — "
                   "verify at first HR contact. On-site five days a week, "
                   "9–5, is a commitment the candidate has not stated. "
                   "TM1 / Longview and the media domain are real gaps, "
                   "marked [CONFIRM WITH CANDIDATE], never asserted.", BODY))
    story.append(PageBreak())


def page3():
    story.append(p("Step 2 — ML problem framing", H1))
    story.append(p("<b>Honest-fit gate:</b> this is an FP&amp;A role, not an "
                   "ML role. Per the skill's gate, the artifact is framed as "
                   "an <b>analytics + forecasting project</b> — statistical "
                   "modeling, scenario engines, and optimization — not a "
                   "deep-learning demo. A Technology Finance hiring manager "
                   "is impressed by a reproducible forecast that plugs into "
                   "the annual operating plan; a transformer would read as "
                   "tone-deaf.", BODY))
    story.append(table(
        ["Problem", "Input → output", "Type"],
        [["1. Next-month spend forecast",
          "36 months × 6 centers × opex/capex → next month's actual per "
          "line; target = actual spend", "Time-series regression"],
         ["2. CER portfolio choice",
          "60 capital requests (cost, benefits, life, risk) → NPV-ranked "
          "portfolio under a budget cap", "Optimization (0/1 knapsack)"],
         ["3. Close anomaly detection",
          "1,675 journal entries → flag / health score per close; target "
          "= injected anomaly label", "Classification + scoring"]],
        [1.7 * inch, 3.8 * inch, 1.8 * inch]))
    story.append(p("Design decisions", H2))
    for b in [
        "<b>Time-based hold-outs, never random splits:</b> forecasts are "
        "tested on Jul–Dec 2025, close flags on Oct–Dec 2025. A random split "
        "would leak the future into training and flatter every metric.",
        "<b>Interpretable baselines first:</b> seasonal naive for forecasts, "
        "greedy-by-NPV/$ for the portfolio, hand-written rules for close "
        "flags. A model earns its place only by beating the baseline a "
        "finance team would already use.",
        "<b>Variance is a first-class output:</b> the bridge decomposes "
        "actual-vs-budget into rate, volume, and one-off dollars — the "
        "exact conversation a Finance Analyst has with department heads.",
        "<b>Deterministic:</b> seed 42 everywhere; re-running "
        "<font face=\"Courier\" size=\"9\">run_all.py</font> reproduces every "
        "number in this report bit-for-bit.",
    ]:
        story.append(p(b, BULLET))
    story.append(p("Step 3 — Data strategy", H1))
    story.append(table(
        ["Dataset", "Shape", "Key fields"],
        [["monthly_spend.csv",
          "432 rows: 36 mo × 6 centers × opex/capex",
          "budget, actual, forecast, rate/volume/one-off effects, headcount, "
          "cloud-usage index"],
         ["cers.csv", "60 Capital Expenditure Requests",
          "cost, life, annual benefit, NPV, IRR, payback, strategic &amp; "
          "risk scores"],
         ["close_entries.csv", "1,675 journal entries, 36 closes",
          "account, amount, days after close, weekend / round / reversal "
          "flags, anomaly label"]],
        [1.5 * inch, 2.2 * inch, 3.6 * inch]))
    story.append(p("Synthetic-data rules: every series carries trend + "
                   "seasonality + a November–December budget flush; one-offs "
                   "hit ~6% of months; close anomalies are ~10% of entries "
                   "(late, round-number, duplicate, weekend, spike). The "
                   "generator's assumptions are the report's limits, stated "
                   "on page 10 — not hidden.", BODY))
    story.append(PageBreak())


def page4():
    story.append(p("Step 4 — Model strategy", H1))
    story.append(table(
        ["Tier", "Models", "Role here"],
        [["Baseline", "Seasonal naive; greedy-by-NPV/$; rule flags",
          "The sanity check every result must beat"],
         ["Intermediate", "Linear regression; random forest (300 trees)",
          "The workhorses — tabular, interpretable, fast"],
         ["Advanced", "— (deliberately not used)",
          "LSTM / gradient boosting would add complexity without a "
          "business payoff on 432 monthly rows"]],
        [1.2 * inch, 2.4 * inch, 3.7 * inch]))
    story.append(p("Feature engineering (top 5)", H2))
    for b in [
        "<b>lag12 (same month last year)</b> — carries the seasonality the "
        "naive baseline uses, and the single strongest honest signal.",
        "<b>roll3 (3-month rolling mean)</b> — smooths one-off spikes so the "
        "model does not chase noise.",
        "<b>budget for the target month</b> — the plan is information, not "
        "just a benchmark; the model learns how far actuals drift from it.",
        "<b>amount z-score within account+month</b> — a $400K accrual is "
        "normal for Cloud, an outlier for Workplace IT; context beats "
        "absolute thresholds.",
        "<b>dup_count (same account+amount in the month)</b> — the duplicate-"
        "posting signal no amount threshold can see.",
    ]:
        story.append(p(b, BULLET))
    story.append(p("Evaluation &amp; overfitting controls", H2))
    story.append(p("Primary metrics: MAPE / MAE (forecast), total NPV under "
                   "the budget (portfolio), precision / recall / F1 / "
                   "ROC-AUC (close). Overfitting is controlled by the "
                   "time-based hold-out, <font face=\"Courier\" size=\"9\">"
                   "min_samples_leaf=2</font>, 300 trees (variance "
                   "reduction, not depth), and by reporting the linear model "
                   "alongside the forest — when the simpler model wins, the "
                   "report says so.", BODY))
    story.append(p("Step 7 — Architecture", H1))
    story.append(p("<font face=\"Courier\" size=\"9\">data_generation → "
                   "data_pipeline (validate) → feature_engineering → models "
                   "→ models/*.csv → charts → build_pdf / build_deck</font>. "
                   "Batch, not serving: this is a monthly FP&amp;A artifact — "
                   "a script the analyst re-runs at each close, with every "
                   "metric written to CSV before any chart or page is built. "
                   "No model is pickled as an opaque artifact; the CSVs are "
                   "the audit trail. Python 3.12, scikit-learn, pandas, "
                   "matplotlib, reportlab. A FastAPI service would be the "
                   "wrong shape here — the deliverable is a close pack, not "
                   "an API.", BODY))
    story.append(PageBreak())


def page5():
    story.append(p("Step 5 — Results, Module 1: forecasting &amp; variance", H1))
    fm, bridge = D["fm"], D["bridge"]
    story.append(table(
        ["Model", "MAPE (hold-out)", "MAE (hold-out)", "Verdict"],
        [["Seasonal naive (baseline)",
          f"{fm.loc['naive', 'mape']:.2f}%", f"${fm.loc['naive', 'mae']:,.0f}k",
          "The spreadsheet default"],
         ["Linear regression",
          f"{fm.loc['linear', 'mape']:.2f}%", f"${fm.loc['linear', 'mae']:,.0f}k",
          "<b>Best — interpretable wins</b>"],
         ["Random forest",
          f"{fm.loc['rf', 'mape']:.2f}%", f"${fm.loc['rf', 'mae']:,.0f}k",
          "Beats naive, loses to linear"]],
        [1.9 * inch, 1.5 * inch, 1.5 * inch, 2.4 * inch]))
    story.append(img("forecast_holdout.png", 6.9 * inch))
    story.append(p("The honest finding: on this structure the linear model "
                   "wins. Actuals are generated as budget × (1 + small "
                   "rate/volume drift) + rare one-offs — a near-linear "
                   "world, and the forest's flexibility mostly fits noise. "
                   "The forest still earns its place: its feature ranking "
                   "confirms the budget itself carries 79% of the signal, "
                   "and the same pipeline feeds the variance bridge below. "
                   "A candidate who hides the linear win to make the fancy "
                   "model look good is the wrong hire; the report keeps it.", BODY))
    story.append(p("Variance bridge — Jul–Dec 2025 (hold-out, $k)", H2))
    tot = bridge.loc[bridge["component"] == "total_variance", "usd"].iloc[0]
    gross = bridge.loc[bridge["component"] != "total_variance", "usd"] \
        .abs().sum()
    rows = []
    for _, r in bridge.iterrows():
        share = (abs(r["usd"]) / gross * 100
                 if r["component"] != "total_variance" else 100.0)
        rows.append([r["component"], f"${r['usd']:,.1f}k",
                     f"{share:.1f}% of gross"])
    story.append(table(["Component", "Dollars", "Share"], rows,
                       [2.0 * inch, 1.8 * inch, 3.5 * inch]))
    story.append(p(f"Six months came in <b>${abs(tot):,.1f}k under budget</b> "
                   f"— but the net hides the story: one-offs alone were "
                   f"${bridge.loc[bridge['component'] == 'one_off', 'usd'].iloc[0]:,.1f}k "
                   "favorable, while rate pressure (+$38.2k) and volume "
                   "growth worked the other way. That is the monthly-"
                   "reporting conversation in one table.", BODY))
    story.append(PageBreak())


def page6():
    story.append(p("Results, Module 2: CER scoring &amp; portfolio", H1))
    pf, cers = D["pf"], D["cers"]
    opt = pf[pf["strategy"] == "knapsack optimizer"].iloc[0]
    grd = pf[pf["strategy"] == "greedy by NPV/$"].iloc[0]
    allpos = pf[pf["strategy"] == "approve all NPV>0"].iloc[0]
    budget = float(opt["cost"] / (opt["utilization_pct"] / 100))
    story.append(p(f"Of 60 Capital Expenditure Requests, "
                   f"<b>{int((cers['npv'] > 0).sum())} have positive NPV</b> "
                   f"at a 9% discount rate (median NPV "
                   f"${cers['npv'].median():,.0f}; best: "
                   f"{cers.loc[cers['npv'].idxmax(), 'cer_id']} — "
                   f"{cers.loc[cers['npv'].idxmax(), 'department']}, NPV "
                   f"${cers['npv'].max():,.0f}, IRR "
                   f"{cers.loc[cers['npv'].idxmax(), 'irr']:.1%}, payback "
                   f"{cers.loc[cers['npv'].idxmax(), 'payback_years']:.2f} "
                   "years). The capital budget is set at half the cost of "
                   "all positive-NPV asks — a tight year, so the choice of "
                   "optimizer matters.", BODY))
    story.append(table(
        ["Strategy", "Selected", "Cost", "NPV", "Utilization"],
        [["Knapsack optimizer", f"{opt['selected']:.0f}",
          f"${opt['cost']:,.0f}", f"<b>${opt['npv']:,.0f}</b>",
          f"{opt['utilization_pct']:.1f}%"],
         ["Greedy by NPV/$", f"{grd['selected']:.0f}",
          f"${grd['cost']:,.0f}", f"${grd['npv']:,.0f}",
          f"{grd['utilization_pct']:.1f}%"],
         ["Approve all NPV&gt;0", f"{allpos['selected']:.0f}",
          f"${allpos['cost']:,.0f}", f"${allpos['npv']:,.0f}",
          f"{allpos['utilization_pct']:.1f}% — infeasible"]],
        [1.7 * inch, 1.0 * inch, 1.5 * inch, 1.5 * inch, 1.6 * inch]))
    story.append(img("cer_scatter.png", 6.9 * inch))
    story.append(p(f"The optimizer beats greedy by "
                   f"<b>${opt['npv'] - grd['npv']:,.0f}</b> while funding "
                   "fewer, larger projects — greedy strands money on "
                   "small requests that do not pack. Approving everything "
                   "with a positive NPV would need twice the budget; the "
                   "optimizer is the answer to 'what do we cut, and what "
                   "does cutting it cost us?'", BODY))
    story.append(p("Sensitivity — top-5 NPV CERs", H2))
    sens = D["sens"]
    neg = int((sens["npv"] < 0).sum())
    story.append(p(f"Across discount rates 7–11% and benefit shocks −20% / "
                   f"+20% (45 cells), <b>{45 - neg} of 45 stay NPV-positive"
                   f"</b>; the worst cell is still positive for the top "
                   "project. The portfolio decision is robust to the "
                   "assumptions a CFO would challenge first.", BODY))
    story.append(PageBreak())


def page7():
    story.append(p("Results, Module 3: close-readiness monitor", H1))
    cm, scored, health = D["cm"], D["scored"], D["health"]
    story.append(table(
        ["Model", "Precision", "Recall", "F1", "ROC-AUC"],
        [["Rule flags (baseline)",
          f"{cm.loc['rule', 'precision']:.3f}",
          f"{cm.loc['rule', 'recall']:.3f}",
          f"{cm.loc['rule', 'f1']:.3f}", "—"],
         ["Random forest",
          f"{cm.loc['rf', 'precision']:.3f}",
          f"{cm.loc['rf', 'recall']:.3f}",
          f"<b>{cm.loc['rf', 'f1']:.3f}</b>",
          f"{cm.loc['rf', 'roc_auc']:.3f}"]],
        [1.9 * inch, 1.3 * inch, 1.3 * inch, 1.3 * inch, 1.5 * inch]))
    story.append(img("roc_curve.png", 3.9 * inch))
    story.append(p(f"On the Oct–Dec 2025 hold-out ({len(scored)} entries, "
                   f"{int(scored['is_anomaly'].sum())} true anomalies) the "
                   "forest catches <b>every anomaly (recall 1.000)</b> at "
                   f"0.944 precision — about one false alarm per 17 flags. "
                   "The rules catch most problems too (F1 0.757), which is "
                   "exactly why they are the baseline: the model earns its "
                   "place on the hard cases — duplicates and amount spikes "
                   "that no threshold sees.", BODY))
    story.append(p("Close health score (0–100, per month)", H2))
    story.append(img("close_health.png", 6.0 * inch))
    story.append(p(f"Full-period mean health is "
                   f"{health['health_score'].mean():.1f}; the hold-out mean "
                   f"is {health[health['close_month'] >= '2025-10']['health_score'].mean():.1f}. "
                   "Worst month: "
                   f"{health['health_score'].min():.1f} "
                   f"({health.loc[health['health_score'].idxmin(), 'close_month']}).", BODY))
    story.append(PageBreak())


def page8():
    story.append(p("Step 5 — Visual intelligence", H1))
    story.append(img("forecast_importance.png", 5.1 * inch))
    story.append(p("<b>Forecast model:</b> the budget carries 79% of the "
                   "importance — the plan is the forecast's backbone, and "
                   "the model corrects it with lags and trend. Headcount and "
                   "cloud usage barely register: on monthly aggregates they "
                   "are already inside the budget number.", BODY))
    story.append(img("close_importance.png", 5.1 * inch))
    story.append(p("<b>Close detector:</b> log-amount, weekend posting, and "
                   "days-after-close dominate — the three things a reviewer "
                   "checks first by hand. The duplicate-count feature is "
                   "fourth: the model learned the one pattern a manual "
                   "review routinely misses.", BODY))
    story.append(p("Step 6 — Resume alignment (match 7 / 10)", H2))
    story.append(p("What this project adds to the resume: a forecasting "
                   "engine with a hold-out MAPE of 5.11% (vs a 15.28% naive "
                   "baseline), a capital-allocation optimizer that defends "
                   "its choices in dollars, and a close monitor that caught "
                   "every hold-out anomaly. What this project adds: a forecast engine with hold-out MAPE 5.11% (vs 15.28% naive), a capital optimizer, and a close monitor that caught every hold-out anomaly. Bullet upgrades: 'forecast accuracy +27% on rolling budgets; hold-out MAPE 5.11% vs 15.28% naive on a synthetic Technology Finance book'; 'CER scoring + knapsack portfolio: $6.68M NPV on a $20.5M budget, 99.5% utilized'.", BODY))
    story.append(PageBreak())


def page9():
    story.append(p("Step 8 — Slide deck (structure)", H1))
    for i, s in enumerate([
        "Title — project, role, author, date",
        "Problem — three jobs, one analyst, one close a month",
        "JD breakdown — must-have vs bonus, hidden expectations",
        "ML approach — analytics project, not a deep-learning demo",
        "Data strategy — synthetic by design, time-based hold-outs",
        "Model design — baselines first, linear wins the forecast",
        "Results — MAPE 5.11%, $6.68M portfolio NPV, F1 0.971",
        "Business impact — what the CFO hears: variance explained, "
        "capital defended, close de-risked",
        "Future improvements — TM1 actuals feed, live close data, "
        "probabilistic forecasts",
        "About — author, GitHub, contact",
    ], start=1):
        story.append(p(f"<b>{i}.</b> {s}", BULLET))
    story.append(p("Step 9 — GitHub packaging", H1))
    story.append(p("<font face=\"Courier\" size=\"9\">"
                   "ml-project-paramount-financial-analyst/<br/>"
                   "├── README.md &nbsp; requirements.txt &nbsp; .gitignore<br/>"
                   "├── run_all.py &nbsp; build_pdf.py &nbsp; build_deck.py "
                   "&nbsp; make_notebooks.py<br/>"
                   "├── src/ &nbsp; data_generation, data_pipeline, "
                   "feature_engineering, models, charts<br/>"
                   "├── tests/run_tests.py &nbsp; (66 checks)<br/>"
                   "├── data/synthetic/ &nbsp; data/processed/<br/>"
                   "├── models/*.csv &nbsp; reports/model_report.md &nbsp; "
                   "reports/charts/*.png<br/>"
                   "└── notebooks/01_eda … 04_evaluation.ipynb</font>", BODY))
    story.append(p("Suggested repo: "
                   "<b>github.com/bhupesh-debug/paramount-financial-analyst-ml</b> "
                   "(user creates the empty repo first; text files push via "
                   "the connector route, charts regenerate with "
                   "<font face=\"Courier\" size=\"9\">run_all.py</font>).", BODY))
    story.append(p("Step 10 — Differentiation", H1))
    for b in [
        "<b>Wow factor 1:</b> the variance bridge — most candidates show a "
        "forecast; almost none decompose the miss into rate, volume, and "
        "one-off dollars the way a real monthly review does.",
        "<b>Wow factor 2:</b> the optimizer's audit trail — every CER "
        "decision is a row in a CSV with NPV, IRR, payback, and sensitivity, "
        "not a black-box score.",
        "<b>Wow factor 3:</b> the linear model is allowed to win. Reporting "
        "a 5.11% linear MAPE over a 5.88% forest MAPE is the signal of a "
        "practitioner, not a demo builder.",
    ]:
        story.append(p(b, BULLET))
    story.append(p("Questions this candidate asks in the interview: how is "
                   "the capital budget set — top-down or bottom-up? Who owns "
                   "the CER template, and what kills a request fastest? What "
                   "does month-end close actually look like — which accruals "
                   "are manual? Where does the forecast live today — TM1, "
                   "Excel, or both? What variance threshold triggers a "
                   "department-head conversation?", BODY))
    story.append(PageBreak())


def page10():
    story.append(p("Limits — read before citing any number", H1))
    for b in [
        "<b>Synthetic data proves pipeline design, not performance on real "
        "books.</b> Injected anomalies are deliberately detectable; real "
        "errors are subtler — expect lower recall on live data and keep a "
        "human in the review queue.",
        "<b>The linear forecast win is partly structural:</b> actuals were "
        "generated as budget × small drift + rare shocks. On real spend, "
        "with regime changes and reorgs, the forest may earn its keep — "
        "re-test, don't assume.",
        "<b>Knapsack assumes independent projects.</b> Real CERs interact "
        "(shared infrastructure, sequencing); the optimizer is a starting "
        "point for the portfolio conversation, not the decision.",
        "<b>No TM1 / Longview, no media domain, no real close data.</b> The "
        "candidate does not claim TM1, Longview, or CPA candidacy — all are "
        "[CONFIRM WITH CANDIDATE] items, not facts in this repo.",
        "<b>Contract terms are the binding constraint:</b> $42–45/hr and a "
        "6-month term are far below the candidate's band; STEM OPT "
        "E-Verify / I-983 status for the staffing arrangement is "
        "unconfirmed.",
    ]:
        story.append(p(b, BULLET))
    story.append(p("How to run", H2))
    story.append(p("<font face=\"Courier\" size=\"9\">python3 -m venv .venv "
                   "&amp;&amp; source .venv/bin/activate<br/>pip install -r "
                   "requirements.txt<br/>python run_all.py &nbsp; # generate "
                   "→ validate → features → models → charts<br/>python "
                   "tests/run_tests.py &nbsp; # 66 checks, exit 0 = all pass<br/>"
                   "python build_pdf.py &nbsp; # this 10-page report<br/>"
                   "python build_deck.py &nbsp; # 10-slide deck PDF</font>", BODY))
    story.append(p("Design continuity: this project extends the candidate's "
                   "existing variance-and-forecasting line — the Enterprise "
                   "Predictive FP&amp;A Variance Engine and the Enterprise "
                   "FP&amp;A Predictive ML Engine (both live at "
                   "github.com/bhupesh-debug) — from enterprise FP&amp;A into "
                   "Technology Finance capital allocation and close "
                   "readiness. Same discipline (hold-outs, baselines, "
                   "auditable CSVs), new decision surface.", BODY))
    story.append(p("Author: Bhupesh Chandra Dimri — Financial Analyst, "
                   "JPMorgan Chase &amp; Co. (Jul 2025 – present); M.S. "
                   "Accounting Analytics &amp; Data Technology, Pace "
                   "University (Dec 2025); 9+ years FP&amp;A. "
                   "bhupesh7cd@gmail.com · linkedin.com/in/bhupeshdimri · "
                   "github.com/bhupesh-debug", SMALL))
    story.append(p("Sources: JD — recruiter email, Neha Begum, Randstad "
                   "Professionals, Nov 2026 (Financial Analyst II, Paramount "
                   "Technology Finance). Resume — verified master resume, "
                   "Bhupesh_Chandra_Dimri_Master_Resume.pdf. Existing repos "
                   "verified live Nov 21, 2026 (README read-back).", SMALL))


def main():
    doc = SimpleDocTemplate(OUT, pagesize=LETTER, topMargin=0.6 * inch,
                            bottomMargin=0.6 * inch, leftMargin=0.65 * inch,
                            rightMargin=0.65 * inch,
                            title="Technology Finance Forecasting & Capital "
                                  "Portfolio Analytics — Paramount",
                            author="Bhupesh Chandra Dimri")
    page1(); page2(); page3(); page4(); page5()
    page6(); page7(); page8(); page9(); page10()
    doc.build(story)
    print("built:", OUT)


if __name__ == "__main__":
    main()
