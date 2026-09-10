
import streamlit as st
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression


# -----------------------------
# V20 — AI CFO Executive Brief
# -----------------------------
def _v20_num(value, default=0.0):
    try:
        x = pd.to_numeric(value, errors="coerce")
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

def _v20_fmt_money(x):
    x = _v20_num(x)
    ax = abs(x)
    sign = "-" if x < 0 else ""
    if ax >= 1_000_000_000:
        return f"{sign}${ax/1_000_000_000:.2f}B"
    if ax >= 1_000_000:
        return f"{sign}${ax/1_000_000:.2f}M"
    if ax >= 1_000:
        return f"{sign}${ax/1_000:.1f}K"
    return f"{sign}${ax:,.0f}"

def _v20_find_col(df, names):
    lower = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if str(n).strip().lower() in lower:
            return lower[str(n).strip().lower()]
    return None

def _v20_build_brief(df, actions_df=None):
    revenue_col = _v20_find_col(df, ["Revenue USD", "Revenue"])
    budget_col = _v20_find_col(df, ["Budget USD", "Budget"])
    actual_col = _v20_find_col(df, ["Actual USD", "Actual"])
    gp_col = _v20_find_col(df, ["Gross Profit", "Gross Profit USD"])
    ebitda_col = _v20_find_col(df, ["EBITDA"])
    pat_col = _v20_find_col(df, ["PAT"])
    cash_col = _v20_find_col(df, ["Closing Cash", "Closing Cash USD"])
    variance_col = _v20_find_col(df, ["Variance USD", "Variance"])

    revenue = _v20_num(df[revenue_col].sum()) if revenue_col else 0
    budget = _v20_num(df[budget_col].sum()) if budget_col else 0
    actual = _v20_num(df[actual_col].sum()) if actual_col else 0
    variance = _v20_num(df[variance_col].sum()) if variance_col else actual - budget

    # Management metrics in V3 are monthly values repeated across transaction rows.
    def latest_metric(col):
        if not col:
            return 0.0
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        return float(s.iloc[-1]) if len(s) else 0.0

    # Prefer the latest monthly management-layer value when present.
    gross_profit = latest_metric(gp_col)
    ebitda = latest_metric(ebitda_col)
    pat = latest_metric(pat_col)
    cash = latest_metric(cash_col)

    if gp_col is None:
        gross_profit = revenue - actual
    if ebitda_col is None:
        ebitda = gross_profit
    if pat_col is None:
        pat = ebitda * 0.60
    if cash_col is None:
        cash = max(0.0, revenue - actual)

    ebitda_margin = ebitda / revenue if revenue else 0
    pat_margin = pat / revenue if revenue else 0

    # Unfavorable cost exposure.
    unfavorable = max(0.0, variance)

    # Risk/anomaly evidence.
    anomaly_col = _v20_find_col(df, ["Anomaly Flag", "anomaly_flag"])
    anomaly_count = int(pd.to_numeric(df[anomaly_col], errors="coerce").fillna(0).gt(0).sum()) if anomaly_col else 0

    # Top variance driver.
    group_col = _v20_find_col(df, ["Cost Category", "Account / Cost Category", "account"])
    top_driver = "Not available"
    top_driver_var = 0.0
    if group_col and variance_col:
        tmp = df.groupby(group_col, dropna=False)[variance_col].sum().sort_values(ascending=False)
        if len(tmp):
            top_driver = str(tmp.index[0])
            top_driver_var = _v20_num(tmp.iloc[0])

    # Regional hotspot.
    region_col = _v20_find_col(df, ["Region", "Business Unit", "business_unit"])
    top_region = "Not available"
    top_region_var = 0.0
    if region_col and variance_col:
        tmp = df.groupby(region_col, dropna=False)[variance_col].sum().sort_values(ascending=False)
        if len(tmp):
            top_region = str(tmp.index[0])
            top_region_var = _v20_num(tmp.iloc[0])

    open_actions = 0
    high_actions = 0
    if actions_df is not None and len(actions_df):
        status_col = _v20_find_col(actions_df, ["Status"])
        priority_col = _v20_find_col(actions_df, ["Priority"])
        if status_col:
            open_actions = int((~actions_df[status_col].astype(str).str.lower().eq("resolved")).sum())
        if priority_col:
            high_actions = int(actions_df[priority_col].astype(str).str.lower().isin(["high", "critical"]).sum())

    if variance > 0:
        budget_sentence = f"Costs are {_v20_fmt_money(variance)} above the current budget."
    elif variance < 0:
        budget_sentence = f"Costs are {_v20_fmt_money(abs(variance))} below the current budget."
    else:
        budget_sentence = "Costs are broadly in line with budget."

    risk_level = "Elevated" if anomaly_count >= 25 or unfavorable >= max(1_000_000, budget * 0.01) else "Moderate"
    if anomaly_count == 0 and unfavorable == 0:
        risk_level = "Low"

    summary = (
        f"Revenue is {_v20_fmt_money(revenue)} with Gross Profit of {_v20_fmt_money(gross_profit)} "
        f"and EBITDA of {_v20_fmt_money(ebitda)} ({ebitda_margin:.1%} margin). "
        f"{budget_sentence} The largest cost-variance driver is {top_driver}, while "
        f"{top_region} is the leading regional variance hotspot. "
        f"Liquidity is currently {_v20_fmt_money(cash)}, with {anomaly_count} flagged anomaly transactions. "
        f"Overall management risk is assessed as {risk_level.lower()}."
    )

    recommendations = []
    if unfavorable > 0:
        recommendations.append(f"Investigate {top_driver} variance and validate the highest-impact transactions.")
    if top_region_var > 0:
        recommendations.append(f"Review {top_region} performance and identify the operational driver behind the variance.")
    if anomaly_count > 0:
        recommendations.append("Prioritize anomaly transactions with material financial exposure for controller review.")
    if open_actions > 0:
        recommendations.append(f"Close or escalate the {open_actions} outstanding management action(s) based on financial impact.")
    if not recommendations:
        recommendations.append("Continue monitoring performance against budget and refresh the CFO brief at the next reporting cycle.")

    return {
        "revenue": revenue, "gross_profit": gross_profit, "ebitda": ebitda,
        "pat": pat, "cash": cash, "ebitda_margin": ebitda_margin,
        "pat_margin": pat_margin, "budget": budget, "actual": actual,
        "variance": variance, "unfavorable": unfavorable,
        "anomaly_count": anomaly_count, "top_driver": top_driver,
        "top_driver_var": top_driver_var, "top_region": top_region,
        "top_region_var": top_region_var, "risk_level": risk_level,
        "open_actions": open_actions, "high_actions": high_actions,
        "summary": summary, "recommendations": recommendations,
    }

def _v20_render_cfo_brief(df, actions_df=None):
    st.subheader("🧠 AI CFO Executive Brief")
    st.caption("One-click management synthesis across performance, budget, liquidity, risk, forecast signals and actions.")

    brief = _v20_build_brief(df, actions_df)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Revenue", _v20_fmt_money(brief["revenue"]))
    c2.metric("Gross Profit", _v20_fmt_money(brief["gross_profit"]))
    c3.metric("EBITDA", _v20_fmt_money(brief["ebitda"]))
    c4.metric("PAT", _v20_fmt_money(brief["pat"]))
    c5.metric("Cash", _v20_fmt_money(brief["cash"]))
    c6.metric("Risk", brief["risk_level"])

    st.markdown("### Executive Summary")
    st.info(brief["summary"])

    left, right = st.columns(2)
    with left:
        st.markdown("### Financial Performance")
        perf = pd.DataFrame([
            ["Revenue", _v20_fmt_money(brief["revenue"])],
            ["Gross Profit", _v20_fmt_money(brief["gross_profit"])],
            ["EBITDA", _v20_fmt_money(brief["ebitda"])],
            ["EBITDA Margin", f'{brief["ebitda_margin"]:.1%}'],
            ["PAT", _v20_fmt_money(brief["pat"])],
            ["PAT Margin", f'{brief["pat_margin"]:.1%}'],
        ], columns=["Metric", "Value"])
        st.dataframe(perf, use_container_width=True, hide_index=True)

    with right:
        st.markdown("### Budget & Risk")
        risk_tbl = pd.DataFrame([
            ["Budget", _v20_fmt_money(brief["budget"])],
            ["Actual Cost", _v20_fmt_money(brief["actual"])],
            ["Variance", _v20_fmt_money(brief["variance"])],
            ["Unfavorable Exposure", _v20_fmt_money(brief["unfavorable"])],
            ["Top Cost Driver", brief["top_driver"]],
            ["Top Regional Hotspot", brief["top_region"]],
            ["Anomaly Transactions", str(brief["anomaly_count"])],
        ], columns=["Signal", "Value"])
        st.dataframe(risk_tbl, use_container_width=True, hide_index=True)

    st.markdown("### 💧 Cash & Liquidity")
    st.success(f"Current closing cash: **{_v20_fmt_money(brief['cash'])}**. Use the Cash Flow Center for detailed runway and cash-generation analysis.")

    st.markdown("### 🎯 Management Priorities")
    for i, rec in enumerate(brief["recommendations"], 1):
        st.markdown(f"**{i}.** {rec}")

    st.markdown("### 📌 Action Pipeline")
    a1, a2, a3 = st.columns(3)
    a1.metric("Open Actions", brief["open_actions"])
    a2.metric("High/Critical Actions", brief["high_actions"])
    a3.metric("Risk Level", brief["risk_level"])

    # Export a standalone HTML management brief.
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>FinSight AI CFO Executive Brief</title>
<style>body{{font-family:Arial,sans-serif;max-width:1000px;margin:40px auto;padding:0 24px;color:#172033}}
h1{{margin-bottom:4px}}h2{{margin-top:28px}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ddd;padding:9px;text-align:left}}.summary{{padding:16px;border:1px solid #ddd;border-radius:8px}}
</style></head><body>
<h1>FinSight AI — CFO Executive Brief</h1>
<p>Generated from the current ERP reporting dataset.</p>
<h2>Executive Summary</h2><div class="summary">{brief["summary"]}</div>
<h2>Financial Performance</h2>
<table><tr><th>Metric</th><th>Value</th></tr>
<tr><td>Revenue</td><td>{_v20_fmt_money(brief["revenue"])}</td></tr>
<tr><td>Gross Profit</td><td>{_v20_fmt_money(brief["gross_profit"])}</td></tr>
<tr><td>EBITDA</td><td>{_v20_fmt_money(brief["ebitda"])}</td></tr>
<tr><td>EBITDA Margin</td><td>{brief["ebitda_margin"]:.1%}</td></tr>
<tr><td>PAT</td><td>{_v20_fmt_money(brief["pat"])}</td></tr>
<tr><td>PAT Margin</td><td>{brief["pat_margin"]:.1%}</td></tr></table>
<h2>Budget & Risk</h2>
<table><tr><th>Signal</th><th>Value</th></tr>
<tr><td>Budget</td><td>{_v20_fmt_money(brief["budget"])}</td></tr>
<tr><td>Actual Cost</td><td>{_v20_fmt_money(brief["actual"])}</td></tr>
<tr><td>Variance</td><td>{_v20_fmt_money(brief["variance"])}</td></tr>
<tr><td>Unfavorable Exposure</td><td>{_v20_fmt_money(brief["unfavorable"])}</td></tr>
<tr><td>Top Cost Driver</td><td>{brief["top_driver"]}</td></tr>
<tr><td>Top Regional Hotspot</td><td>{brief["top_region"]}</td></tr>
<tr><td>Anomaly Transactions</td><td>{brief["anomaly_count"]}</td></tr></table>
<h2>Management Priorities</h2><ol>{"".join(f"<li>{r}</li>" for r in brief["recommendations"])}</ol>
<h2>Action Pipeline</h2><p>Open actions: {brief["open_actions"]} | High/Critical: {brief["high_actions"]} | Risk: {brief["risk_level"]}</p>
</body></html>"""

    st.download_button(
        "📄 Download AI CFO Executive Brief",
        data=html.encode("utf-8"),
        file_name="finsight_ai_cfo_executive_brief.html",
        mime="text/html",
        use_container_width=True,
    )


st.set_page_config(
    page_title="FinSight AI — CFO Command Center",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# THEME — FIN SIGHT AI / DARK CFO COMMAND CENTER
# ============================================================
st.markdown("""
<style>
/* ---------- Global canvas ---------- */
.stApp {
    background: radial-gradient(circle at 72% -10%, #142442 0%, #0A1020 38%, #070B14 100%);
    color: #E8EEF8;
}
html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.block-container {
    padding-top: 1.05rem;
    padding-bottom: 3.5rem;
    max-width: 1540px;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080E19 0%, #0A1220 55%, #080D17 100%);
    border-right: 1px solid #1B2940;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #F4F7FF;
    letter-spacing: -0.02em;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #9DAEC4;
    border-radius: 9px;
    padding: 4px 7px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #111C30;
    color: #F3F7FF;
}
[data-testid="stSidebar"] hr {
    border-color: #1B2940;
}

/* ---------- Command-center hero ---------- */
.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, rgba(18,35,64,.98), rgba(9,17,31,.98) 62%, rgba(17,14,42,.98));
    border: 1px solid #243754;
    border-radius: 20px;
    padding: 24px 28px;
    margin-bottom: 18px;
    box-shadow: 0 18px 50px rgba(0,0,0,.20), inset 0 1px 0 rgba(255,255,255,.03);
}
.hero:after {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    right: -100px;
    top: -130px;
    border-radius: 50%;
    background: rgba(83, 96, 255, .18);
    filter: blur(28px);
}
.hero-title {
    position: relative;
    z-index: 1;
    font-size: 29px;
    font-weight: 800;
    color: #F6F8FF;
    margin-bottom: 5px;
    letter-spacing: -0.035em;
}
.hero-subtitle {
    position: relative;
    z-index: 1;
    font-size: 13px;
    color: #91A4BF;
}
.pill {
    position: relative;
    z-index: 1;
    display: inline-block;
    border: 1px solid #30486B;
    border-radius: 999px;
    padding: 5px 10px;
    margin-right: 6px;
    margin-top: 12px;
    color: #9FB9FF;
    background: rgba(33,53,88,.55);
    font-size: 11px;
}

/* ---------- KPI / metric cards ---------- */
.kpi {
    background: linear-gradient(180deg, #101A2C, #0B1322);
    border: 1px solid #202F49;
    border-radius: 15px;
    padding: 15px 14px;
    min-height: 118px;
    overflow: hidden;
    box-shadow: 0 10px 30px rgba(0,0,0,.13);
}
.kpi-label {
    color: #8396B0;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .09em;
    text-transform: uppercase;
}
.kpi-value {
    color: #F5F8FF;
    font-size: 21px;
    font-weight: 800;
    margin-top: 9px;
    white-space: nowrap;
}
.kpi-note {
    color: #58D9B4;
    font-size: 11px;
    margin-top: 7px;
}
[data-testid="stMetricValue"] {
    font-size: 1.34rem !important;
    line-height: 1.15 !important;
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: clip !important;
    min-width: 0 !important;
    letter-spacing: -0.025em !important;
    color: #F4F7FF !important;
}
[data-testid="stMetricLabel"] {
    color: #8B9CB4 !important;
    font-size: .72rem !important;
    font-weight: 650 !important;
}
[data-testid="stMetricDelta"] {
    font-size: .72rem !important;
}
[data-testid="stMetric"] {
    background: linear-gradient(180deg, #101A2C, #0B1322);
    border: 1px solid #202F49;
    padding: 12px 13px;
    border-radius: 14px;
    min-width: 0 !important;
    box-shadow: 0 10px 28px rgba(0,0,0,.10);
}

/* ---------- Panels / alerts ---------- */
.section-card {
    background: #0C1423;
    border: 1px solid #1C2A42;
    border-radius: 15px;
    padding: 18px 20px;
}
.insight, .warning, .danger {
    padding: 13px 16px;
    border-radius: 10px;
    margin: 7px 0;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.025);
}
.insight {
    background: rgba(17,53,51,.58);
    border: 1px solid #1C5E57;
    border-left: 4px solid #42D7B3;
}
.warning {
    background: rgba(60,46,17,.60);
    border: 1px solid #66511F;
    border-left: 4px solid #E7B64B;
}
.danger {
    background: rgba(64,23,31,.60);
    border: 1px solid #6D2C39;
    border-left: 4px solid #F06B7B;
}
.small { color: #8294AD; font-size: 11px; }

/* ---------- Streamlit containers ---------- */
div[data-testid="stExpander"] {
    background: #0B1322;
    border: 1px solid #1D2B43;
    border-radius: 12px;
}
div[data-testid="stTabs"] button {
    color: #8497B0;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #F2F6FF;
}
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-testid="stTextInput"] > div > div {
    background: #0D1626 !important;
    border-color: #253653 !important;
}
input, textarea {
    color: #EAF0FA !important;
}
button[kind="primary"] {
    background: linear-gradient(90deg, #4B63F5, #6956E8) !important;
    border: 1px solid #7182FF !important;
    color: white !important;
}
button[kind="secondary"] {
    background: #101A2C !important;
    border: 1px solid #293A57 !important;
    color: #DCE5F3 !important;
}

/* ---------- Tables ---------- */
[data-testid="stDataFrame"] {
    border: 1px solid #1D2B43;
    border-radius: 11px;
    overflow: hidden;
}
[data-testid="stDataFrame"] [role="columnheader"] {
    background: #111C30 !important;
    color: #AFC0D7 !important;
}

/* ---------- Headings / dividers ---------- */
h1, h2, h3, h4 {
    color: #F0F4FB !important;
    letter-spacing: -0.02em;
}
hr { border-color: #1A2940 !important; }

/* ---------- Hide default Streamlit decoration ---------- */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ---------- Scrollbar ---------- */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #070C15; }
::-webkit-scrollbar-thumb { background: #263754; border-radius: 999px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================
def html_block(html):
    """Render HTML safely without Markdown treating indentation as code."""
    st.markdown(html.replace("\n", " ").strip(), unsafe_allow_html=True)


def money_usd(x):
    x = float(x or 0)
    ax = abs(x)
    sign = "-" if x < 0 else ""
    ax = abs(x)
    if ax >= 1_000_000_000:
        return f"{sign}${ax/1_000_000_000:,.2f}B"
    if ax >= 1_000_000:
        return f"{sign}${ax/1_000_000:,.2f}M"
    if ax >= 1_000:
        return f"{sign}${ax/1_000:,.1f}K"
    return f"{sign}${ax:,.0f}"


def money_local(x, currency):
    symbols = {"INR": "₹", "SGD": "S$", "EUR": "€", "USD": "$"}
    return f"{symbols.get(currency, currency + ' ')}{float(x or 0):,.0f}"


def pct(x):
    return f"{float(x or 0):,.1f}%"


def safe_num(series):
    return pd.to_numeric(series, errors="coerce").fillna(0)


# ============================================================
# DATA LOADING
# ============================================================
def find_repo_csv():
    # Use pathlib locally so this function is self-contained.
    import pathlib

    candidates = [
        pathlib.Path("synthetic_erp_financials.csv"),
        pathlib.Path("synthetic_erp_financials_global_v3.csv"),
        pathlib.Path("synthetic_erp_financials_global_v2.csv"),
        pathlib.Path("data/synthetic_erp_financials_global_v3.csv"),
        pathlib.Path("data/synthetic_erp_financials.csv"),
        pathlib.Path("data/synthetic_erp_financials_global_v2.csv"),
    ]

    for csv_path in candidates:
        if csv_path.exists():
            return csv_path

    return None


def load_csv(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    path = find_repo_csv()
    if path is not None:
        return pd.read_csv(path)

    return pd.DataFrame()


def normalize_data(raw):
    if raw.empty:
        return raw

    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # New global ERP schema -> canonical dashboard schema
    aliases = {
        "month": "Date",
        "date": "Date",
        "business_unit": "Business Unit",
        "Business Unit": "Business Unit",
        "Region": "Region",
        "region": "Region",
        "department": "Department",
        "Department": "Department",
        "cost_category": "Cost Category",
        "Cost Category": "Cost Category",
        "account": "Account / Cost Category",
        "Account / Cost Category": "Account / Cost Category",
        "Budget USD": "Budget USD",
        "Actual USD": "Actual USD",
        "Revenue USD": "Revenue USD",
        "Variance USD": "Variance USD",
        "Profit USD": "Profit USD",
        "budget": "Budget USD",
        "actual": "Actual USD",
        "revenue": "Revenue USD",
        "variance": "Variance USD",
        "profit": "Profit USD",
    }

    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    # Date: never replace a usable month/date with today's timestamp
    if "Date" not in df.columns:
        if "Fiscal Period" in df.columns:
            df["Date"] = pd.to_datetime(df["Fiscal Period"], errors="coerce")
        else:
            df["Date"] = pd.NaT
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Preserve the real hierarchy
    if "Region" not in df.columns:
        if "Business Unit" in df.columns:
            df["Region"] = df["Business Unit"].astype(str)
        else:
            df["Region"] = "Unknown"

    if "Business Unit" not in df.columns:
        df["Business Unit"] = df["Region"]

    if "Department" not in df.columns:
        df["Department"] = "Unmapped"

    if "Cost Category" not in df.columns:
        if "Account / Cost Category" in df.columns:
            df["Cost Category"] = df["Account / Cost Category"]
        else:
            df["Cost Category"] = "Unmapped"

    if "Account / Cost Category" not in df.columns:
        df["Account / Cost Category"] = df["Cost Category"]

    if "Country" not in df.columns:
        df["Country"] = "Unspecified"

    if "Legal Entity" not in df.columns:
        df["Legal Entity"] = "Unspecified"

    if "Currency" not in df.columns:
        df["Currency"] = "USD"

    if "FX Rate to USD" not in df.columns:
        df["FX Rate to USD"] = 1.0

    # Prefer explicit USD fields.
    for c in ["Revenue USD", "Budget USD", "Actual USD", "Variance USD", "Profit USD"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = safe_num(df[c])

    # If only local fields exist, convert using FX.
    if "Revenue Local" in df.columns:
        missing = df["Revenue USD"].eq(0)
        df.loc[missing, "Revenue USD"] = safe_num(df.loc[missing, "Revenue Local"]) * safe_num(df.loc[missing, "FX Rate to USD"])
    if "Budget Local" in df.columns:
        missing = df["Budget USD"].eq(0)
        df.loc[missing, "Budget USD"] = safe_num(df.loc[missing, "Budget Local"]) * safe_num(df.loc[missing, "FX Rate to USD"])
    if "Actual Local" in df.columns:
        missing = df["Actual USD"].eq(0)
        df.loc[missing, "Actual USD"] = safe_num(df.loc[missing, "Actual Local"]) * safe_num(df.loc[missing, "FX Rate to USD"])

    if "Variance USD" not in df.columns or df["Variance USD"].eq(0).all():
        df["Variance USD"] = df["Actual USD"] - df["Budget USD"]

    if "Profit USD" not in df.columns or df["Profit USD"].eq(0).all():
        df["Profit USD"] = df["Revenue USD"] - df["Actual USD"]

    df["Variance %"] = np.where(
        df["Budget USD"].abs() > 0,
        df["Variance USD"] / df["Budget USD"].abs() * 100,
        0,
    )

    df["Margin %"] = np.where(
        df["Revenue USD"].abs() > 0,
        df["Profit USD"] / df["Revenue USD"].abs() * 100,
        0,
    )

    if "Anomaly Flag" not in df.columns:
        df["Anomaly Flag"] = 0
    df["Anomaly Flag"] = pd.to_numeric(df["Anomaly Flag"], errors="coerce").fillna(0).astype(int)

    if "Anomaly" not in df.columns:
        df["Anomaly"] = np.where(df["Anomaly Flag"] == 1, "Anomaly", "Normal")

    if "Transaction ID" not in df.columns:
        if "transaction_id" in df.columns:
            df["Transaction ID"] = df["transaction_id"].astype(str)
        else:
            df["Transaction ID"] = [f"TXN-{i+1:05d}" for i in range(len(df))]

    # Backward-compatible canonical display fields
    df["Revenue"] = df["Revenue USD"]
    df["Budget"] = df["Budget USD"]
    df["Actual"] = df["Actual USD"]
    df["Variance"] = df["Variance USD"]
    df["Profit"] = df["Profit USD"]

    return df


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 📊 FinSight AI")
    st.caption("Agentic FP&A & ERP Intelligence")
    st.divider()

    uploaded = st.file_uploader(
        "Upload ERP CSV",
        type=["csv"],
        help="Upload a transaction-level ERP export using the FinSight global schema.",
    )

    st.divider()

    modules = [
        "Executive Dashboard",
        "AI CFO",
        "Autonomous AI CFO", "AI CFO Executive Brief",
        "Financial Performance",
        "Cash Flow Center",
        "Budget vs Actuals",
        "Forecasting & Planning",
        "What-if Scenarios",
        "Risk & Anomaly Detection",
        "Cost Intelligence",
        "AI CFO Action Center",
        "Reports Library",
        "Upload Data",
        "ERP Connections",
        "Data Mapping",
        "Data Quality",
        "Alerts & Notifications",
        "Ratio Analysis",
        "Settings",
        "Audit Logs",
    ]

    page = st.radio("Command Center", modules, index=0)

    market_uploaded = None
    if page == "Ratio Analysis":
        st.divider()
        market_uploaded = st.file_uploader(
            "Upload Market CSV (optional)",
            type=["csv"],
            key="market_csv",
            help="Price history with Date, Asset Price and Benchmark Price columns for Alpha, Beta and technical analytics.",
        )
        demo_market_path = Path("finsight_demo_market_data_v24.csv")
        if demo_market_path.exists():
            with open(demo_market_path, "rb") as _f:
                st.download_button(
                    "⬇ Download Demo Market CSV",
                    data=_f.read(),
                    file_name="finsight_demo_market_data_v24.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    st.divider()
    currency_view = st.selectbox(
        "Reporting Currency",
        ["USD Consolidated", "Local Currency"],
        index=0,
    )

    st.caption("USD is the group consolidation base. Local currencies remain available by entity.")
    st.caption("Prototype data may contain modeled cash-flow fields.")


# ============================================================
# LOAD / FILTER DATA
# ============================================================
raw = load_csv(uploaded)
df = normalize_data(raw)

if df.empty:
    html_block("""
    <div class="hero">
      <div class="hero-title">FinSight AI — CFO Command Center</div>
      <div class="hero-subtitle">Agentic FP&A • ERP Analytics • Forecasting • Risk • Cash & Liquidity • Ratios • Management Actions</div>
      <span class="pill">ERP Intelligence</span>
      <span class="pill">AI CFO Prototype</span>
      <span class="pill">USD Consolidation</span>
    </div>
    """)
    st.warning("No ERP dataset detected. Upload a CSV from the sidebar or add synthetic_erp_financials.csv to the repository.")
    st.stop()


# Global filters
with st.sidebar:
    st.markdown("### Filters")

    regions = sorted(df["Region"].dropna().astype(str).unique())
    region_filter = st.multiselect("Region", regions, default=regions)

    countries = sorted(df["Country"].dropna().astype(str).unique())
    country_filter = st.multiselect("Country", countries, default=countries)

    departments = sorted(df["Department"].dropna().astype(str).unique())
    dept_filter = st.multiselect("Department", departments, default=departments)

    date_min = df["Date"].min()
    date_max = df["Date"].max()

    if pd.notna(date_min) and pd.notna(date_max):
        date_range = st.date_input(
            "Period",
            value=(date_min.date(), date_max.date()),
        )
    else:
        date_range = None

view = df.copy()

if region_filter:
    view = view[view["Region"].astype(str).isin(region_filter)]

if country_filter:
    view = view[view["Country"].astype(str).isin(country_filter)]

if dept_filter:
    view = view[view["Department"].astype(str).isin(dept_filter)]

if date_range and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    view = view[(view["Date"] >= start) & (view["Date"] <= end + pd.Timedelta(days=1))]

if view.empty:
    st.warning("No transactions match the selected filters.")
    st.stop()



# ============================================================
# AI CFO ACTION WORKFLOW — PERSISTENT SQLITE V19
# ============================================================
DB_PATH = Path(__file__).with_name("finsight_actions.db")


def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS management_actions (
            action_id TEXT PRIMARY KEY,
            created TEXT NOT NULL,
            priority TEXT NOT NULL,
            area TEXT NOT NULL,
            issue TEXT NOT NULL,
            financial_impact TEXT,
            recommendation TEXT NOT NULL,
            owner TEXT NOT NULL,
            due_date TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Open'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_name TEXT NOT NULL,
            action TEXT NOT NULL,
            module TEXT NOT NULL,
            status TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def load_cfo_actions():
    conn = get_db()
    rows = conn.execute("SELECT action_id, created, priority, area, issue, financial_impact, recommendation, owner, due_date, status FROM management_actions ORDER BY rowid DESC").fetchall()
    conn.close()
    cols = ["Action ID", "Created", "Priority", "Area", "Issue", "Financial Impact", "Recommendation", "Owner", "Due Date", "Status"]
    return [dict(zip(cols, row)) for row in rows]


def load_cfo_audit():
    conn = get_db()
    rows = conn.execute("SELECT timestamp, user_name, action, module, status FROM audit_events ORDER BY id DESC LIMIT 200").fetchall()
    conn.close()
    cols = ["Timestamp", "User", "Action", "Module", "Status"]
    return [dict(zip(cols, row)) for row in rows]


def log_cfo_event(action, module="AI CFO", status="Success"):
    conn = get_db()
    conn.execute(
        "INSERT INTO audit_events(timestamp, user_name, action, module, status) VALUES (?, ?, ?, ?, ?)",
        (pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "Demo CFO", action, module, status),
    )
    conn.commit()
    conn.close()


def create_cfo_action(priority, area, issue, impact, recommendation, owner):
    conn = get_db()
    existing = conn.execute(
        "SELECT action_id FROM management_actions WHERE issue = ? AND status != 'Resolved' LIMIT 1", (issue,)
    ).fetchone()
    if existing:
        conn.close()
        return False
    now = pd.Timestamp.now()
    action_id = f"ACT-{now.strftime('%Y%m%d%H%M%S')}-{int(now.microsecond/1000):03d}"
    created = now.strftime("%Y-%m-%d %H:%M")
    due_date = (now + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    conn.execute(
        "INSERT INTO management_actions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (action_id, created, priority, area, issue, impact, recommendation, owner, due_date, "Open"),
    )
    conn.commit()
    conn.close()
    log_cfo_event(f"Created management action {action_id}: {area}")
    return True


def update_cfo_action_status(action_id, new_status):
    conn = get_db()
    cur = conn.execute("UPDATE management_actions SET status = ? WHERE action_id = ?", (new_status, action_id))
    conn.commit()
    conn.close()
    if cur.rowcount:
        log_cfo_event(f"Updated {action_id} to {new_status}", module="AI CFO Action Center")
        return True
    return False


def set_cfo_action_due_date(action_id, due_date):
    conn = get_db()
    conn.execute("UPDATE management_actions SET due_date = ? WHERE action_id = ?", (str(due_date), action_id))
    conn.commit()
    conn.close()


def action_status_counts():
    conn = get_db()
    rows = conn.execute("SELECT status, COUNT(*) FROM management_actions GROUP BY status").fetchall()
    conn.close()
    counts = {status: int(n) for status, n in rows}
    return counts.get("Open", 0), counts.get("In Progress", 0), counts.get("Resolved", 0)


# ============================================================
# HEADER
# ============================================================
html_block("""
<div class="hero">
  <div class="hero-title">FinSight AI — CFO Command Center</div>
  <div class="hero-subtitle">Agentic FP&A • ERP Analytics • Forecasting • Risk • Cash & Liquidity • Management Actions</div>
  <span class="pill">USD Consolidation</span>
  <span class="pill">AI CFO Prototype</span>
  <span class="pill">Global ERP Hierarchy</span>
</div>
""")


# ============================================================
# COMMON METRICS
# ============================================================
revenue = view["Revenue USD"].sum()
actual = view["Actual USD"].sum()
budget = view["Budget USD"].sum()
variance = actual - budget
profit = view["Profit USD"].sum()
margin = profit / revenue * 100 if revenue else 0
variance_pct = variance / budget * 100 if budget else 0


def chart_layout(fig, height=370):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="#0A1020",
        plot_bgcolor="#0A1020",
        margin=dict(l=24, r=24, t=82, b=48),
        title=dict(
            x=0.02,
            xanchor="left",
            y=0.98,
            yanchor="top",
            font=dict(size=16, color="#EAF2FF"),
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.90,
            x=0.02,
            xanchor="left",
            font=dict(size=11, color="#9FB0C7"),
            bgcolor="rgba(0,0,0,0)",
        ),
        font=dict(color="#B8C6D9"),
        hoverlabel=dict(
            bgcolor="#111B2E",
            bordercolor="#334563",
            font=dict(color="#F4F7FB"),
        ),
        xaxis=dict(gridcolor="#18253A", zerolinecolor="#263650"),
        yaxis=dict(gridcolor="#18253A", zerolinecolor="#263650"),
    )
    return fig


# ============================================================
# EXECUTIVE DASHBOARD — CFO-GRADE V7
# ============================================================

# ============================================================
# V22 — RATIO ANALYSIS
# ============================================================
def _v22_num(v, default=0.0):
    try:
        x = pd.to_numeric(v, errors="coerce")
        return default if pd.isna(x) else float(x)
    except Exception:
        return default

def _v22_col(df, names):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if str(n).strip().lower() in lookup:
            return lookup[str(n).strip().lower()]
    return None

def _v22_latest(df, col):
    if not col:
        return 0.0
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(s.iloc[-1]) if len(s) else 0.0

def _v22_sum(df, names):
    c = _v22_col(df, names)
    return _v22_num(df[c].sum()) if c else 0.0

def _v22_ratio_pct(a, b):
    return (a / b * 100.0) if b else 0.0

def _v22_ratio_analysis(view):
    st.header("📐 Ratio Analysis")
    st.caption(
        "CFO financial health, liquidity, leverage and working-capital analytics. "
        "Market and technical analytics are separated because they require price and benchmark data."
    )

    revenue = _v22_sum(view, ["Revenue USD", "Revenue"])
    actual = _v22_sum(view, ["Actual USD", "Actual"])
    gp = _v22_latest(view, _v22_col(view, ["Gross Profit"]))
    ebitda = _v22_latest(view, _v22_col(view, ["EBITDA"]))
    pat = _v22_latest(view, _v22_col(view, ["PAT"]))
    ebit = _v22_latest(view, _v22_col(view, ["EBIT"]))
    interest = _v22_latest(view, _v22_col(view, ["Interest / Finance Cost", "Interest"]))
    assets = _v22_latest(view, _v22_col(view, ["Total Assets", "Assets"]))
    equity = _v22_latest(view, _v22_col(view, ["Total Equity", "Equity"]))
    debt = _v22_latest(view, _v22_col(view, ["Total Debt", "Debt"]))
    current_assets = _v22_latest(view, _v22_col(view, ["Current Assets"]))
    current_liabilities = _v22_latest(view, _v22_col(view, ["Current Liabilities"]))
    inventory = _v22_latest(view, _v22_col(view, ["Inventory"]))
    cash = _v22_latest(view, _v22_col(view, ["Closing Cash", "Cash Balance"]))
    receivables = _v22_latest(view, _v22_col(view, ["Receivables", "Accounts Receivable"]))
    payables = _v22_latest(view, _v22_col(view, ["Payables", "Accounts Payable"]))

    # V3 provides management P&L but not a full balance sheet. Keep unsupported ratios explicit.
    gross_margin = _v22_ratio_pct(gp, revenue)
    ebitda_margin = _v22_ratio_pct(ebitda, revenue)
    ebit_margin = _v22_ratio_pct(ebit, revenue)
    pat_margin = _v22_ratio_pct(pat, revenue)
    opex_pct = _v22_ratio_pct(max(0.0, actual - (revenue - gp)), revenue) if revenue and gp else 0.0
    cogs_pct = _v22_ratio_pct(max(0.0, revenue - gp), revenue) if revenue and gp else 0.0

    current_ratio = current_assets / current_liabilities if current_liabilities else None
    quick_ratio = (current_assets - inventory) / current_liabilities if current_liabilities else None
    debt_equity = debt / equity if equity else None
    interest_coverage = ebit / interest if interest else None
    roa = pat / assets * 100 if assets else None
    roe = pat / equity * 100 if equity else None

    # Working-capital days use latest month-end balances against
    # annualized latest-month flows.
    date_col = _v22_col(view, ["Date", "date"])
    latest_rows = view
    if date_col:
        _dates = pd.to_datetime(view[date_col], errors="coerce")
        if _dates.notna().any():
            _latest_period = _dates.max().to_period("M")
            latest_rows = view[_dates.dt.to_period("M") == _latest_period]

    latest_month_revenue = _v22_sum(latest_rows, ["Revenue USD", "Revenue"])
    latest_month_cogs = _v22_latest(latest_rows, _v22_col(latest_rows, ["COGS"]))
    annualized_revenue = latest_month_revenue * 12
    annualized_cogs = latest_month_cogs * 12

    dso = receivables / annualized_revenue * 365 if receivables and annualized_revenue else None
    dpo = payables / annualized_cogs * 365 if payables and annualized_cogs else None
    ccc = None if dso is None or dpo is None else dso - dpo

    tabs = st.tabs([
        "Financial Health",
        "Liquidity & Leverage",
        "Working Capital",
        "Market Analytics",
        "Technical Analytics",
    ])

    with tabs[0]:
        st.subheader("Financial Health")
        cols = st.columns(4)
        metrics = [
            ("Gross Margin", f"{gross_margin:.1f}%"),
            ("EBITDA Margin", f"{ebitda_margin:.1f}%"),
            ("EBIT Margin", f"{ebit_margin:.1f}%"),
            ("PAT Margin", f"{pat_margin:.1f}%"),
            ("ROA", "N/A" if roa is None else f"{roa:.1f}%"),
            ("ROE", "N/A" if roe is None else f"{roe:.1f}%"),
            ("COGS / Revenue", f"{cogs_pct:.1f}%"),
            ("Opex / Revenue", f"{opex_pct:.1f}%"),
        ]
        for i, (label, value) in enumerate(metrics):
            cols[i % 4].metric(label, value)

        st.info(
            "ROA, ROE and other balance-sheet ratios show N/A when the uploaded ERP dataset "
            "does not contain the required balance-sheet fields. No values are fabricated."
        )

    with tabs[1]:
        st.subheader("Liquidity & Leverage")
        rows = [
            ["Current Ratio", "N/A" if current_ratio is None else f"{current_ratio:.2f}x"],
            ["Quick Ratio", "N/A" if quick_ratio is None else f"{quick_ratio:.2f}x"],
            ["Debt / Equity", "N/A" if debt_equity is None else f"{debt_equity:.2f}x"],
            ["Interest Coverage", "N/A" if interest_coverage is None else f"{interest_coverage:.2f}x"],
            ["Cash Balance", f"${cash/1_000_000:.2f}M" if cash else "N/A"],
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Ratio", "Value"]), use_container_width=True, hide_index=True)
        if assets or equity or debt or current_assets or current_liabilities:
            bs_check = assets - (debt + current_liabilities + equity)
            st.markdown("### Balance Sheet Reconciliation")
            st.dataframe(pd.DataFrame([
                ["Total Assets", f"${assets/1_000_000:.2f}M"],
                ["Total Liabilities", f"${(debt + current_liabilities)/1_000_000:.2f}M"],
                ["Total Equity", f"${equity/1_000_000:.2f}M"],
                ["Assets − Liabilities − Equity", f"${bs_check/1_000_000:.4f}M"],
            ], columns=["Balance Sheet Check", "Value"]), use_container_width=True, hide_index=True)
        st.caption("V4 includes a reconciled management balance-sheet layer.")

    with tabs[2]:
        st.subheader("Working Capital")
        wc = [
            ["DSO — Days Sales Outstanding", "N/A" if dso is None else f"{dso:.1f} days"],
            ["DPO — Days Payable Outstanding", "N/A" if dpo is None else f"{dpo:.1f} days"],
            ["Cash Conversion Cycle", "N/A" if ccc is None else f"{ccc:.1f} days"],
            ["Receivables", "N/A" if not receivables else f"${receivables/1_000_000:.2f}M"],
            ["Payables", "N/A" if not payables else f"${payables/1_000_000:.2f}M"],
        ]
        st.dataframe(pd.DataFrame(wc, columns=["Metric", "Value"]), use_container_width=True, hide_index=True)
        st.caption("DSO/DPO/CCC use the V4 balance-sheet layer and latest-month annualized operating flows.")

    with tabs[3]:
        st.subheader("Market Analytics")
        st.info("Market analytics use a separate market-price dataset. ERP financial transactions are not used as a substitute.")
        uploaded_market = st.file_uploader(
            "Upload Market Price CSV",
            type=["csv"],
            key="v26_market_csv",
            help="Expected fields: Date, Asset Price/Price/Close, and optionally Benchmark Price.",
        )

        if uploaded_market is not None:
            market = pd.read_csv(uploaded_market)
            market.columns = [str(c).strip() for c in market.columns]
            date_col = _v22_col(market, ["Date", "date", "Datetime", "Timestamp"])
            price_col = _v22_col(market, ["Asset Price", "Price", "Close", "Adj Close"])
            # IMPORTANT: prefer the numeric benchmark-price field before a text benchmark-name field.
            benchmark_col = _v22_col(market, ["Benchmark Price", "Benchmark Close", "Index Price", "Benchmark", "Index"])

            if not date_col or not price_col:
                st.error("Market CSV needs Date and Asset Price/Price/Close columns.")
            else:
                market[date_col] = pd.to_datetime(market[date_col], errors="coerce")
                market[price_col] = pd.to_numeric(market[price_col], errors="coerce")
                market = market.dropna(subset=[date_col, price_col]).sort_values(date_col).copy()

                # If the detected benchmark column is text (e.g. a benchmark name),
                # try the numeric Benchmark Price column explicitly.
                if benchmark_col:
                    numeric_benchmark = pd.to_numeric(market[benchmark_col], errors="coerce")
                    if numeric_benchmark.notna().sum() < 2 and "Benchmark Price" in market.columns:
                        benchmark_col = "Benchmark Price"
                        numeric_benchmark = pd.to_numeric(market[benchmark_col], errors="coerce")
                    market["__BenchmarkPrice"] = numeric_benchmark
                else:
                    market["__BenchmarkPrice"] = np.nan

                market["Asset Return"] = market[price_col].pct_change()
                market["Benchmark Return"] = market["__BenchmarkPrice"].pct_change()

                # Keep the uploaded market dataset available to Technical Analytics.
                st.session_state["finsight_market_data"] = market.copy()
                st.session_state["finsight_market_date_col"] = date_col
                st.session_state["finsight_market_price_col"] = price_col

        else:
            market = st.session_state.get("finsight_market_data", pd.DataFrame())
            date_col = st.session_state.get("finsight_market_date_col", "Date")
            price_col = st.session_state.get("finsight_market_price_col", "Asset Price")

        if not market.empty and "Asset Return" in market.columns:
            has_benchmark = market["Benchmark Return"].notna().sum() >= 2
            aligned = market[["Asset Return", "Benchmark Return"]].dropna()
            ret = market.dropna(subset=["Asset Return"]).copy()

            asset_std = ret["Asset Return"].std()
            asset_vol = asset_std * np.sqrt(252) if pd.notna(asset_std) else np.nan
            sharpe = ret["Asset Return"].mean() / asset_std * np.sqrt(252) if asset_std and asset_std > 0 else np.nan

            beta = corr = r2 = alpha = info_ratio = np.nan
            if has_benchmark and len(aligned) >= 2:
                bvar = aligned["Benchmark Return"].var()
                if bvar > 0:
                    beta = aligned["Asset Return"].cov(aligned["Benchmark Return"]) / bvar
                    corr = aligned["Asset Return"].corr(aligned["Benchmark Return"])
                    r2 = corr ** 2 if pd.notna(corr) else np.nan
                    alpha = (aligned["Asset Return"].mean() - beta * aligned["Benchmark Return"].mean()) * 252
                active = aligned["Asset Return"] - aligned["Benchmark Return"]
                info_ratio = active.mean() / active.std() * np.sqrt(252) if active.std() > 0 else np.nan

            downside = ret.loc[ret["Asset Return"] < 0, "Asset Return"].std()
            sortino = ret["Asset Return"].mean() / downside * np.sqrt(252) if downside and downside > 0 else np.nan
            treynor = (ret["Asset Return"].mean() * 252) / beta if pd.notna(beta) and beta != 0 else np.nan
            var95 = ret["Asset Return"].quantile(0.05)
            cvar95 = ret.loc[ret["Asset Return"] <= var95, "Asset Return"].mean()
            wealth = (1 + ret["Asset Return"]).cumprod()
            drawdown = wealth / wealth.cummax() - 1

            st.markdown("### 📈 Market Performance")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Alpha", f"{alpha:.2%}" if pd.notna(alpha) else "N/A")
            c2.metric("Beta", f"{beta:.2f}" if pd.notna(beta) else "N/A")
            c3.metric("R²", f"{r2:.2f}" if pd.notna(r2) else "N/A")
            c4.metric("Sharpe", f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A")

            perf = market[[date_col, price_col]].copy()
            perf["Asset"] = perf[price_col] / perf[price_col].iloc[0] * 100
            if has_benchmark:
                b = market["__BenchmarkPrice"]
                first_valid = b.dropna()
                if len(first_valid):
                    perf["Benchmark"] = b / first_valid.iloc[0] * 100

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=perf[date_col], y=perf["Asset"], mode="lines", name="Asset"))
            if "Benchmark" in perf:
                fig.add_trace(go.Scatter(x=perf[date_col], y=perf["Benchmark"], mode="lines", name="Benchmark"))
            chart_layout(fig, height=390)
            fig.update_yaxes(title_text="Indexed Performance")
            fig.update_xaxes(title_text="Date")
            st.plotly_chart(fig, use_container_width=True)

            dd = pd.DataFrame({"Date": ret[date_col], "Drawdown": drawdown.values})
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=dd["Date"], y=dd["Drawdown"] * 100, mode="lines", name="Drawdown", fill="tozeroy"))
            chart_layout(fig_dd, height=300)
            fig_dd.update_yaxes(title_text="Drawdown (%)")
            fig_dd.update_xaxes(title_text="Date")
            st.plotly_chart(fig_dd, use_container_width=True)

            metrics = pd.DataFrame([
                ["Alpha", f"{alpha:.2%}" if pd.notna(alpha) else "N/A", "Annualized active return after beta adjustment"],
                ["Beta", f"{beta:.2f}" if pd.notna(beta) else "N/A", "Sensitivity to benchmark"],
                ["R / Correlation", f"{corr:.2f}" if pd.notna(corr) else "N/A", "Relationship with benchmark"],
                ["R²", f"{r2:.2f}" if pd.notna(r2) else "N/A", "Benchmark variance explained"],
                ["Sharpe Ratio", f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A", "Return per unit of total risk"],
                ["Sortino Ratio", f"{sortino:.2f}" if pd.notna(sortino) else "N/A", "Return per unit of downside risk"],
                ["Treynor Ratio", f"{treynor:.2f}" if pd.notna(treynor) else "N/A", "Return per unit of systematic risk"],
                ["Information Ratio", f"{info_ratio:.2f}" if pd.notna(info_ratio) else "N/A", "Active return per tracking error"],
                ["Annualized Volatility", f"{asset_vol:.2%}" if pd.notna(asset_vol) else "N/A", "Annualized return volatility"],
                ["VaR 95%", f"{var95:.2%}", "5th percentile one-period return"],
                ["CVaR 95%", f"{cvar95:.2%}", "Average return in worst 5%"],
                ["Maximum Drawdown", f"{drawdown.min():.2%}", "Peak-to-trough decline"],
            ], columns=["Market Metric", "Value", "Interpretation"])
            st.dataframe(metrics, use_container_width=True, hide_index=True)
        else:
            st.warning("Upload the demo market CSV to activate Market Analytics.")
            st.caption("Expected columns: Date, Asset Price, Benchmark Price.")
    with tabs[4]:
        st.subheader("Technical Analytics")
        st.info("Technical indicators use chronological market-price data and are intentionally separated from ERP accounting data.")

        tech_file = st.file_uploader(
            "Upload Price CSV for Technical Analysis (optional if Market CSV is loaded)",
            type=["csv"],
            key="v26_tech_csv",
            help="You can upload a separate price file, or use the Market CSV already uploaded in Market Analytics.",
        )

        if tech_file is not None:
            tech = pd.read_csv(tech_file)
            tech.columns = [str(c).strip() for c in tech.columns]
            dcol = _v22_col(tech, ["Date", "date", "Datetime", "Timestamp"])
            pcol = _v22_col(tech, ["Asset Price", "Price", "Close", "Adj Close"])
            if not dcol or not pcol:
                st.error("Technical CSV needs Date and Asset Price/Price/Close columns.")
                tech = pd.DataFrame()
            else:
                tech[dcol] = pd.to_datetime(tech[dcol], errors="coerce")
                tech[pcol] = pd.to_numeric(tech[pcol], errors="coerce")
                tech = tech.dropna(subset=[dcol, pcol]).sort_values(dcol).copy()
        else:
            tech = st.session_state.get("finsight_market_data", pd.DataFrame()).copy()
            dcol = st.session_state.get("finsight_market_date_col", "Date")
            pcol = st.session_state.get("finsight_market_price_col", "Asset Price")

        if not tech.empty and dcol in tech.columns and pcol in tech.columns:
            price = pd.to_numeric(tech[pcol], errors="coerce")
            tech["SMA 20"] = price.rolling(20).mean()
            tech["SMA 50"] = price.rolling(50).mean()

            delta = price.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            tech["RSI 14"] = 100 - (100 / (1 + rs))

            ema12 = price.ewm(span=12, adjust=False).mean()
            ema26 = price.ewm(span=26, adjust=False).mean()
            tech["MACD"] = ema12 - ema26
            tech["Signal"] = tech["MACD"].ewm(span=9, adjust=False).mean()

            mid = price.rolling(20).mean()
            std = price.rolling(20).std()
            tech["Upper BB"] = mid + 2 * std
            tech["Lower BB"] = mid - 2 * std

            high = float(price.max())
            low = float(price.min())
            diff = high - low
            fib = [
                ("0.0%", high), ("23.6%", high - .236*diff), ("38.2%", high - .382*diff),
                ("50.0%", high - .500*diff), ("61.8%", high - .618*diff),
                ("78.6%", high - .786*diff), ("100.0%", low)
            ]

            latest = tech.iloc[-1]
            a,b,c,d = st.columns(4)
            a.metric("Latest Price", f"{price.iloc[-1]:,.2f}")
            b.metric("RSI 14", "N/A" if pd.isna(latest["RSI 14"]) else f"{latest['RSI 14']:.1f}")
            c.metric("SMA 20", "N/A" if pd.isna(latest["SMA 20"]) else f"{latest['SMA 20']:,.2f}")
            d.metric("SMA 50", "N/A" if pd.isna(latest["SMA 50"]) else f"{latest['SMA 50']:,.2f}")

            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(x=tech[dcol], y=tech[pcol], mode="lines", name="Price"))
            fig_price.add_trace(go.Scatter(x=tech[dcol], y=tech["SMA 20"], mode="lines", name="SMA 20"))
            fig_price.add_trace(go.Scatter(x=tech[dcol], y=tech["SMA 50"], mode="lines", name="SMA 50"))
            fig_price.add_trace(go.Scatter(x=tech[dcol], y=tech["Upper BB"], mode="lines", name="Upper BB"))
            fig_price.add_trace(go.Scatter(x=tech[dcol], y=tech["Lower BB"], mode="lines", name="Lower BB"))
            chart_layout(fig_price, height=420)
            fig_price.update_yaxes(title_text="Price")
            fig_price.update_xaxes(title_text="Date")
            st.plotly_chart(fig_price, use_container_width=True)

            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=tech[dcol], y=tech["RSI 14"], mode="lines", name="RSI 14"))
            fig_rsi.add_hline(y=70, line_dash="dash", annotation_text="70 Overbought")
            fig_rsi.add_hline(y=30, line_dash="dash", annotation_text="30 Oversold")
            chart_layout(fig_rsi, height=280)
            fig_rsi.update_yaxes(title_text="RSI", range=[0,100])
            st.plotly_chart(fig_rsi, use_container_width=True)

            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=tech[dcol], y=tech["MACD"], mode="lines", name="MACD"))
            fig_macd.add_trace(go.Scatter(x=tech[dcol], y=tech["Signal"], mode="lines", name="Signal"))
            chart_layout(fig_macd, height=280)
            fig_macd.update_yaxes(title_text="MACD")
            st.plotly_chart(fig_macd, use_container_width=True)

            fig_fib = go.Figure()
            fig_fib.add_trace(go.Scatter(x=tech[dcol], y=tech[pcol], mode="lines", name="Price"))
            for level, value in fib:
                fig_fib.add_hline(y=value, line_dash="dot", annotation_text=f"Fib {level}")
            chart_layout(fig_fib, height=420)
            fig_fib.update_yaxes(title_text="Price")
            st.plotly_chart(fig_fib, use_container_width=True)

            st.markdown("### Fibonacci Retracement")
            st.dataframe(pd.DataFrame(fib, columns=["Level", "Price"]), use_container_width=True, hide_index=True)
            st.caption("Technical indicators are analytical reference tools, not investment recommendations.")
        else:
            st.warning("Upload the market CSV in Market Analytics first, or upload a price CSV here.")


if page == "Executive Dashboard":
    st.subheader("Executive Dashboard")
    st.caption("CFO view of growth, profitability, liquidity, budget performance and enterprise risk.")

    # Dashboard period control. Global sidebar filters still apply first.
    period_col, view_col = st.columns([1, 2.4])
    with period_col:
        dashboard_period = st.selectbox(
            "Analysis Period",
            ["Selected Range", "Monthly", "Quarterly", "YTD", "FY"],
            index=0,
            key="dashboard_period_v7",
        )
    with view_col:
        st.caption("Period controls sit on top of the global Region / Country / Department / Date filters.")

    dashboard_view = view.copy()
    dashboard_view["Month"] = pd.to_datetime(dashboard_view["Date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    max_date = dashboard_view["Date"].max()

    if pd.notna(max_date):
        if dashboard_period == "YTD":
            dashboard_view = dashboard_view[dashboard_view["Date"].dt.year == max_date.year].copy()
        elif dashboard_period == "FY":
            dashboard_view = dashboard_view[dashboard_view["Date"].dt.year == max_date.year].copy()
        elif dashboard_period == "Quarterly":
            q_period = pd.Period(max_date, freq="Q")
            dashboard_view = dashboard_view[dashboard_view["Date"].dt.to_period("Q") == q_period].copy()
        elif dashboard_period == "Monthly":
            m_period = pd.Period(max_date, freq="M")
            dashboard_view = dashboard_view[dashboard_view["Date"].dt.to_period("M") == m_period].copy()
        # Selected Range intentionally keeps the full globally filtered view.

    if dashboard_view.empty:
        st.warning("No data is available for the selected dashboard period. Adjust the global date filter or choose another period.")
        dashboard_view = view.copy()
        dashboard_view["Month"] = pd.to_datetime(dashboard_view["Date"], errors="coerce").dt.to_period("M").dt.to_timestamp()

    # Core P&L measures for the selected dashboard period.
    revenue = float(dashboard_view["Revenue USD"].sum())
    actual = float(dashboard_view["Actual USD"].sum())
    budget = float(dashboard_view["Budget USD"].sum())
    variance = actual - budget

    mgmt = dashboard_view.copy()
    mgmt["Month"] = pd.to_datetime(mgmt["Date"], errors="coerce").dt.to_period("M").dt.to_timestamp()

    if "Cost Type" in mgmt.columns:
        cogs_total = float(mgmt.loc[mgmt["Cost Type"].eq("COGS"), "Actual USD"].sum())
        opex_total = float(mgmt.loc[mgmt["Cost Type"].eq("Opex"), "Actual USD"].sum())
    else:
        cogs_total = 0.0
        opex_total = actual

    if "Gross Profit" in mgmt.columns:
        gross_profit = float(mgmt.groupby("Month")["Gross Profit"].first().sum())
    else:
        gross_profit = float(revenue - cogs_total)

    if "EBITDA" in mgmt.columns:
        ebitda = float(mgmt.groupby("Month")["EBITDA"].first().sum())
    else:
        ebitda = float(gross_profit - opex_total)

    if "PAT" in mgmt.columns:
        pat = float(mgmt.groupby("Month")["PAT"].first().sum())
    else:
        pat = float(ebitda - revenue * 0.04)

    if "Closing Cash" in mgmt.columns:
        cash_series = mgmt.groupby("Month")["Closing Cash"].first().sort_index()
        cash_balance = float(cash_series.iloc[-1]) if len(cash_series) else 0.0
    else:
        monthly_cash = mgmt.groupby("Month").agg(Revenue=("Revenue USD", "sum"), Actual=("Actual USD", "sum"))
        cash_balance = float((monthly_cash["Revenue"] - monthly_cash["Actual"]).cumsum().iloc[-1]) if len(monthly_cash) else 0.0

    gross_margin = gross_profit / revenue * 100 if revenue else 0
    ebitda_margin = ebitda / revenue * 100 if revenue else 0
    pat_margin = pat / revenue * 100 if revenue else 0
    variance_pct = variance / budget * 100 if budget else 0

    anomaly_count = int(dashboard_view["Anomaly Flag"].sum())
    variance_risk = min(abs(variance_pct) * 8, 55)
    anomaly_risk = min(anomaly_count / max(len(dashboard_view), 1) * 100 * 1.8, 35)
    risk_score = min(max(variance_risk + anomaly_risk, 0), 100)

    # KPI cards — the six metrics a CFO should see first.
    kpi_cols = st.columns(6)
    cards = [
        ("Revenue", money_usd(revenue), "Group revenue"),
        ("Gross Profit", money_usd(gross_profit), f"Gross margin {gross_margin:.1f}%"),
        ("EBITDA", money_usd(ebitda), f"EBITDA margin {ebitda_margin:.1f}%"),
        ("Cash Balance", money_usd(cash_balance), "Closing modeled cash"),
        ("EBITDA Margin", pct(ebitda_margin), "Operating profitability"),
        ("Risk Score", f"{risk_score:.0f}/100", f"{anomaly_count:,} anomaly flags"),
    ]
    for col, (label, value, note) in zip(kpi_cols, cards):
        with col:
            html_block(
                f'<div class="kpi"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value}</div>'
                f'<div class="kpi-note">{note}</div></div>'
            )

    st.write("")

    # CFO pulse: compact interpretation of the selected period.
    pulse1, pulse2, pulse3 = st.columns(3)
    with pulse1:
        if variance > 0:
            html_block(f'<div class="danger"><b>🔴 Budget Pressure</b><br>{money_usd(variance)} unfavorable cost variance.<br><span class="small">{pct(abs(variance_pct))} of budget.</span></div>')
        else:
            html_block(f'<div class="insight"><b>🟢 Budget Performance</b><br>{money_usd(abs(variance))} favorable vs budget.<br><span class="small">Validate whether savings are sustainable.</span></div>')
    with pulse2:
        status = "Healthy" if ebitda_margin >= 18 else "Watch" if ebitda_margin >= 15 else "Pressure"
        cls = "insight" if status == "Healthy" else "warning" if status == "Watch" else "danger"
        html_block(f'<div class="{cls}"><b>📈 Profitability: {status}</b><br>EBITDA margin is {pct(ebitda_margin)}.<br><span class="small">PAT margin: {pct(pat_margin)}.</span></div>')
    with pulse3:
        risk_label = "Low" if risk_score < 30 else "Moderate" if risk_score < 60 else "High"
        cls = "insight" if risk_label == "Low" else "warning" if risk_label == "Moderate" else "danger"
        html_block(f'<div class="{cls}"><b>🛡 Enterprise Risk: {risk_label}</b><br>Risk score {risk_score:.0f}/100.<br><span class="small">{anomaly_count:,} transactions flagged for review.</span></div>')

    # Monthly management view. V3 management metrics are repeated on transaction rows,
    # so monthly P&L values use FIRST rather than SUM.
    monthly_base = view.copy()
    monthly_base["Month"] = pd.to_datetime(monthly_base["Date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    monthly = monthly_base.groupby("Month", as_index=False).agg(
        Revenue=("Revenue USD", "sum"),
        Budget=("Budget USD", "sum"),
        Actual_Cost=("Actual USD", "sum"),
    )

    if "Cost Type" in monthly_base.columns:
        cogs_month = monthly_base.loc[monthly_base["Cost Type"].eq("COGS")].groupby("Month")["Actual USD"].sum()
        opex_month = monthly_base.loc[monthly_base["Cost Type"].eq("Opex")].groupby("Month")["Actual USD"].sum()
        monthly["COGS"] = monthly["Month"].map(cogs_month).fillna(0)
        monthly["Opex"] = monthly["Month"].map(opex_month).fillna(0)
    else:
        monthly["COGS"] = 0.0
        monthly["Opex"] = monthly["Actual_Cost"]

    monthly["Gross_Profit"] = monthly["Revenue"] - monthly["COGS"]
    monthly["EBITDA"] = monthly["Gross_Profit"] - monthly["Opex"]

    if "Gross Profit" in monthly_base.columns:
        gp = monthly_base.groupby("Month")["Gross Profit"].first()
        monthly["Gross_Profit"] = monthly["Month"].map(gp).fillna(monthly["Gross_Profit"])
    if "EBITDA" in monthly_base.columns:
        eb = monthly_base.groupby("Month")["EBITDA"].first()
        monthly["EBITDA"] = monthly["Month"].map(eb).fillna(monthly["EBITDA"])
    if "PAT" in monthly_base.columns:
        pat_series = monthly_base.groupby("Month")["PAT"].first()
        monthly["PAT"] = monthly["Month"].map(pat_series)
    else:
        monthly["PAT"] = np.nan
    if "Closing Cash" in monthly_base.columns:
        cash_series = monthly_base.groupby("Month")["Closing Cash"].first()
        monthly["Cash"] = monthly["Month"].map(cash_series)
    else:
        monthly["Cash"] = (monthly["Revenue"] - monthly["Actual_Cost"]).cumsum()
    monthly["PAT"] = monthly["PAT"].fillna(monthly["EBITDA"] - monthly["Revenue"] * 0.04)

    left, right = st.columns([1.65, 1])
    with left:
        st.subheader("Revenue vs Budget")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly["Revenue"], mode="lines+markers", name="Revenue"))
        fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly["Budget"], mode="lines", name="Budget"))
        chart_layout(fig, height=350)
        fig.update_yaxes(tickprefix="$", tickformat="~s")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("P&L Summary")
        pnl = pd.DataFrame([
            ["Revenue", money_usd(revenue)],
            ["Gross Profit", money_usd(gross_profit)],
            ["Gross Margin", pct(gross_margin)],
            ["EBITDA", money_usd(ebitda)],
            ["EBITDA Margin", pct(ebitda_margin)],
            ["PAT", money_usd(pat)],
            ["PAT Margin", pct(pat_margin)],
        ], columns=["Metric", "Value"])
        st.dataframe(pnl, use_container_width=True, hide_index=True)

    st.subheader("Profitability Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly["Gross_Profit"], mode="lines+markers", name="Gross Profit"))
    fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly["EBITDA"], mode="lines+markers", name="EBITDA"))
    fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly["PAT"], mode="lines", name="PAT"))
    chart_layout(fig, height=330)
    fig.update_yaxes(tickprefix="$", tickformat="~s")
    st.plotly_chart(fig, use_container_width=True)

    cash_col, region_col = st.columns([1.15, 1])
    with cash_col:
        st.subheader("Cash Flow Overview")
        cash_chart = go.Figure()
        cash_chart.add_trace(go.Scatter(x=monthly["Month"], y=monthly["Cash"], mode="lines+markers", name="Closing Cash"))
        chart_layout(cash_chart, height=300)
        cash_chart.update_yaxes(tickprefix="$", tickformat="~s")
        st.plotly_chart(cash_chart, use_container_width=True)

    with region_col:
        st.subheader("Regional Performance")
        regional = dashboard_view.groupby("Region", as_index=False).agg(
            Revenue=("Revenue USD", "sum"),
            Budget=("Budget USD", "sum"),
            Actual=("Actual USD", "sum"),
        )
        regional["Variance"] = regional["Actual"] - regional["Budget"]
        regional["Revenue Share"] = regional["Revenue"] / max(regional["Revenue"].sum(), 1)
        regional["EBITDA"] = ebitda * regional["Revenue Share"]
        regional["EBITDA Margin %"] = np.where(regional["Revenue"] != 0, regional["EBITDA"] / regional["Revenue"] * 100, 0)
        display = regional.drop(columns=["Revenue Share"]).copy()
        for c in ["Revenue", "Budget", "Actual", "Variance", "EBITDA"]:
            display[c] = display[c].map(money_usd)
        display["EBITDA Margin %"] = regional["EBITDA Margin %"].map(pct)
        st.dataframe(display, use_container_width=True, hide_index=True)

    cost_col, action_col = st.columns([1.15, 1])
    with cost_col:
        st.subheader("Top Cost Drivers")
        cost = dashboard_view.groupby("Cost Category", as_index=False).agg(Budget=("Budget USD", "sum"), Actual=("Actual USD", "sum"))
        cost["Variance"] = cost["Actual"] - cost["Budget"]
        top_cost = cost.sort_values("Variance", ascending=False).head(8)
        fig = px.bar(top_cost.sort_values("Variance"), x="Variance", y="Cost Category", orientation="h", title="Unfavorable Variance")
        chart_layout(fig, height=330)
        fig.update_xaxes(tickprefix="$", tickformat="~s")
        st.plotly_chart(fig, use_container_width=True)

    with action_col:
        st.subheader("AI CFO Recommended Actions")
        if variance > 0:
            html_block(f'<div class="danger"><b>🔴 Cost Control</b><br>Actual cost is {money_usd(variance)} above budget.<br><span class="small">Owner: Finance Controller · Investigate the largest cost drivers.</span></div>')
        else:
            html_block(f'<div class="insight"><b>🟢 Cost Performance</b><br>Actual cost is {money_usd(abs(variance))} below budget.<br><span class="small">Owner: FP&A · Validate sustainability of the savings.</span></div>')
        if anomaly_count:
            html_block(f'<div class="warning"><b>🟠 Risk Investigation</b><br>{anomaly_count:,} transactions are flagged for review.<br><span class="small">Owner: Controller · Prioritize high-value anomalies.</span></div>')
        open_count, progress_count, resolved_count = action_status_counts()
        html_block(f'<div class="insight"><b>🎯 Management Actions</b><br>{open_count} open · {progress_count} in progress · {resolved_count} resolved.<br><span class="small">Use AI CFO Action Center to track ownership and closure.</span></div>')
        html_block(f'<div class="insight"><b>🟢 Profitability</b><br>Gross margin is {pct(gross_margin)}, EBITDA margin is {pct(ebitda_margin)}, and PAT margin is {pct(pat_margin)}.<br><span class="small">CFO focus: protect profitable growth and cash generation.</span></div>')

    st.subheader("CFO Management Snapshot")
    snap = pd.DataFrame([
        ["Revenue", revenue, "Growth / top-line performance"],
        ["Gross Profit", gross_profit, "Gross margin protection"],
        ["EBITDA", ebitda, "Operating earnings"],
        ["PAT", pat, "Bottom-line earnings"],
        ["Cash Balance", cash_balance, "Liquidity position"],
        ["Unfavorable Variance", max(variance, 0), "Immediate cost-control exposure"],
    ], columns=["Metric", "Value", "CFO Focus"])
    snap["Value"] = snap["Value"].map(money_usd)
    st.dataframe(snap, use_container_width=True, hide_index=True)


# ============================================================
# AI CFO — MULTI-STEP INVESTIGATION V18
# ============================================================
elif page == "Ratio Analysis":
    _v22_ratio_analysis(view)

elif page == "AI CFO":
    st.subheader("🤖 AI CFO")
    st.caption("Multi-step financial investigation engine: detect → quantify → trace the driver → explain → recommend → route to action.")

    ai = view.copy()
    for col in ["Budget USD", "Actual USD", "Revenue USD", "Variance USD", "Profit USD"]:
        if col in ai.columns:
            ai[col] = pd.to_numeric(ai[col], errors="coerce").fillna(0.0)
    ai["Unfavorable Variance"] = ai["Variance USD"].clip(lower=0)

    # --------------------------------------------------------
    # Step 1 — Materiality scan
    # --------------------------------------------------------
    total_budget = float(ai["Budget USD"].sum())
    total_actual = float(ai["Actual USD"].sum())
    total_variance = float(ai["Variance USD"].sum())
    unfavorable = float(ai["Unfavorable Variance"].sum())
    anomalies = int(pd.to_numeric(ai.get("Anomaly Flag", 0), errors="coerce").fillna(0).sum())

    findings = []
    if unfavorable > 0:
        findings.append({
            "Priority": "High" if unfavorable / max(total_budget, 1) >= 0.03 else "Medium",
            "Area": "Cost Control",
            "Issue": "Cost over budget",
            "Finding": f"Actual cost is {money_usd(total_variance)} above budget on the current scope.",
            "Impact": money_usd(unfavorable),
            "Risk": "Margin and cash pressure if the unfavorable variance persists.",
            "Recommendation": "Trace the variance through Region → Department → Cost Category → GL → Transaction and address the largest controllable drivers.",
            "Owner": "Finance Controller",
        })
    if anomalies:
        findings.append({
            "Priority": "High" if anomalies >= 25 else "Medium",
            "Area": "Risk & Controls",
            "Issue": "Anomaly exposure",
            "Finding": f"{anomalies:,} transaction(s) carry source anomaly flags in the current scope.",
            "Impact": f"{anomalies:,} transactions",
            "Risk": "Potential unusual-spend, posting or control risk.",
            "Recommendation": "Prioritize high-value anomalous transactions and document investigation outcomes.",
            "Owner": "Controller",
        })

    if not findings:
        findings.append({
            "Priority": "Low", "Area": "Financial Health", "Issue": "No material exception detected",
            "Finding": "No material management exception was detected from the current scope.",
            "Impact": "Monitoring", "Risk": "Low",
            "Recommendation": "Continue routine monitoring and periodic forecast review.", "Owner": "FP&A",
        })

    open_count, progress_count, resolved_count = action_status_counts()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Material Signals", len(findings))
    k2.metric("Unfavorable Exposure", money_usd(unfavorable))
    k3.metric("Anomalies", f"{anomalies:,}")
    k4.metric("Open Actions", open_count)
    k5.metric("Resolved", resolved_count)

    st.markdown("### CFO Monitoring Feed")
    for idx, f in enumerate(findings):
        cls = "danger" if f["Priority"] == "High" else "warning" if f["Priority"] == "Medium" else "insight"
        html_block(
            f'<div class="{cls}"><b>{f["Priority"]} · {f["Area"]} · {f["Issue"]}</b><br>'
            f'{f["Finding"]}<br><span class="small">Financial impact: {f["Impact"]} · Owner: {f["Owner"]}</span></div>'
        )
        with st.expander(f"🔎 Investigate: {f['Issue']}"):
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Finding**")
                st.write(f["Finding"])
                st.write("**Financial impact**")
                st.write(f["Impact"])
                st.write("**Risk**")
                st.write(f["Risk"])
            with c2:
                st.write("**Recommended action**")
                st.write(f["Recommendation"])
                st.write("**Owner**")
                st.write(f["Owner"])
                if st.button("Create Management Action", key=f"create_ai_action_{idx}"):
                    created = create_cfo_action(f["Priority"], f["Area"], f["Issue"], f["Impact"], f["Recommendation"], f["Owner"])
                    if created:
                        st.success("Management action created and added to the Action Center.")
                    else:
                        st.info("An open action for this issue already exists.")

    # --------------------------------------------------------
    # Step 2–5 — Multi-step driver investigation
    # --------------------------------------------------------
    st.markdown("### 🧠 Multi-Step CFO Investigation")
    st.caption("Select a starting scope. FinSight then traces the unfavorable variance down the management hierarchy and surfaces the highest-value drivers.")

    level1, level2 = st.columns(2)
    with level1:
        regions = ["All Regions"] + sorted(ai["Region"].dropna().astype(str).unique().tolist())
        selected_region = st.selectbox("1 · Region", regions, key="ai_region_v18")
    with level2:
        countries = ["All Countries"] + sorted(ai["Country"].dropna().astype(str).unique().tolist())
        selected_country = st.selectbox("Country", countries, key="ai_country_v18")

    scoped = ai.copy()
    scope_parts = []
    if selected_region != "All Regions":
        scoped = scoped[scoped["Region"].astype(str) == selected_region]
        scope_parts.append(selected_region)
    if selected_country != "All Countries":
        scoped = scoped[scoped["Country"].astype(str) == selected_country]
        scope_parts.append(selected_country)

    if scoped.empty:
        st.warning("No transactions match the selected scope.")
    else:
        # Step 2: Region/country scope -> department
        dept = scoped.groupby("Department", as_index=False).agg(
            Budget=("Budget USD", "sum"), Actual=("Actual USD", "sum"),
            Unfavorable=("Unfavorable Variance", "sum"), Transactions=("Transaction ID", "count")
        )
        dept["Variance"] = dept["Actual"] - dept["Budget"]
        dept = dept.sort_values("Unfavorable", ascending=False)
        dept_options = dept["Department"].tolist()
        selected_dept = st.selectbox("2 · Department", ["All Departments"] + dept_options, key="ai_dept_v18")

        scoped2 = scoped.copy()
        if selected_dept != "All Departments":
            scoped2 = scoped2[scoped2["Department"].astype(str) == selected_dept]

        # Step 3: department -> cost category
        cat = scoped2.groupby("Cost Category", as_index=False).agg(
            Budget=("Budget USD", "sum"), Actual=("Actual USD", "sum"),
            Unfavorable=("Unfavorable Variance", "sum"), Transactions=("Transaction ID", "count")
        )
        cat["Variance"] = cat["Actual"] - cat["Budget"]
        cat = cat.sort_values("Unfavorable", ascending=False)
        cat_options = cat["Cost Category"].tolist()
        selected_cat = st.selectbox("3 · Cost Category", ["All Categories"] + cat_options, key="ai_cat_v18")

        scoped3 = scoped2.copy()
        if selected_cat != "All Categories":
            scoped3 = scoped3[scoped3["Cost Category"].astype(str) == selected_cat]

        # Step 4: category -> GL account
        gl = scoped3.groupby("GL Account", as_index=False).agg(
            Budget=("Budget USD", "sum"), Actual=("Actual USD", "sum"),
            Unfavorable=("Unfavorable Variance", "sum"), Transactions=("Transaction ID", "count")
        )
        gl["Variance"] = gl["Actual"] - gl["Budget"]
        gl = gl.sort_values("Unfavorable", ascending=False)
        gl_options = gl["GL Account"].astype(str).tolist()
        selected_gl = st.selectbox("4 · GL Account", ["All GL Accounts"] + gl_options, key="ai_gl_v18")

        scoped4 = scoped3.copy()
        if selected_gl != "All GL Accounts":
            scoped4 = scoped4[scoped4["GL Account"].astype(str) == selected_gl]

        # Step 5: transaction-level evidence
        txn = scoped4.copy()
        txn["Unfavorable Variance"] = txn["Variance USD"].clip(lower=0)
        txn = txn.sort_values(["Unfavorable Variance", "Actual USD"], ascending=False).head(10)

        final_budget = float(scoped4["Budget USD"].sum())
        final_actual = float(scoped4["Actual USD"].sum())
        final_variance = final_actual - final_budget
        final_unfav = float(scoped4["Unfavorable Variance"].sum())
        final_anomalies = int(pd.to_numeric(scoped4.get("Anomaly Flag", 0), errors="coerce").fillna(0).sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Scoped Budget", money_usd(final_budget))
        m2.metric("Scoped Actual", money_usd(final_actual))
        m3.metric("Unfavorable Variance", money_usd(final_unfav))
        m4.metric("Anomaly Flags", f"{final_anomalies:,}")

        if final_unfav > 0:
            # Root-cause ranking inside the selected scope.
            root = scoped4.groupby("Cost Category", as_index=False).agg(Unfavorable=("Unfavorable Variance", "sum"), Actual=("Actual USD", "sum"))
            root = root.sort_values("Unfavorable", ascending=False)
            root_driver = root.iloc[0] if not root.empty else None
            if root_driver is not None:
                driver_name = str(root_driver["Cost Category"])
                driver_impact = float(root_driver["Unfavorable"])
                share = driver_impact / final_unfav * 100 if final_unfav else 0
                scope_label = " → ".join(scope_parts) if scope_parts else "All Regions / Countries"
                st.success(
                    f"**CFO conclusion:** Within {scope_label}, the selected management path has "
                    f"{money_usd(final_unfav)} unfavorable exposure. **{driver_name}** is the largest "
                    f"remaining cost driver at {money_usd(driver_impact)} ({share:.1f}% of scoped unfavorable exposure)."
                )
                recommendation = (
                    f"Investigate {driver_name} within the selected management path, starting with the highest-value transactions. "
                    f"Validate budget assumptions, headcount/vendor drivers and posting accuracy before approving corrective action."
                )
                r1, r2 = st.columns([2, 1])
                with r1:
                    st.write("**AI CFO recommended next step**")
                    st.write(recommendation)
                with r2:
                    if st.button("🎯 Create Driver Action", key="ai_driver_action_v18"):
                        issue = f"Unfavorable driver: {driver_name}"
                        created = create_cfo_action(
                            "High" if share >= 25 else "Medium",
                            "AI CFO Investigation",
                            issue,
                            money_usd(driver_impact),
                            recommendation,
                            "Finance Controller",
                        )
                        if created:
                            st.success("Driver action created in Action Center.")
                        else:
                            st.info("An open action for this driver already exists.")

        st.markdown("#### Transaction Evidence")
        display_cols = ["Transaction ID", "Date", "Region", "Department", "Cost Category", "GL Account", "Budget USD", "Actual USD", "Variance USD", "Anomaly Flag"]
        display_cols = [c for c in display_cols if c in txn.columns]
        evidence = txn[display_cols].copy()
        for c in ["Budget USD", "Actual USD", "Variance USD"]:
            if c in evidence.columns:
                evidence[c] = evidence[c].map(money_usd)
        st.dataframe(evidence, use_container_width=True, hide_index=True)

    # --------------------------------------------------------
    # Ask the CFO — scoped natural-language routing
    # --------------------------------------------------------
    st.markdown("### 💬 Ask the CFO")
    question = st.text_input(
        "Ask a finance question",
        placeholder="Why is cost above budget in EMEA? Which department is driving Payroll variance?",
        key="ask_cfo_v18",
    )
    if question:
        q = question.lower()
        q_view = ai.copy()
        applied_filters = []
        for col, label in [("Region", "Region"), ("Country", "Country"), ("Department", "Department"), ("Cost Category", "Cost Category"), ("GL Account", "GL Account")]:
            values = sorted(ai[col].dropna().astype(str).unique(), key=len, reverse=True)
            for value in values:
                if value.lower() in q:
                    q_view = q_view[q_view[col].astype(str).str.lower() == value.lower()]
                    applied_filters.append(f"{label}={value}")
                    break

        if q_view.empty:
            st.warning("No transactions matched the dimensions detected in the question.")
        else:
            scope = ", ".join(applied_filters) if applied_filters else "current filtered view"
            if "department" in q:
                temp = q_view.groupby("Department", as_index=False).agg(Budget=("Budget USD", "sum"), Actual=("Actual USD", "sum"))
                temp["Variance"] = temp["Actual"] - temp["Budget"]
                top = temp.sort_values("Variance", ascending=False).iloc[0]
                st.success(f"Within {scope}, **{top['Department']}** is the largest unfavorable departmental variance at **{money_usd(max(float(top['Variance']),0))}**.")
            elif "region" in q or "business unit" in q:
                temp = q_view.groupby("Region", as_index=False).agg(Budget=("Budget USD", "sum"), Actual=("Actual USD", "sum"))
                temp["Variance"] = temp["Actual"] - temp["Budget"]
                top = temp.sort_values("Variance", ascending=False).iloc[0]
                st.success(f"Within {scope}, **{top['Region']}** is the largest unfavorable regional variance at **{money_usd(max(float(top['Variance']),0))}**.")
            elif "cost" in q or "category" in q or "driver" in q or "payroll" in q:
                temp = q_view.groupby("Cost Category", as_index=False).agg(Budget=("Budget USD", "sum"), Actual=("Actual USD", "sum"))
                temp["Variance"] = temp["Actual"] - temp["Budget"]
                top = temp.sort_values("Variance", ascending=False).iloc[0]
                st.success(f"Within {scope}, **{top['Cost Category']}** is the largest unfavorable cost driver at **{money_usd(max(float(top['Variance']),0))}**.")
            else:
                st.info("Ask about a region, department, cost category, GL account, budget variance, or driver.")

    st.subheader("CFO Executive Brief")
    st.info(
        f"Revenue is {money_usd(float(ai['Revenue USD'].sum()))}, actual cost is {money_usd(total_actual)}, "
        f"budget is {money_usd(total_budget)}, and unfavorable cost exposure is {money_usd(unfavorable)}. "
        f"The multi-step investigation engine is ready to trace material exceptions to transaction evidence."
    )

# ============================================================
# FINANCIAL PERFORMANCE — MANAGEMENT P&L V13
# ============================================================
elif page == "Financial Performance":
    st.subheader("Financial Performance")
    st.caption("Management P&L with actual vs budget performance, profitability drivers and regional / departmental drill-down.")

    fp = view.copy()
    fp["Month"] = pd.to_datetime(fp["Date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    fp["Cost Type"] = fp.get("Cost Type", "Opex").astype(str)

    # Revenue is allocated across the transaction-level cost rows, so total revenue
    # is the sum of Revenue USD across the full filtered view. COGS/Opex are split
    # using the explicit Cost Type field. This reconciles to the V3 management P&L.
    total_revenue = float(fp["Revenue USD"].sum())
    total_budget = float(fp["Budget USD"].sum())
    total_actual_cost = float(fp["Actual USD"].sum())
    total_cogs = float(fp.loc[fp["Cost Type"].str.upper().eq("COGS"), "Actual USD"].sum())
    total_opex = float(fp.loc[fp["Cost Type"].str.upper().eq("OPEX"), "Actual USD"].sum())
    total_budget_cogs = float(fp.loc[fp["Cost Type"].str.upper().eq("COGS"), "Budget USD"].sum())
    total_budget_opex = float(fp.loc[fp["Cost Type"].str.upper().eq("OPEX"), "Budget USD"].sum())

    gross_profit = total_revenue - total_cogs
    ebitda = total_revenue - total_cogs - total_opex

    # Management metrics are repeated at transaction level in V3. Use one value per month.
    monthly_management = None
    if "Gross Profit" in fp.columns and "EBITDA" in fp.columns:
        mg_cols = [c for c in ["Gross Profit", "EBITDA", "D&A", "EBIT", "Interest / Finance Cost", "EBT", "Tax", "PAT", "Closing Cash"] if c in fp.columns]
        if mg_cols:
            monthly_management = fp.groupby("Month", as_index=False)[mg_cols].first().sort_values("Month")

    def latest_or_sum(col, fallback):
        if monthly_management is not None and col in monthly_management.columns and not monthly_management.empty:
            return float(monthly_management[col].sum())
        return float(fallback)

    mg_gross_profit = latest_or_sum("Gross Profit", gross_profit)
    mg_ebitda = latest_or_sum("EBITDA", ebitda)
    mg_da = latest_or_sum("D&A", 0.0)
    mg_ebit = latest_or_sum("EBIT", mg_ebitda - mg_da)
    mg_interest = latest_or_sum("Interest / Finance Cost", 0.0)
    mg_ebt = latest_or_sum("EBT", mg_ebit - mg_interest)
    mg_tax = latest_or_sum("Tax", 0.0)
    mg_pat = latest_or_sum("PAT", mg_ebt - mg_tax)

    gross_margin = mg_gross_profit / total_revenue * 100 if total_revenue else 0
    ebitda_margin = mg_ebitda / total_revenue * 100 if total_revenue else 0
    pat_margin = mg_pat / total_revenue * 100 if total_revenue else 0

    # Budget P&L is supported through EBITDA because the source provides budgeted
    # revenue and operating costs, but does not provide budget D&A / tax assumptions.
    budget_revenue = total_budget
    budget_gross_profit = budget_revenue - total_budget_cogs
    budget_ebitda = budget_revenue - total_budget_cogs - total_budget_opex
    budget_gross_margin = budget_gross_profit / budget_revenue * 100 if budget_revenue else 0
    budget_ebitda_margin = budget_ebitda / budget_revenue * 100 if budget_revenue else 0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Revenue", money_usd(total_revenue))
    k2.metric("Gross Profit", money_usd(mg_gross_profit))
    k3.metric("EBITDA", money_usd(mg_ebitda))
    k4.metric("EBITDA Margin", pct(ebitda_margin))
    k5.metric("PAT", money_usd(mg_pat))
    k6.metric("PAT Margin", pct(pat_margin))

    st.markdown("### Management P&L")
    pnl = pd.DataFrame([
        ["Revenue", total_revenue, budget_revenue, total_revenue - budget_revenue, (total_revenue-budget_revenue)/budget_revenue*100 if budget_revenue else 0],
        ["COGS", total_cogs, total_budget_cogs, total_cogs - total_budget_cogs, (total_cogs-total_budget_cogs)/total_budget_cogs*100 if total_budget_cogs else 0],
        ["Gross Profit", mg_gross_profit, budget_gross_profit, mg_gross_profit-budget_gross_profit, (mg_gross_profit-budget_gross_profit)/budget_gross_profit*100 if budget_gross_profit else 0],
        ["Opex", total_opex, total_budget_opex, total_opex-total_budget_opex, (total_opex-total_budget_opex)/total_budget_opex*100 if total_budget_opex else 0],
        ["EBITDA", mg_ebitda, budget_ebitda, mg_ebitda-budget_ebitda, (mg_ebitda-budget_ebitda)/budget_ebitda*100 if budget_ebitda else 0],
        ["D&A", mg_da, np.nan, np.nan, np.nan],
        ["EBIT", mg_ebit, np.nan, np.nan, np.nan],
        ["Interest / Finance Cost", mg_interest, np.nan, np.nan, np.nan],
        ["EBT", mg_ebt, np.nan, np.nan, np.nan],
        ["Tax", mg_tax, np.nan, np.nan, np.nan],
        ["PAT", mg_pat, np.nan, np.nan, np.nan],
    ], columns=["Metric", "Actual", "Budget", "Variance", "Variance %"])

    pnl_display = pnl.copy()
    for c in ["Actual", "Budget", "Variance"]:
        pnl_display[c] = pnl[c].map(lambda x: money_usd(x) if pd.notna(x) else "N/A")
    pnl_display["Variance %"] = pnl["Variance %"].map(lambda x: pct(x) if pd.notna(x) else "N/A")
    st.dataframe(pnl_display, use_container_width=True, hide_index=True)
    st.caption("Budget comparison is available through EBITDA. The V3 source does not contain budget D&A, interest or tax assumptions, so those lines are shown as N/A rather than estimated.")

    st.markdown("### Profitability Trend")
    monthly = fp.groupby("Month", as_index=False).agg(
        Revenue=("Revenue USD", "sum"),
        Budget_Revenue=("Budget USD", "sum"),
        COGS=("Actual USD", lambda s: 0.0),
    )
    cogs_month = fp[fp["Cost Type"].str.upper().eq("COGS")].groupby("Month", as_index=False).agg(COGS=("Actual USD", "sum"))
    opex_month = fp[fp["Cost Type"].str.upper().eq("OPEX")].groupby("Month", as_index=False).agg(Opex=("Actual USD", "sum"))
    monthly = monthly.drop(columns=["COGS"]).merge(cogs_month, on="Month", how="left").merge(opex_month, on="Month", how="left")
    monthly[["COGS", "Opex"]] = monthly[["COGS", "Opex"]].fillna(0)
    monthly["Gross Profit"] = monthly["Revenue"] - monthly["COGS"]
    monthly["EBITDA"] = monthly["Gross Profit"] - monthly["Opex"]

    if monthly_management is not None:
        keep = [c for c in ["Month", "D&A", "EBIT", "PAT"] if c in monthly_management.columns]
        if len(keep) > 1:
            monthly = monthly.merge(monthly_management[keep], on="Month", how="left")
    for c in ["EBIT", "PAT"]:
        if c not in monthly.columns:
            monthly[c] = monthly["EBITDA"] if c == "EBIT" else monthly["EBITDA"] * (mg_pat / mg_ebitda if mg_ebitda else 0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly["Revenue"], mode="lines+markers", name="Revenue"))
    fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly["Gross Profit"], mode="lines+markers", name="Gross Profit"))
    fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly["EBITDA"], mode="lines+markers", name="EBITDA"))
    fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly["PAT"], mode="lines+markers", name="PAT"))
    chart_layout(fig, height=410)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### EBITDA Bridge")
    bridge = pd.DataFrame({
        "Component": ["Revenue", "COGS", "Opex", "EBITDA"],
        "Impact": [total_revenue, -total_cogs, -total_opex, mg_ebitda],
    })
    fig_bridge = go.Figure(go.Waterfall(
        name="EBITDA",
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=bridge["Component"],
        y=bridge["Impact"],
        text=[money_usd(v) for v in bridge["Impact"]],
        textposition="outside",
        connector={"line": {"width": 1}},
    ))
    fig_bridge.update_layout(template="plotly_dark", height=400, paper_bgcolor="#0b1827", plot_bgcolor="#0b1827", margin=dict(l=20,r=20,t=45,b=20), title="Revenue to EBITDA")
    st.plotly_chart(fig_bridge, use_container_width=True)

    st.markdown("### Profitability by Region")
    regional = fp.groupby("Region", as_index=False).agg(
        Revenue=("Revenue USD", "sum"),
        Budget=("Budget USD", "sum"),
    )
    reg_cogs = fp[fp["Cost Type"].str.upper().eq("COGS")].groupby("Region", as_index=False).agg(COGS=("Actual USD", "sum"))
    reg_opex = fp[fp["Cost Type"].str.upper().eq("OPEX")].groupby("Region", as_index=False).agg(Opex=("Actual USD", "sum"))
    regional = regional.merge(reg_cogs, on="Region", how="left").merge(reg_opex, on="Region", how="left").fillna(0)
    regional["Gross Profit"] = regional["Revenue"] - regional["COGS"]
    regional["EBITDA"] = regional["Gross Profit"] - regional["Opex"]
    regional["EBITDA Margin %"] = np.where(regional["Revenue"] != 0, regional["EBITDA"] / regional["Revenue"] * 100, 0)

    r1, r2 = st.columns(2)
    with r1:
        fig = px.bar(regional.sort_values("EBITDA"), x="EBITDA", y="Region", orientation="h", title="EBITDA by Region")
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
    with r2:
        fig = px.bar(regional.sort_values("EBITDA Margin %"), x="EBITDA Margin %", y="Region", orientation="h", title="EBITDA Margin by Region")
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    regional_display = regional.copy()
    for c in ["Revenue", "Budget", "COGS", "Opex", "Gross Profit", "EBITDA"]:
        regional_display[c] = regional[c].map(money_usd)
    regional_display["EBITDA Margin %"] = regional["EBITDA Margin %"].map(pct)
    st.dataframe(regional_display, use_container_width=True, hide_index=True)

    st.markdown("### Department Profitability")
    dept = fp.groupby("Department", as_index=False).agg(Revenue=("Revenue USD", "sum"))
    d_cogs = fp[fp["Cost Type"].str.upper().eq("COGS")].groupby("Department", as_index=False).agg(COGS=("Actual USD", "sum"))
    d_opex = fp[fp["Cost Type"].str.upper().eq("OPEX")].groupby("Department", as_index=False).agg(Opex=("Actual USD", "sum"))
    dept = dept.merge(d_cogs, on="Department", how="left").merge(d_opex, on="Department", how="left").fillna(0)
    dept["Gross Profit"] = dept["Revenue"] - dept["COGS"]
    dept["EBITDA"] = dept["Gross Profit"] - dept["Opex"]
    dept["EBITDA Margin %"] = np.where(dept["Revenue"] != 0, dept["EBITDA"] / dept["Revenue"] * 100, 0)
    dept_display = dept.copy()
    for c in ["Revenue", "COGS", "Opex", "Gross Profit", "EBITDA"]:
        dept_display[c] = dept[c].map(money_usd)
    dept_display["EBITDA Margin %"] = dept["EBITDA Margin %"].map(pct)
    st.dataframe(dept_display.sort_values("EBITDA", ascending=False), use_container_width=True, hide_index=True)

    st.info(
        f"CFO view: revenue is {money_usd(total_revenue)}, gross margin is {pct(gross_margin)}, "
        f"EBITDA is {money_usd(mg_ebitda)} at {pct(ebitda_margin)}, and PAT is {money_usd(mg_pat)} at {pct(pat_margin)}."
    )


# ============================================================
# CASH FLOW
# ============================================================
elif page == "Cash Flow Center":
    st.subheader("💧 Cash Flow Center")
    st.caption("CFO liquidity command center using the V3 ERP cash layer. Cash metrics are monthly management values and are not double-counted across transaction rows.")

    cash = view.copy()
    cash["Month"] = pd.to_datetime(cash["Date"]).dt.to_period("M").dt.to_timestamp()

    # V3 stores monthly cash management metrics on transaction rows. Use the
    # first record per month for those fields instead of summing repeated values.
    cash_cols = [c for c in ["Cash Inflow", "Cash Outflow", "Net Cash Flow", "Opening Cash", "Closing Cash"] if c in cash.columns]
    if not cash_cols:
        st.warning("Cash Flow Center requires cash-flow fields in the uploaded ERP dataset. The current dataset does not contain them.")
    else:
        monthly_cash = cash.sort_values(["Month", "Date"]).groupby("Month", as_index=False).agg(
            **{c: (c, "first") for c in cash_cols}
        )

        # Defensive fallback for datasets that have inflow/outflow but no net cash.
        if "Net Cash Flow" not in monthly_cash.columns and {"Cash Inflow", "Cash Outflow"}.issubset(monthly_cash.columns):
            monthly_cash["Net Cash Flow"] = monthly_cash["Cash Inflow"] - monthly_cash["Cash Outflow"]

        # Derive closing cash if it is not supplied.
        if "Closing Cash" not in monthly_cash.columns:
            opening = float(monthly_cash["Opening Cash"].iloc[0]) if "Opening Cash" in monthly_cash.columns else 0.0
            monthly_cash["Closing Cash"] = opening + monthly_cash["Net Cash Flow"].cumsum()

        latest = monthly_cash.iloc[-1]
        total_inflow = float(monthly_cash["Cash Inflow"].sum()) if "Cash Inflow" in monthly_cash.columns else 0.0
        total_outflow = float(monthly_cash["Cash Outflow"].sum()) if "Cash Outflow" in monthly_cash.columns else 0.0
        total_net = float(monthly_cash["Net Cash Flow"].sum()) if "Net Cash Flow" in monthly_cash.columns else total_inflow - total_outflow
        closing_cash = float(latest["Closing Cash"])
        avg_monthly_outflow = float(monthly_cash["Cash Outflow"].tail(3).mean()) if "Cash Outflow" in monthly_cash.columns else 0.0
        runway = closing_cash / avg_monthly_outflow if avg_monthly_outflow > 0 else np.nan
        liquidity_status = "Healthy" if closing_cash > avg_monthly_outflow * 3 else ("Watch" if closing_cash > avg_monthly_outflow * 1.5 else "Critical")

        # CFO KPI cards
        a, b, c, d, e = st.columns(5)
        a.metric("Cash Balance", money_usd(closing_cash))
        b.metric("Cash Inflow", money_usd(total_inflow))
        c.metric("Cash Outflow", money_usd(total_outflow))
        d.metric("Net Cash Flow", money_usd(total_net))
        e.metric("Liquidity Status", liquidity_status)

        st.markdown("### Liquidity Snapshot")
        x1, x2, x3 = st.columns(3)
        x1.metric("Latest Month Net Cash", money_usd(float(latest.get("Net Cash Flow", 0))))
        x2.metric("3-Month Average Outflow", money_usd(avg_monthly_outflow))
        x3.metric("Cash Runway", f"{runway:.1f} months" if np.isfinite(runway) else "N/A")

        # Monthly liquidity trend — one point per month, never double-counted.
        fig = go.Figure()
        if "Cash Inflow" in monthly_cash.columns:
            fig.add_trace(go.Scatter(x=monthly_cash["Month"], y=monthly_cash["Cash Inflow"], mode="lines+markers", name="Cash Inflow"))
        if "Cash Outflow" in monthly_cash.columns:
            fig.add_trace(go.Scatter(x=monthly_cash["Month"], y=monthly_cash["Cash Outflow"], mode="lines+markers", name="Cash Outflow"))
        fig.add_trace(go.Scatter(x=monthly_cash["Month"], y=monthly_cash["Closing Cash"], mode="lines", name="Closing Cash", yaxis="y2"))
        fig.update_layout(
            title="Monthly Cash Inflow, Outflow & Closing Cash",
            yaxis=dict(title="Monthly Flow (USD)"),
            yaxis2=dict(title="Closing Cash (USD)", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.08),
        )
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Cash Flow Overview")
        overview = monthly_cash[[c for c in ["Month", "Cash Inflow", "Cash Outflow", "Net Cash Flow", "Opening Cash", "Closing Cash"] if c in monthly_cash.columns]].copy()
        for c in overview.columns:
            if c != "Month":
                overview[c] = overview[c].map(money_usd)
        st.dataframe(overview.sort_values("Month", ascending=False), use_container_width=True, hide_index=True)

        st.markdown("### Regional Cash Generation")
        if "Region" in cash.columns:
            # Revenue and actual cost are transaction-level and therefore safe to aggregate.
            regional_cash = cash.groupby("Region", as_index=False).agg(
                Revenue=("Revenue USD", "sum"),
                Actual_Cost=("Actual USD", "sum"),
                Transactions=("Transaction ID", "count"),
            )
            regional_cash["Operating Cash Proxy"] = regional_cash["Revenue"] - regional_cash["Actual_Cost"]
            regional_cash["Cash Conversion %"] = np.where(regional_cash["Revenue"] != 0, regional_cash["Operating Cash Proxy"] / regional_cash["Revenue"] * 100, 0)
            regional_display = regional_cash.copy()
            for c in ["Revenue", "Actual_Cost", "Operating Cash Proxy"]:
                regional_display[c] = regional_display[c].map(money_usd)
            regional_display["Cash Conversion %"] = regional_display["Cash Conversion %"].map(pct)
            st.dataframe(regional_display.sort_values("Operating Cash Proxy", ascending=False), use_container_width=True, hide_index=True)

        st.markdown("### CFO Liquidity Signals")
        signals = []
        if total_net < 0:
            signals.append("🔴 Net cash flow is negative across the selected period; investigate cash outflow drivers.")
        else:
            signals.append("🟢 Net cash generation is positive across the selected period.")
        if np.isfinite(runway) and runway < 3:
            signals.append("🔴 Cash runway is below three months based on the recent average outflow.")
        elif np.isfinite(runway):
            signals.append(f"🟢 Cash runway is approximately {runway:.1f} months based on the recent average outflow.")
        if "Payment Status" in cash.columns:
            delayed = cash["Payment Status"].astype(str).str.lower().str.contains("7|15|30", regex=True).mean() * 100
            signals.append(f"ℹ️ {delayed:.1f}% of transaction rows carry a delayed payment status and should be monitored for working-capital pressure.")
        for s in signals:
            st.write(s)

        st.info("Cash Flow Center uses the V3 ERP cash layer where available. It is a synthetic/demo treasury model, not a live bank or treasury integration.")


# ============================================================
# BUDGET VS ACTUAL
# ============================================================
elif page == "Budget vs Actuals":
    st.subheader("Budget vs Actuals")
    st.caption("CFO variance engine: Budget → Actual → Variance → Root Cause → Management Action.")

    bva = view.copy()
    bva["Date"] = pd.to_datetime(bva["Date"], errors="coerce")
    bva = bva.dropna(subset=["Date"])

    # V3 Budget / Actual fields represent cost planning. Revenue does not have a
    # separate Revenue Budget field, so this module deliberately does not invent one.
    total_budget = float(pd.to_numeric(bva["Budget USD"], errors="coerce").fillna(0).sum())
    total_actual = float(pd.to_numeric(bva["Actual USD"], errors="coerce").fillna(0).sum())
    total_variance = total_actual - total_budget
    total_variance_pct = (total_variance / total_budget * 100) if total_budget else 0.0
    unfavorable = max(total_variance, 0.0)
    favorable = max(-total_variance, 0.0)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Budget", money_usd(total_budget))
    with k2:
        st.metric("Actual", money_usd(total_actual))
    with k3:
        st.metric("Net Variance", money_usd(total_variance), delta=f"{total_variance_pct:+.2f}%")
    with k4:
        st.metric("Unfavorable Exposure", money_usd(unfavorable))
    with k5:
        status = "Over Budget" if total_variance > 0 else "Within / Under Budget"
        st.metric("Control Status", status)

    if total_variance > 0:
        st.error(f"⚠️ Costs are {money_usd(total_variance)} above budget ({total_variance_pct:+.2f}%).")
    elif total_variance < 0:
        st.success(f"✅ Costs are {money_usd(abs(total_variance))} below budget ({abs(total_variance_pct):.2f}% favorable).")
    else:
        st.info("Budget and actual costs are exactly aligned for the selected period.")

    st.markdown("### Variance by Management Dimension")
    dim = st.selectbox(
        "Analyze by",
        ["Region", "Country", "Department", "Cost Category", "GL Account", "Cost Center", "Profit Center"],
        key="bva_dimension",
    )

    if dim not in bva.columns:
        st.info(f"{dim} is not present in the uploaded ERP dataset.")
    else:
        g = bva.groupby(dim, dropna=False, as_index=False).agg(
            Budget=("Budget USD", "sum"),
            Actual=("Actual USD", "sum"),
        )
        g["Variance"] = g["Actual"] - g["Budget"]
        g["Variance %"] = np.where(g["Budget"] != 0, g["Variance"] / g["Budget"] * 100, 0.0)
        g["Status"] = np.where(g["Variance"] > 0, "Unfavorable", "Favorable")
        g = g.sort_values("Variance", ascending=False)

        left, right = st.columns(2)
        with left:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=g[dim], y=g["Budget"], name="Budget"))
            fig.add_trace(go.Bar(x=g[dim], y=g["Actual"], name="Actual"))
            fig.update_layout(barmode="group", title=f"Budget vs Actual — {dim}")
            chart_layout(fig, height=390)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            var_plot = g.sort_values("Variance", ascending=True).tail(12)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=var_plot["Variance"], y=var_plot[dim], orientation="h", name="Variance"))
            fig2.update_layout(title=f"Largest Variance Drivers — {dim}", xaxis_title="Variance")
            chart_layout(fig2, height=390)
            st.plotly_chart(fig2, use_container_width=True)

        display = g.copy()
        for c in ["Budget", "Actual", "Variance"]:
            display[c] = display[c].map(money_usd)
        display["Variance %"] = g["Variance %"].map(lambda x: f"{x:+.2f}%")
        st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("### Monthly Variance Trend")
    monthly = bva.assign(Month=bva["Date"].dt.to_period("M").dt.to_timestamp()).groupby("Month", as_index=False).agg(
        Budget=("Budget USD", "sum"),
        Actual=("Actual USD", "sum"),
    )
    monthly["Variance"] = monthly["Actual"] - monthly["Budget"]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly["Month"], y=monthly["Variance"], name="Variance"))
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(title="Monthly Cost Variance", xaxis_title="Month", yaxis_title="Actual − Budget")
    chart_layout(fig, height=360)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Root-Cause Drill-down")
    st.caption("Follow the unfavorable variance from Region → Department → Cost Category → GL → Transaction.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        r_opts = ["All"] + sorted(bva["Region"].dropna().astype(str).unique().tolist()) if "Region" in bva.columns else ["All"]
        selected_region = st.selectbox("Region", r_opts, key="bva_region")
    drill = bva.copy()
    if selected_region != "All" and "Region" in drill.columns:
        drill = drill[drill["Region"].astype(str) == selected_region]

    with c2:
        d_opts = ["All"] + sorted(drill["Department"].dropna().astype(str).unique().tolist()) if "Department" in drill.columns else ["All"]
        selected_dept = st.selectbox("Department", d_opts, key="bva_department")
    if selected_dept != "All" and "Department" in drill.columns:
        drill = drill[drill["Department"].astype(str) == selected_dept]

    with c3:
        cc_opts = ["All"] + sorted(drill["Cost Category"].dropna().astype(str).unique().tolist()) if "Cost Category" in drill.columns else ["All"]
        selected_cc = st.selectbox("Cost Category", cc_opts, key="bva_cost_category")
    if selected_cc != "All" and "Cost Category" in drill.columns:
        drill = drill[drill["Cost Category"].astype(str) == selected_cc]

    with c4:
        gl_opts = ["All"] + sorted(drill["GL Account"].dropna().astype(str).unique().tolist()) if "GL Account" in drill.columns else ["All"]
        selected_gl = st.selectbox("GL Account", gl_opts, key="bva_gl")
    if selected_gl != "All" and "GL Account" in drill.columns:
        drill = drill[drill["GL Account"].astype(str) == selected_gl]

    drill_budget = float(pd.to_numeric(drill["Budget USD"], errors="coerce").fillna(0).sum())
    drill_actual = float(pd.to_numeric(drill["Actual USD"], errors="coerce").fillna(0).sum())
    drill_var = drill_actual - drill_budget
    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        st.metric("Selected Budget", money_usd(drill_budget))
    with dc2:
        st.metric("Selected Actual", money_usd(drill_actual))
    with dc3:
        st.metric("Selected Variance", money_usd(drill_var), delta="Unfavorable" if drill_var > 0 else "Favorable")

    if not drill.empty:
        tx = drill.copy()
        tx["Variance"] = pd.to_numeric(tx["Actual USD"], errors="coerce").fillna(0) - pd.to_numeric(tx["Budget USD"], errors="coerce").fillna(0)
        tx = tx.sort_values("Variance", ascending=False)
        cols = [c for c in ["Transaction ID", "Date", "Region", "Department", "Cost Category", "GL Account", "Cost Center", "Budget USD", "Actual USD", "Variance"] if c in tx.columns]
        tx_display = tx[cols].head(30).copy()
        for c in ["Budget USD", "Actual USD", "Variance"]:
            if c in tx_display.columns:
                tx_display[c] = tx_display[c].map(money_usd)
        st.dataframe(tx_display, use_container_width=True, hide_index=True)

        if drill_var > 0:
            issue = "Budget variance" + (f" — {selected_region}" if selected_region != "All" else "")
            if selected_dept != "All":
                issue += f" / {selected_dept}"
            if selected_cc != "All":
                issue += f" / {selected_cc}"
            if selected_gl != "All":
                issue += f" / {selected_gl}"
            if st.button("🎯 Create AI CFO Management Action", key="bva_create_action"):
                created = create_cfo_action(
                    "High" if drill_var > max(total_budget * 0.02, 1000000) else "Medium",
                    "Budget Control",
                    issue,
                    money_usd(drill_var) + " unfavorable",
                    "Investigate the selected variance, validate the underlying transactions, and define a corrective cost-control action.",
                    "Finance Controller",
                )
                if created:
                    st.success("Management action created in AI CFO Action Center.")
                else:
                    st.info("An open management action already exists for this issue.")


# ============================================================
# FORECASTING & PLANNING
# ============================================================
elif page == "Forecasting & Planning":
    st.subheader("Forecasting & Planning")
    st.caption("Integrated FP&A view: Actuals → Budget → Forecast → Latest Estimate, with management scenarios.")

    plan = view.copy()
    plan["Month"] = pd.to_datetime(plan["Date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    plan = plan.dropna(subset=["Month"])

    if len(plan) < 3:
        st.warning("At least three monthly periods are required for forecasting.")
    else:
        # V3 management metrics are repeated on transaction rows. Build one monthly
        # management layer without double-counting Gross Profit / EBITDA / PAT / Cash.
        monthly = plan.groupby("Month", as_index=False).agg(
            Revenue=("Revenue USD", "sum"),
            Cost_Budget=("Budget USD", "sum"),
            Actual_Cost=("Actual USD", "sum"),
        ).sort_values("Month")

        if "Cost Type" in plan.columns:
            cogs = plan.loc[plan["Cost Type"].eq("COGS")].groupby("Month")["Actual USD"].sum()
            opex = plan.loc[plan["Cost Type"].eq("Opex")].groupby("Month")["Actual USD"].sum()
            monthly["COGS"] = monthly["Month"].map(cogs).fillna(0.0)
            monthly["Opex"] = monthly["Month"].map(opex).fillna(0.0)
        else:
            monthly["COGS"] = monthly["Actual_Cost"] * 0.40
            monthly["Opex"] = monthly["Actual_Cost"] * 0.60

        monthly["Gross_Profit"] = monthly["Revenue"] - monthly["COGS"]
        monthly["EBITDA"] = monthly["Gross_Profit"] - monthly["Opex"]
        monthly["PAT"] = monthly["EBITDA"] - monthly["Revenue"] * 0.04

        if "Gross Profit" in plan.columns:
            gp = plan.groupby("Month")["Gross Profit"].first()
            monthly["Gross_Profit"] = monthly["Month"].map(gp).fillna(monthly["Gross_Profit"])
        if "EBITDA" in plan.columns:
            eb = plan.groupby("Month")["EBITDA"].first()
            monthly["EBITDA"] = monthly["Month"].map(eb).fillna(monthly["EBITDA"])
        if "PAT" in plan.columns:
            pat_s = plan.groupby("Month")["PAT"].first()
            monthly["PAT"] = monthly["Month"].map(pat_s).fillna(monthly["PAT"])
        if "Closing Cash" in plan.columns:
            cash_s = plan.groupby("Month")["Closing Cash"].first()
            monthly["Cash"] = monthly["Month"].map(cash_s)
        else:
            monthly["Cash"] = (monthly["Revenue"] - monthly["Actual_Cost"]).cumsum()

        # Planning controls.
        ctl1, ctl2, ctl3, ctl4 = st.columns(4)
        with ctl1:
            horizon = st.slider("Forecast horizon (months)", 3, 12, 6)
        with ctl2:
            revenue_adj = st.slider("Revenue assumption", -15, 20, 0, 1, format="%d%%")
        with ctl3:
            cost_adj = st.slider("Cost assumption", -15, 20, 0, 1, format="%d%%")
        with ctl4:
            scenario = st.selectbox("Planning scenario", ["Base", "Upside", "Downside", "Latest Estimate"])

        # Forecast engine: linear trend blended with recent growth trajectory.
        def forecast_series(series, horizon_value):
            clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            x = np.arange(len(clean), dtype=float).reshape(-1, 1)
            model = LinearRegression().fit(x, clean.values)
            future_x = np.arange(len(clean), len(clean) + horizon_value, dtype=float).reshape(-1, 1)
            trend = model.predict(future_x)
            growth = clean.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
            recent_growth = float(growth.tail(6).mean()) if len(growth) else 0.0
            recent_growth = float(np.clip(recent_growth, -0.15, 0.15))
            last = float(clean.iloc[-1])
            trajectory = np.array([last * ((1 + recent_growth) ** (i + 1)) for i in range(horizon_value)])
            return np.maximum(0.0, trend * 0.65 + trajectory * 0.35), recent_growth

        revenue_base, revenue_growth = forecast_series(monthly["Revenue"], horizon)
        cost_base, cost_growth = forecast_series(monthly["Actual_Cost"], horizon)

        # Management assumptions are deliberately explicit and adjustable.
        revenue_base = revenue_base * (1 + revenue_adj / 100)
        cost_base = cost_base * (1 + cost_adj / 100)

        # Preserve the latest observed gross-profit structure as the operating margin base.
        latest_gm = float(monthly["Gross_Profit"].iloc[-1] / monthly["Revenue"].iloc[-1]) if monthly["Revenue"].iloc[-1] else 0.0
        latest_ebitda_margin = float(monthly["EBITDA"].iloc[-1] / monthly["Revenue"].iloc[-1]) if monthly["Revenue"].iloc[-1] else 0.0
        latest_pat_margin = float(monthly["PAT"].iloc[-1] / monthly["Revenue"].iloc[-1]) if monthly["Revenue"].iloc[-1] else 0.0
        cogs_ratio = float(monthly["COGS"].iloc[-1] / monthly["Revenue"].iloc[-1]) if monthly["Revenue"].iloc[-1] else 0.0

        # Base operating model; scenario overlays change revenue/cost, not accounting logic.
        base_revenue = revenue_base
        base_cost = cost_base
        if scenario == "Upside":
            rev = base_revenue * 1.05
            cost = base_cost * 0.97
        elif scenario == "Downside":
            rev = base_revenue * 0.95
            cost = base_cost * 1.05
        else:
            rev = base_revenue.copy()
            cost = base_cost.copy()

        cogs_f = rev * cogs_ratio
        gross_f = rev - cogs_f
        opex_f = np.maximum(0.0, cost - cogs_f)
        ebitda_f = gross_f - opex_f
        pat_f = ebitda_f - rev * max(0.0, 0.04)

        # Latest Estimate is the explicit management case: latest actual run-rate plus assumptions.
        if scenario == "Latest Estimate":
            rev = base_revenue * (1 + revenue_growth * 0.25)
            cost = base_cost * (1 + cost_growth * 0.25)
            cogs_f = rev * cogs_ratio
            gross_f = rev - cogs_f
            opex_f = np.maximum(0.0, cost - cogs_f)
            ebitda_f = gross_f - opex_f
            pat_f = ebitda_f - rev * max(0.0, 0.04)

        future_dates = pd.date_range(
            monthly["Month"].max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
        )

        forecast = pd.DataFrame({
            "Month": future_dates,
            "Revenue": rev,
            "Cost Budget": np.repeat(float(monthly["Cost_Budget"].tail(3).mean()), horizon),
            "Forecast Cost": cost,
            "Gross Profit": gross_f,
            "EBITDA": ebitda_f,
            "PAT": pat_f,
        })
        forecast["EBITDA Margin %"] = np.where(forecast["Revenue"] != 0, forecast["EBITDA"] / forecast["Revenue"] * 100, 0)
        forecast["PAT Margin %"] = np.where(forecast["Revenue"] != 0, forecast["PAT"] / forecast["Revenue"] * 100, 0)
        forecast["Cost vs Budget"] = forecast["Forecast Cost"] - forecast["Cost Budget"]

        # KPI strip.
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Next Period Revenue", money_usd(forecast["Revenue"].iloc[0]))
        k2.metric("Forecast EBITDA", money_usd(forecast["EBITDA"].sum()))
        k3.metric("EBITDA Margin", pct(forecast["EBITDA"].sum() / forecast["Revenue"].sum() * 100 if forecast["Revenue"].sum() else 0))
        k4.metric("Forecast PAT", money_usd(forecast["PAT"].sum()))
        k5.metric("Forecast Horizon", f"{horizon} months")

        tab1, tab2, tab3 = st.tabs(["📈 Financial Outlook", "🎯 Budget vs Forecast", "🧮 Management Scenarios"])

        with tab1:
            st.subheader("Revenue & Profitability Outlook")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly["Revenue"], mode="lines+markers", name="Historical Revenue"))
            fig.add_trace(go.Scatter(x=forecast["Month"], y=forecast["Revenue"], mode="lines+markers", name=f"{scenario} Revenue"))
            fig.add_trace(go.Scatter(x=forecast["Month"], y=forecast["EBITDA"], mode="lines+markers", name="Forecast EBITDA"))
            fig.add_trace(go.Scatter(x=forecast["Month"], y=forecast["PAT"], mode="lines", name="Forecast PAT"))
            chart_layout(fig, height=390)
            fig.update_yaxes(tickprefix="$", tickformat="~s")
            st.plotly_chart(fig, use_container_width=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Recent Revenue Growth", pct(revenue_growth * 100))
            m2.metric("Current EBITDA Margin", pct(latest_ebitda_margin * 100))
            m3.metric("Current PAT Margin", pct(latest_pat_margin * 100))

        with tab2:
            st.subheader("Cost Budget vs Forecast")
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=forecast["Month"], y=forecast["Cost Budget"], name="Cost Budget"))
            fig2.add_trace(go.Bar(x=forecast["Month"], y=forecast["Forecast Cost"], name="Forecast Cost"))
            chart_layout(fig2, height=360)
            fig2.update_yaxes(tickprefix="$", tickformat="~s")
            st.plotly_chart(fig2, use_container_width=True)

            unfavorable = forecast["Cost vs Budget"].sum()
            if unfavorable > 0:
                st.warning(f"Forecast cost is {money_usd(unfavorable)} above the planning budget across the selected horizon.")
            else:
                st.success(f"Forecast cost is {money_usd(abs(unfavorable))} below the planning budget across the selected horizon.")

        with tab3:
            st.subheader("Scenario Impact")
            base_rev_total = float(base_revenue.sum())
            base_ebitda_total = float((base_revenue * (1 - cogs_ratio) - np.maximum(0.0, base_cost - base_revenue * cogs_ratio)).sum())
            scenario_table = pd.DataFrame([
                ["Base", base_rev_total, base_ebitda_total, base_ebitda_total / base_rev_total * 100 if base_rev_total else 0],
                ["Upside", float((base_revenue * 1.05).sum()), float(ebitda_f.sum()) if scenario == "Upside" else float((base_revenue * 1.05 * (1-cogs_ratio) - np.maximum(0.0, base_cost*0.97 - base_revenue*1.05*cogs_ratio)).sum()), 0],
                ["Downside", float((base_revenue * 0.95).sum()), float((base_revenue*0.95*(1-cogs_ratio) - np.maximum(0.0, base_cost*1.05 - base_revenue*0.95*cogs_ratio)).sum()), 0],
            ], columns=["Scenario", "Revenue", "EBITDA", "EBITDA Margin %"])
            scenario_table["EBITDA Margin %"] = np.where(scenario_table["Revenue"] != 0, scenario_table["EBITDA"] / scenario_table["Revenue"] * 100, 0)
            scenario_display = scenario_table.copy()
            scenario_display["Revenue"] = scenario_display["Revenue"].map(money_usd)
            scenario_display["EBITDA"] = scenario_display["EBITDA"].map(money_usd)
            scenario_display["EBITDA Margin %"] = scenario_table["EBITDA Margin %"].map(pct)
            st.dataframe(scenario_display, use_container_width=True, hide_index=True)
            st.info("Scenario assumptions are transparent planning overlays. They are not historical actuals and should be reviewed before management use.")

        display = forecast.copy()
        for c in ["Revenue", "Cost Budget", "Forecast Cost", "Gross Profit", "EBITDA", "PAT", "Cost vs Budget"]:
            display[c] = display[c].map(money_usd)
        display["EBITDA Margin %"] = forecast["EBITDA Margin %"].map(pct)
        display["PAT Margin %"] = forecast["PAT Margin %"].map(pct)
        st.subheader("Forecast Detail")
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.caption("Forecasting is a portfolio/prototype planning model based on historical ERP trends and explicit assumptions; it is not a production statistical forecast or management-approved plan.")


# ============================================================
# WHAT-IF SCENARIOS — FP&A DECISION ENGINE V10
# ============================================================
elif page == "What-if Scenarios":
    st.subheader("What-if Scenario Planner")
    st.caption("Stress-test management assumptions and see the modeled impact on Revenue, Gross Profit, EBITDA, PAT and Cash.")

    # Recalculate the baseline from the current filtered ERP view so the scenario engine
    # respects the same Region / Country / Department / Date filters as the rest of the app.
    base = view.copy()
    base_revenue = float(base["Revenue USD"].sum())
    base_actual = float(base["Actual USD"].sum())

    # Prefer the management-layer P&L as the scenario baseline when it exists.
    # V3 stores monthly Gross Profit / EBITDA / PAT values on transaction rows, so
    # use the first value per month rather than summing repeated transaction values.
    base_monthly = base.copy()
    base_monthly["Month"] = pd.to_datetime(base_monthly["Date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    if "Gross Profit" in base_monthly.columns:
        base_gross_profit = float(base_monthly.groupby("Month")["Gross Profit"].first().sum())
    else:
        if "Cost Type" in base.columns:
            base_cogs = float(base.loc[base["Cost Type"].eq("COGS"), "Actual USD"].sum())
        else:
            base_cogs = base_revenue * 0.38
        base_gross_profit = base_revenue - base_cogs

    if "EBITDA" in base_monthly.columns:
        base_ebitda = float(base_monthly.groupby("Month")["EBITDA"].first().sum())
    else:
        if "Cost Type" in base.columns:
            base_cogs = float(base.loc[base["Cost Type"].eq("COGS"), "Actual USD"].sum())
            base_opex = float(base.loc[base["Cost Type"].eq("Opex"), "Actual USD"].sum())
        else:
            base_cogs = base_revenue * 0.38
            base_opex = max(base_actual - base_cogs, 0.0)
        base_ebitda = base_gross_profit - base_opex

    # Derive the scenario cost structure from the management P&L baseline so that
    # 0% assumptions reproduce the displayed historical EBITDA exactly.
    base_cogs = max(base_revenue - base_gross_profit, 0.0)
    base_opex = max(base_gross_profit - base_ebitda, 0.0)

    if "PAT" in base_monthly.columns:
        base_pat = float(base_monthly.groupby("Month")["PAT"].first().sum())
    else:
        base_pat = base_ebitda - base_revenue * 0.04

    if "Closing Cash" in base_monthly.columns:
        cash_series = base_monthly.groupby("Month")["Closing Cash"].first().sort_index()
        base_cash = float(cash_series.iloc[-1]) if len(cash_series) else 0.0
    else:
        base_cash = float((base_revenue - base_actual))

    # Identify the cost categories that management can directly influence.
    category_col = "Cost Category" if "Cost Category" in base.columns else "Account / Cost Category"
    category_values = set(base[category_col].astype(str).str.strip()) if category_col in base.columns else set()

    st.markdown("### Management Assumptions")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        revenue_change = st.slider("Revenue change", -20, 30, 0, 1, format="%d%%", key="whatif_revenue_v10")
    with r2:
        payroll_change = st.slider("Payroll change", -20, 30, 0, 1, format="%d%%", key="whatif_payroll_v10")
    with r3:
        cloud_change = st.slider("Cloud cost change", -30, 30, 0, 1, format="%d%%", key="whatif_cloud_v10")
    with r4:
        opex_change = st.slider("Other Opex change", -30, 30, 0, 1, format="%d%%", key="whatif_opex_v10")

    # Scenario mechanics: revenue changes flow through COGS at the baseline COGS ratio;
    # controllable costs move independently. PAT keeps the baseline non-operating/tax
    # relationship by applying the historical PAT-to-EBITDA spread.
    revenue_factor = 1 + revenue_change / 100
    scenario_revenue = base_revenue * revenue_factor

    cogs_ratio = base_cogs / base_revenue if base_revenue else 0.38
    revenue_driven_cogs = scenario_revenue * cogs_ratio

    # Split controllable drivers according to their actual Cost Type. Payroll has
    # both COGS and Opex components in V3; only the Opex portion belongs in the
    # EBITDA Opex bridge. Cloud Infrastructure is COGS in V3, so its assumption
    # changes Gross Profit rather than Opex.
    payroll_mask = (
        base[category_col].astype(str).str.strip().eq("Payroll") &
        base["Cost Type"].astype(str).str.strip().eq("Opex")
    ) if category_col in base.columns and "Cost Type" in base.columns else pd.Series(False, index=base.index)
    cloud_mask = (
        base[category_col].astype(str).str.strip().eq("Cloud Infrastructure") &
        base["Cost Type"].astype(str).str.strip().eq("COGS")
    ) if category_col in base.columns and "Cost Type" in base.columns else pd.Series(False, index=base.index)

    payroll_base = float(base.loc[payroll_mask, "Actual USD"].sum())
    cloud_base = float(base.loc[cloud_mask, "Actual USD"].sum())
    other_opex_base = max(base_opex - payroll_base, 0.0)

    scenario_payroll = payroll_base * (1 + payroll_change / 100)
    scenario_cloud = cloud_base * (1 + cloud_change / 100)
    scenario_other_opex = other_opex_base * (1 + opex_change / 100)

    if "Payroll" not in category_values:
        scenario_payroll = payroll_base
    if "Cloud Infrastructure" not in category_values:
        scenario_cloud = cloud_base

    # Revenue follows the historical COGS ratio. Cloud is an incremental COGS
    # lever; Payroll and other Opex affect EBITDA below.
    revenue_driven_cogs = scenario_revenue * cogs_ratio
    cloud_delta = scenario_cloud - cloud_base
    scenario_gross_profit = scenario_revenue - revenue_driven_cogs - cloud_delta
    scenario_opex = scenario_payroll + scenario_other_opex
    scenario_ebitda = scenario_gross_profit - scenario_opex

    # Preserve the baseline EBITDA -> PAT bridge rather than inventing a new tax rate.
    baseline_non_pat = base_ebitda - base_pat
    scenario_pat = scenario_ebitda - baseline_non_pat

    # Modeled cash impact follows the incremental EBITDA impact. This is a planning
    # approximation, not a treasury forecast.
    cash_impact = scenario_ebitda - base_ebitda
    scenario_cash = base_cash + cash_impact

    scenario_ebitda_margin = scenario_ebitda / scenario_revenue * 100 if scenario_revenue else 0
    scenario_pat_margin = scenario_pat / scenario_revenue * 100 if scenario_revenue else 0
    gross_margin_scenario = scenario_gross_profit / scenario_revenue * 100 if scenario_revenue else 0

    # Compact custom cards avoid Streamlit metric text clipping in six columns.
    st.markdown("### Scenario Impact")
    card_data = [
        ("Scenario Revenue", money_usd(scenario_revenue), f"{revenue_change:+d}%"),
        ("Gross Profit", money_usd(scenario_gross_profit), f"{scenario_gross_profit - base_gross_profit:+,.0f}"),
        ("EBITDA", money_usd(scenario_ebitda), f"{scenario_ebitda - base_ebitda:+,.0f}"),
        ("EBITDA Margin", pct(scenario_ebitda_margin), f"{scenario_ebitda_margin - (base_ebitda / base_revenue * 100 if base_revenue else 0):+.1f} pts"),
        ("PAT", money_usd(scenario_pat), f"{scenario_pat - base_pat:+,.0f}"),
        ("Cash Impact", money_usd(cash_impact), f"{cash_impact:+,.0f}"),
    ]
    cols = st.columns(6)
    for col, (label, value, delta) in zip(cols, card_data):
        with col:
            html_block(f"<div class='kpi' style='min-height:112px;padding:15px 12px;overflow:hidden;'><div class='kpi-label'>{label}</div><div class='kpi-value' style='font-size:20px;white-space:nowrap;letter-spacing:-.02em;'>{value}</div><div class='kpi-note' style='font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{delta}</div></div>")

    # Waterfall-style bridge using the baseline and management levers.
    st.markdown("### EBITDA Bridge")
    bridge = pd.DataFrame({
        "Driver": [
            "Baseline EBITDA",
            "Revenue impact",
            "Payroll impact",
            "Cloud impact",
            "Other Opex impact",
            "Scenario EBITDA",
        ],
        "Impact": [
            base_ebitda,
            scenario_revenue * (1 - cogs_ratio) - base_revenue * (1 - cogs_ratio),
            -(scenario_payroll - payroll_base),
            -(scenario_cloud - cloud_base),
            -(scenario_other_opex - other_opex_base),
            scenario_ebitda,
        ],
    })
    fig = go.Figure(go.Waterfall(
        x=bridge["Driver"],
        y=bridge["Impact"],
        measure=["absolute", "relative", "relative", "relative", "relative", "total"],
        text=[money_usd(v) for v in bridge["Impact"]],
        textposition="outside",
        connector={"line": {"color": "#46627a"}},
    ))
    chart_layout(fig, height=410)
    fig.update_yaxes(tickprefix="$", tickformat="~s")
    st.plotly_chart(fig, use_container_width=True)

    # Base vs scenario comparison.
    comparison = pd.DataFrame([
        ["Revenue", base_revenue, scenario_revenue],
        ["Gross Profit", base_gross_profit, scenario_gross_profit],
        ["EBITDA", base_ebitda, scenario_ebitda],
        ["PAT", base_pat, scenario_pat],
        ["Cash Balance", base_cash, scenario_cash],
        ["Gross Margin", base_gross_profit / base_revenue * 100 if base_revenue else 0, gross_margin_scenario],
        ["EBITDA Margin", base_ebitda / base_revenue * 100 if base_revenue else 0, scenario_ebitda_margin],
        ["PAT Margin", base_pat / base_revenue * 100 if base_revenue else 0, scenario_pat_margin],
    ], columns=["Metric", "Baseline", "Scenario"])

    # Format into a dedicated string dataframe. Do not assign formatted strings
    # back into the numeric dataframe: newer pandas versions raise a TypeError
    # when a string is inserted into a float64 column.
    monetary_metrics = {"Revenue", "Gross Profit", "EBITDA", "PAT", "Cash Balance"}
    display_rows = []
    for _, row in comparison.iterrows():
        metric = row["Metric"]
        if metric in monetary_metrics:
            baseline_value = money_usd(row["Baseline"])
            scenario_value = money_usd(row["Scenario"])
        else:
            baseline_value = pct(row["Baseline"])
            scenario_value = pct(row["Scenario"])
        display_rows.append([metric, baseline_value, scenario_value])
    display = pd.DataFrame(display_rows, columns=["Metric", "Baseline", "Scenario"])

    st.markdown("### Baseline vs Scenario")
    st.dataframe(display, use_container_width=True, hide_index=True)

    # Plain-English management interpretation.
    if cash_impact > 0:
        st.success(f"Scenario improves modeled EBITDA by {money_usd(cash_impact)} and increases modeled cash by the same amount, before working-capital effects.")
    elif cash_impact < 0:
        st.warning(f"Scenario reduces modeled EBITDA by {money_usd(abs(cash_impact))} and creates an equivalent modeled cash pressure before working-capital effects.")
    else:
        st.info("Scenario is neutral versus the current baseline.")

    # Scenario-specific management guidance.
    actions = []
    if revenue_change < 0:
        actions.append("Protect high-margin revenue and review downside assumptions by region and business unit.")
    if payroll_change > 0:
        actions.append("Review workforce plan, hiring pace, contractor mix and utilization before approving the payroll increase.")
    if cloud_change > 0:
        actions.append("Review cloud consumption, commitments and unused resources before accepting higher infrastructure spend.")
    if cloud_change < 0:
        actions.append("Validate that planned cloud savings are operationally achievable and do not impair service levels.")
    if opex_change > 0:
        actions.append("Challenge discretionary Opex and require business-owner justification for incremental spend.")
    if not actions:
        actions.append("Baseline case: use the scenario outputs as the reference point for management planning.")

    st.markdown("### CFO Interpretation")
    for action in actions:
        st.write(f"• {action}")

    st.caption("What-if results are transparent planning simulations based on the current filtered ERP dataset. Cash impact is modeled from the EBITDA delta and should not be treated as a treasury forecast.")


# ============================================================
# RISK & ANOMALY DETECTION
# ============================================================
elif page == "Risk & Anomaly Detection":
    st.subheader("🛡️ Risk & Anomaly Detection")
    st.caption("Detect material financial risk, explain the drivers and route high-priority findings into the AI CFO Action Center.")

    risk = view.copy()
    required_numeric = ["Budget USD", "Actual USD", "Revenue USD", "Variance USD"]
    for col in required_numeric:
        risk[col] = pd.to_numeric(risk[col], errors="coerce").fillna(0.0)

    # Deterministic financial risk layer.
    risk["Unfavorable Variance"] = np.maximum(risk["Variance USD"], 0.0)
    risk["Variance Severity"] = risk["Variance %"].abs().clip(0, 100)
    risk["Anomaly Points"] = np.where(risk["Anomaly Flag"] == 1, 35, 0)

    # AI anomaly layer using transaction-level financial signals.
    if len(risk) >= 20:
        features = risk[["Budget USD", "Actual USD", "Revenue USD", "Variance USD"]].copy()
        features["Variance %"] = risk["Variance %"].fillna(0)
        features = features.replace([np.inf, -np.inf], 0).fillna(0)
        model = IsolationForest(contamination=0.03, random_state=42)
        prediction = model.fit_predict(features)
        risk["AI Anomaly"] = np.where(prediction == -1, "AI Flag", "Normal")
    else:
        risk["AI Anomaly"] = "Insufficient sample"

    risk["AI Points"] = np.where(risk["AI Anomaly"] == "AI Flag", 25, 0)
    risk["Magnitude Points"] = np.minimum(
        (risk["Unfavorable Variance"] / max(risk["Unfavorable Variance"].quantile(0.90), 1.0)) * 25,
        25,
    )
    risk["Risk Score"] = (
        risk["Variance Severity"] * 0.35
        + risk["Anomaly Points"]
        + risk["AI Points"]
        + risk["Magnitude Points"]
    ).clip(0, 100)

    risk["Risk Level"] = np.select(
        [risk["Risk Score"] >= 70, risk["Risk Score"] >= 40],
        ["Critical", "High"],
        default="Moderate",
    )

    financial_exposure = risk["Unfavorable Variance"].sum()
    high_risk = int((risk["Risk Score"] >= 70).sum())
    ai_flags = int((risk["AI Anomaly"] == "AI Flag").sum())
    source_flags = int(risk["Anomaly Flag"].sum())
    enterprise_score = float(risk["Risk Score"].mean()) if len(risk) else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Enterprise Risk Score", f"{enterprise_score:.0f}/100")
    k2.metric("Financial Exposure", money_usd(financial_exposure))
    k3.metric("High / Critical Risks", f"{high_risk:,}")
    k4.metric("AI / Source Flags", f"{ai_flags:,} / {source_flags:,}")

    if enterprise_score >= 70:
        st.error(f"🔴 Critical risk posture — {money_usd(financial_exposure)} of unfavorable variance exposure requires management attention.")
    elif enterprise_score >= 40:
        st.warning(f"🟠 Elevated risk posture — {money_usd(financial_exposure)} of unfavorable variance exposure should be reviewed.")
    else:
        st.success(f"🟢 Controlled risk posture — enterprise risk score is {enterprise_score:.0f}/100.")

    st.markdown("### Risk Hotspots")
    group_cols = ["Region", "Department", "Cost Category"]
    hotspot = risk.groupby(group_cols, as_index=False).agg(
        Financial_Exposure=("Unfavorable Variance", "sum"),
        Variance=("Variance USD", "sum"),
        Transactions=("Transaction ID", "count"),
        Source_Flags=("Anomaly Flag", "sum"),
        Max_Risk=("Risk Score", "max"),
    )
    hotspot["Priority"] = np.select(
        [hotspot["Max_Risk"] >= 70, hotspot["Max_Risk"] >= 40],
        ["Critical", "High"],
        default="Moderate",
    )
    hotspot = hotspot.sort_values(["Priority", "Financial_Exposure"], ascending=[True, False])

    hleft, hright = st.columns([1.25, 1])
    with hleft:
        chart = hotspot.sort_values("Financial_Exposure", ascending=False).head(12).copy()
        chart["Label"] = chart["Region"] + " · " + chart["Department"] + " · " + chart["Cost Category"]
        fig = px.bar(
            chart.sort_values("Financial_Exposure"),
            x="Financial_Exposure",
            y="Label",
            orientation="h",
            title="Top Financial Risk Exposure",
            labels={"Financial_Exposure": "Unfavorable Variance ($)"},
        )
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with hright:
        level_counts = risk["Risk Level"].value_counts().reindex(["Critical", "High", "Moderate"], fill_value=0).reset_index()
        level_counts.columns = ["Risk Level", "Transactions"]
        fig = px.bar(level_counts, x="Risk Level", y="Transactions", title="Risk Severity")
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Investigation Path")
    if not hotspot.empty:
        selected_hotspot = st.selectbox(
            "Select a risk hotspot",
            hotspot.index.tolist(),
            format_func=lambda idx: f"{hotspot.loc[idx, 'Region']} → {hotspot.loc[idx, 'Department']} → {hotspot.loc[idx, 'Cost Category']} | {money_usd(hotspot.loc[idx, 'Financial_Exposure'])} exposure",
            key="risk_hotspot_select",
        )
        selected = hotspot.loc[selected_hotspot]
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Exposure", money_usd(selected["Financial_Exposure"]))
        h2.metric("Variance", money_usd(selected["Variance"]))
        h3.metric("Transactions", f"{int(selected['Transactions']):,}")
        h4.metric("Peak Risk", f"{selected['Max_Risk']:.0f}/100")

        selected_rows = risk[
            (risk["Region"] == selected["Region"])
            & (risk["Department"] == selected["Department"])
            & (risk["Cost Category"] == selected["Cost Category"])
        ].copy()
        selected_rows = selected_rows.sort_values("Risk Score", ascending=False).head(15)

        display = selected_rows[["Transaction ID", "Date", "Region", "Department", "Cost Category", "Budget USD", "Actual USD", "Variance USD", "Variance %", "Risk Score", "Risk Level", "AI Anomaly"]].copy()
        for col in ["Budget USD", "Actual USD", "Variance USD"]:
            display[col] = display[col].map(money_usd)
        display["Variance %"] = selected_rows["Variance %"].map(pct)
        display["Risk Score"] = selected_rows["Risk Score"].map(lambda x: f"{x:.0f}/100")
        st.dataframe(display, use_container_width=True, hide_index=True)

        exposure_text = money_usd(float(selected["Financial_Exposure"]))
        issue = f"Risk hotspot: {selected['Region']} / {selected['Department']} / {selected['Cost Category']}"
        recommendation = (
            f"Investigate {selected['Cost Category']} spend in {selected['Department']} ({selected['Region']}), "
            f"review the largest unfavorable transactions and validate the underlying budget and operational drivers. "
            f"Estimated unfavorable exposure: {exposure_text}."
        )
        if st.button("🎯 Create AI CFO Management Action", key="create_risk_action"):
            created = create_cfo_action(
                "High" if selected["Max_Risk"] >= 40 else "Medium",
                "Risk & Controls",
                issue,
                f"{exposure_text} unfavorable",
                recommendation,
                "Finance Controller",
            )
            if created:
                st.success("Management action created. Open AI CFO Action Center to track it.")
            else:
                st.info("An open management action already exists for this risk hotspot.")

    st.markdown("### Risk Register")
    top = risk.sort_values("Risk Score", ascending=False).head(25).copy()
    for col in ["Budget USD", "Actual USD", "Revenue USD", "Variance USD"]:
        top[col] = top[col].map(money_usd)
    top["Variance %"] = risk.sort_values("Risk Score", ascending=False).head(25)["Variance %"].map(pct)
    top["Risk Score"] = risk.sort_values("Risk Score", ascending=False).head(25)["Risk Score"].map(lambda x: f"{x:.0f}/100")
    st.dataframe(top, use_container_width=True, hide_index=True)

    st.caption("Risk scoring combines financial variance severity, source anomaly flags, AI anomaly detection and transaction magnitude. It is a decision-support score, not a statutory or credit risk rating.")


# ============================================================
# ============================================================
# COST INTELLIGENCE
# ============================================================
elif page == "Cost Intelligence":
    st.subheader("💰 Cost Intelligence")
    st.caption("Understand cost drivers, unfavorable variance, concentration and actionable savings opportunities.")

    # Core cost view
    cost = view.groupby("Cost Category", as_index=False).agg(
        Budget=("Budget USD", "sum"),
        Actual=("Actual USD", "sum"),
        Revenue=("Revenue USD", "sum"),
        Transactions=("Transaction ID", "count"),
    )
    cost["Variance"] = cost["Actual"] - cost["Budget"]
    cost["Variance %"] = np.where(cost["Budget"] != 0, cost["Variance"] / cost["Budget"] * 100, 0)
    cost["Unfavorable"] = cost["Variance"].clip(lower=0)
    total_unfav = float(cost["Unfavorable"].sum())
    total_actual = float(cost["Actual"].sum())
    total_budget = float(cost["Budget"].sum())
    total_variance = total_actual - total_budget
    cost["Share of Actual"] = np.where(total_actual != 0, cost["Actual"] / total_actual * 100, 0)
    cost["Pareto %"] = cost.sort_values("Unfavorable", ascending=False)["Unfavorable"].cumsum() / max(total_unfav, 1) * 100

    # CFO summary cards
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Actual Cost", money_usd(total_actual))
    k2.metric("Budget", money_usd(total_budget))
    cost_variance_pct = (total_variance / total_budget * 100) if total_budget else 0.0
    # Cost variance: positive = unfavorable, negative = favorable.
    # Show enough precision so small but material variances do not display as 0.0%.
    k3.metric(
        "Net Variance",
        money_usd(total_variance),
        delta=f"{cost_variance_pct:+.2f}%",
        delta_color="inverse",
    )
    k4.metric("Unfavorable Exposure", money_usd(total_unfav))

    st.markdown("### Cost Driver Analysis")
    left, right = st.columns(2)
    with left:
        driver = cost.sort_values("Variance", ascending=True).copy()
        driver["Variance Status"] = np.where(
            driver["Variance"] > 0,
            "Unfavorable",
            np.where(driver["Variance"] < 0, "Favorable", "On Budget"),
        )
        fig = px.bar(
            driver,
            x="Variance",
            y="Cost Category",
            orientation="h",
            color="Variance Status",
            color_discrete_map={
                "Unfavorable": "#FF5C5C",
                "Favorable": "#35D07F",
                "On Budget": "#8FA3B8",
            },
            title="Budget vs Actual Variance",
            hover_data={"Variance Status": True, "Variance": ":$,.0f"},
        )
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        mix = cost.sort_values("Actual", ascending=False).copy()
        fig = px.bar(mix, x="Actual", y="Cost Category", orientation="h", title="Actual Cost by Category")
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Cost Concentration & Pareto")
    pareto = cost.sort_values("Actual", ascending=False).copy()
    pareto["Cumulative Share"] = pareto["Actual"].cumsum() / max(total_actual, 1) * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(x=pareto["Cost Category"], y=pareto["Actual"], name="Actual Cost"))
    fig.add_trace(go.Scatter(x=pareto["Cost Category"], y=pareto["Cumulative Share"], name="Cumulative %", yaxis="y2", mode="lines+markers"))
    fig.update_layout(
        title="Cost Concentration — Pareto View",
        yaxis=dict(title="Actual Cost"),
        yaxis2=dict(title="Cumulative Share %", overlaying="y", side="right", range=[0, 110]),
        legend=dict(orientation="h", y=1.08),
    )
    chart_layout(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Hotspot selector: Region -> Department -> Cost Category
    st.markdown("### Cost Hotspot Drill-down")
    h1, h2, h3 = st.columns(3)
    regions = ["All"] + sorted(view["Region"].dropna().astype(str).unique().tolist())
    with h1:
        selected_region = st.selectbox("Region", regions, key="cost_region")
    region_view = view if selected_region == "All" else view[view["Region"].astype(str) == selected_region]
    depts = ["All"] + sorted(region_view["Department"].dropna().astype(str).unique().tolist())
    with h2:
        selected_dept = st.selectbox("Department", depts, key="cost_department")
    dept_view = region_view if selected_dept == "All" else region_view[region_view["Department"].astype(str) == selected_dept]
    categories = ["All"] + sorted(dept_view["Cost Category"].dropna().astype(str).unique().tolist())
    with h3:
        selected_category = st.selectbox("Cost Category", categories, key="cost_category")
    hotspot_view = dept_view if selected_category == "All" else dept_view[dept_view["Cost Category"].astype(str) == selected_category]

    hv_budget = float(hotspot_view["Budget USD"].sum())
    hv_actual = float(hotspot_view["Actual USD"].sum())
    hv_var = hv_actual - hv_budget
    hv_txn = len(hotspot_view)
    hs1, hs2, hs3, hs4 = st.columns(4)
    hs1.metric("Selected Actual", money_usd(hv_actual))
    hs2.metric("Selected Budget", money_usd(hv_budget))
    hs3.metric("Variance", money_usd(hv_var))
    hs4.metric("Transactions", f"{hv_txn:,}")

    # Top transaction drivers
    if hv_txn:
        tx = hotspot_view[["Transaction ID", "Date", "Region", "Department", "Cost Category", "Budget USD", "Actual USD", "Variance USD"]].copy()
        tx = tx.sort_values("Variance USD", ascending=False).head(15)
        tx_display = tx.copy()
        for c in ["Budget USD", "Actual USD", "Variance USD"]:
            tx_display[c] = tx[c].map(money_usd)
        st.dataframe(tx_display, use_container_width=True, hide_index=True)

    # Savings opportunity model — conservative and transparent
    st.markdown("### Savings Opportunity")
    st.caption("Illustrative decision-support estimate: only unfavorable variance is considered, with a conservative 25% addressable assumption.")
    savings = cost[cost["Unfavorable"] > 0].copy()
    savings["Addressable Opportunity"] = savings["Unfavorable"] * 0.25
    savings = savings.sort_values("Addressable Opportunity", ascending=False)
    total_opportunity = float(savings["Addressable Opportunity"].sum())
    s1, s2 = st.columns([1, 2])
    with s1:
        st.metric("Estimated Addressable Opportunity", money_usd(total_opportunity))
        if total_actual:
            st.metric("Opportunity / Actual Cost", pct(total_opportunity / total_actual * 100))
    with s2:
        if not savings.empty:
            fig = px.bar(savings.head(8), x="Addressable Opportunity", y="Cost Category", orientation="h", title="Priority Savings Opportunities")
            chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No unfavorable cost variance detected in the current view.")

    st.markdown("### Management Actions")
    top_unfavorable = cost.sort_values("Unfavorable", ascending=False).head(5)
    for _, row in top_unfavorable.iterrows():
        if row["Unfavorable"] <= 0:
            continue
        impact = money_usd(float(row["Unfavorable"]))
        recommendation = (
            f"Investigate {row['Cost Category']} variance of {impact}; review the largest transactions, "
            f"validate operational drivers and identify a sustainable cost-control plan. "
            f"Illustrative addressable opportunity: {money_usd(float(row['Unfavorable']) * 0.25)}."
        )
        with st.expander(f"{row['Cost Category']} · {impact} unfavorable"):
            st.write(f"**Budget:** {money_usd(float(row['Budget']))}")
            st.write(f"**Actual:** {money_usd(float(row['Actual']))}")
            st.write(f"**Variance:** {impact}")
            st.write(f"**Recommended action:** {recommendation}")
            if st.button("🎯 Create AI CFO Management Action", key=f"cost_action_{row['Cost Category']}"):
                created = create_cfo_action(
                    "High" if row["Unfavorable"] > total_unfav * 0.25 else "Medium",
                    "Cost Control",
                    f"{row['Cost Category']} cost variance",
                    f"{impact} unfavorable",
                    recommendation,
                    "Finance / Cost Owner",
                )
                if created:
                    st.success("Management action created. Track it in AI CFO Action Center.")
                else:
                    st.info("An open management action already exists for this issue.")

    st.markdown("### Cost Intelligence Register")
    display = cost.sort_values("Unfavorable", ascending=False).copy()
    for c in ["Budget", "Actual", "Variance", "Unfavorable"]:
        display[c] = display[c].map(money_usd)
    display["Variance %"] = cost.sort_values("Unfavorable", ascending=False)["Variance %"].map(pct)
    display["Share of Actual"] = cost.sort_values("Unfavorable", ascending=False)["Share of Actual"].map(pct)
    display["Transactions"] = cost.sort_values("Unfavorable", ascending=False)["Transactions"].map(lambda x: f"{int(x):,}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.caption("Savings opportunities are illustrative estimates for planning and should be validated by Finance and business owners before execution.")


# AUTONOMOUS AI CFO MONITORING V17
# ============================================================
elif page == "Autonomous AI CFO":
    st.subheader("🧠 Autonomous AI CFO")
    st.caption("Proactive FP&A monitoring: Detect → Explain → Recommend → Route → Track.")
    st.info(
        "This prototype performs an on-demand CFO scan against the loaded ERP view. "
        "It can detect material exceptions, quantify financial impact, generate recommendations "
        "and route approved findings to the AI CFO Action Center. Production autonomy would add "
        "scheduled execution, live ERP feeds and governed notifications."
    )

    # Persistent scan history for the current Streamlit session.
    if "cfo_scan_history_v27" not in st.session_state:
        st.session_state["cfo_scan_history_v27"] = []

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        materiality_pct = st.slider(
            "Variance materiality", 0.5, 10.0, 2.0, 0.5,
            format="%.1f%%", key="auto_materiality_v27"
        )
    with c2:
        anomaly_trigger = st.number_input(
            "Anomaly trigger", 1, 500, 5, 1, key="auto_anomaly_v27"
        )
    with c3:
        action_mode = st.selectbox(
            "Action governance",
            ["Review before routing", "Auto-draft critical actions"],
            key="auto_action_mode_v27"
        )
    with c4:
        scan_scope = st.selectbox(
            "Scan scope",
            ["Current filtered view", "Full loaded dataset"],
            key="auto_scan_scope_v27"
        )

    run_scan = st.button("🔎 Run CFO Scan Now", type="primary", use_container_width=True)

    scan_df = df.copy() if scan_scope == "Full loaded dataset" else view.copy()
    if scan_df.empty:
        st.warning("No records are available for the selected scan scope.")
        st.stop()

    for col in ["Budget USD", "Actual USD", "Revenue USD", "Variance USD", "Anomaly Flag"]:
        if col in scan_df.columns:
            scan_df[col] = pd.to_numeric(scan_df[col], errors="coerce").fillna(0.0)

    scan_budget = float(scan_df.get("Budget USD", pd.Series(dtype=float)).sum())
    scan_actual = float(scan_df.get("Actual USD", pd.Series(dtype=float)).sum())
    scan_variance = scan_actual - scan_budget
    scan_variance_pct = abs(scan_variance) / scan_budget * 100 if scan_budget else 0.0
    scan_anomalies = int(scan_df.get("Anomaly Flag", pd.Series(dtype=float)).sum())

    findings = []

    if scan_budget and scan_variance > 0 and scan_variance_pct >= materiality_pct:
        findings.append({
            "Priority": "High" if scan_variance_pct >= materiality_pct * 2 else "Medium",
            "Signal": "Material cost variance",
            "Area": "Cost Control",
            "Impact": money_usd(scan_variance),
            "Metric": f"{scan_variance_pct:.2f}% above budget",
            "Why": "Unfavorable cost movement can compress EBITDA and reduce forecast headroom.",
            "Recommendation": "Investigate the largest unfavorable regions, departments and cost categories and reconcile the driver to forecast.",
            "Owner": "Finance Controller",
            "Action": "Review material cost variance and root causes",
        })

    if scan_anomalies >= anomaly_trigger:
        exposure = (
            float(scan_df.loc[scan_df["Anomaly Flag"] > 0, "Variance USD"].abs().sum())
            if "Variance USD" in scan_df.columns else 0.0
        )
        findings.append({
            "Priority": "High",
            "Signal": "Anomaly concentration",
            "Area": "Risk & Controls",
            "Impact": money_usd(exposure) if exposure else f"{scan_anomalies:,} transactions",
            "Metric": f"{scan_anomalies:,} flagged transactions",
            "Why": "Unusual transactions may indicate posting, classification, control or spend-management issues.",
            "Recommendation": "Prioritize the highest-value flagged transactions and document investigation outcomes.",
            "Owner": "Controller",
            "Action": "Investigate high-value anomaly transactions",
        })

    if "Region" in scan_df.columns and "Variance USD" in scan_df.columns:
        reg = scan_df.groupby("Region", as_index=False).agg(
            Budget=("Budget USD", "sum"), Variance=("Variance USD", "sum")
        )
        if not reg.empty:
            r = reg.assign(Abs=reg["Variance"].abs()).sort_values("Abs", ascending=False).iloc[0]
            rp = abs(float(r["Variance"])) / float(r["Budget"]) * 100 if float(r["Budget"]) else 0
            if float(r["Variance"]) > 0 and rp >= materiality_pct:
                findings.append({
                    "Priority": "High" if rp >= materiality_pct * 2 else "Medium",
                    "Signal": "Regional variance hotspot",
                    "Area": "Regional Performance",
                    "Impact": money_usd(float(r["Variance"])),
                    "Metric": f"{r['Region']} · {rp:.2f}% above budget",
                    "Why": "A concentrated regional variance can indicate a localized operating or planning issue.",
                    "Recommendation": f"Investigate {r['Region']} at department and cost-category level and reconcile the driver to the operating plan.",
                    "Owner": "Regional Finance",
                    "Action": f"Investigate {r['Region']} variance hotspot",
                })

    if "Cost Category" in scan_df.columns and "Variance USD" in scan_df.columns:
        cat = scan_df.groupby("Cost Category", as_index=False).agg(
            Budget=("Budget USD", "sum"), Variance=("Variance USD", "sum")
        )
        if not cat.empty:
            x = cat.assign(Abs=cat["Variance"].abs()).sort_values("Abs", ascending=False).iloc[0]
            cp = abs(float(x["Variance"])) / float(x["Budget"]) * 100 if float(x["Budget"]) else 0
            if float(x["Variance"]) > 0 and cp >= materiality_pct:
                findings.append({
                    "Priority": "High" if cp >= materiality_pct * 2 else "Medium",
                    "Signal": "Cost-category hotspot",
                    "Area": "Cost Intelligence",
                    "Impact": money_usd(float(x["Variance"])),
                    "Metric": f"{x['Cost Category']} · {cp:.2f}% above budget",
                    "Why": "A concentrated cost-category variance can affect forecast accuracy and operating margin.",
                    "Recommendation": f"Drill into {x['Cost Category']} by region, department, GL and transaction to identify the controllable driver.",
                    "Owner": "FP&A",
                    "Action": f"Investigate {x['Cost Category']} cost hotspot",
                })

    if not findings:
        findings.append({
            "Priority": "Low",
            "Signal": "No material exception",
            "Area": "Financial Health",
            "Impact": "Monitoring",
            "Metric": "Within configured thresholds",
            "Why": "No monitored signal exceeded the configured thresholds.",
            "Recommendation": "Continue routine monitoring and review the forecast as new ERP periods close.",
            "Owner": "FP&A",
            "Action": "Continue routine CFO monitoring",
        })

    high_findings = [f for f in findings if f["Priority"] == "High" and f["Signal"] != "No material exception"]
    high = sum(f["Priority"] == "High" for f in findings)
    critical = sum(
        f["Priority"] == "High" and f["Area"] in ["Cost Control", "Risk & Controls"]
        for f in findings
    )

    # Record a scan when the user explicitly runs it; also show the latest live result.
    if run_scan:
        import datetime as _dt
        st.session_state["cfo_scan_history_v27"].append({
            "Timestamp": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Scope": scan_scope,
            "Variance": scan_variance,
            "Variance %": scan_variance_pct,
            "Anomalies": scan_anomalies,
            "Signals": len(findings),
            "High Priority": high,
        })
        st.success("CFO scan completed and recorded in the monitoring history.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Signals Detected", len(findings))
    k2.metric("High Priority", high)
    k3.metric("Critical Route Candidates", critical)
    k4.metric("Anomalies in Scope", f"{scan_anomalies:,}")

    # Executive status banner.
    if high_findings:
        st.error(
            f"🚨 CFO Alert: {len(high_findings)} high-priority signal(s) require management review."
        )
    else:
        st.success("🟢 CFO Monitor: No high-priority exception is above the configured threshold.")

    st.markdown("### CFO Monitoring Feed")

    if action_mode == "Auto-draft critical actions" and high_findings:
        st.warning(
            f"{len(high_findings)} high-priority signal(s) are eligible for governed action drafting. "
            "Routing still creates trackable actions; consequential decisions remain under management control."
        )
        if st.button("⚡ Route All High-Priority Signals", key="route_all_v27"):
            routed = 0
            already_open = 0
            for f in high_findings:
                created = create_cfo_action(
                    f["Priority"], f["Area"], f["Signal"], f["Impact"],
                    f["Recommendation"], f["Owner"]
                )
                if created:
                    routed += 1
                    log_cfo_event(
                        f"Autonomous CFO routed signal: {f['Signal']}",
                        module="Autonomous AI CFO"
                    )
                else:
                    already_open += 1
            st.success(
                f"Routed {routed} signal(s) to the AI CFO Action Center."
                + (f" {already_open} already had an open action." if already_open else "")
            )
            st.rerun()

    for i, f in enumerate(findings):
        icon = "🔴" if f["Priority"] == "High" else "🟠" if f["Priority"] == "Medium" else "🟢"
        with st.expander(f"{icon} {f['Priority']} · {f['Signal']} · {f['Area']}", expanded=(i == 0)):
            a, b = st.columns(2)
            with a:
                st.write(f"**Financial impact:** {f['Impact']}")
                st.write(f"**Signal:** {f['Metric']}")
                st.write(f"**Why it matters:** {f['Why']}")
            with b:
                st.write(f"**Recommendation:** {f['Recommendation']}")
                st.write(f"**Owner:** {f['Owner']}")
                st.write(f"**Proposed action:** {f['Action']}")
            if f["Priority"] == "High":
                if st.button("🎯 Route to AI CFO Action Center", key=f"auto_route_v27_{i}"):
                    created = create_cfo_action(
                        f["Priority"], f["Area"], f["Signal"], f["Impact"],
                        f["Recommendation"], f["Owner"]
                    )
                    if created:
                        log_cfo_event(
                            f"Autonomous CFO routed signal: {f['Signal']}",
                            module="Autonomous AI CFO"
                        )
                        st.success("Signal routed to the AI CFO Action Center.")
                    else:
                        st.info("An open management action for this signal already exists.")

    st.markdown("### Monitoring History")
    history = pd.DataFrame(st.session_state["cfo_scan_history_v27"])
    if history.empty:
        st.info("No scans recorded yet. Click **Run CFO Scan Now** to create the first monitoring checkpoint.")
    else:
        st.dataframe(history.tail(20), use_container_width=True, hide_index=True)
        if len(history) >= 2:
            fig_scan = go.Figure()
            fig_scan.add_trace(go.Scatter(
                x=history["Timestamp"], y=history["Variance %"],
                mode="lines+markers", name="Variance %"
            ))
            fig_scan.add_trace(go.Scatter(
                x=history["Timestamp"], y=history["Signals"],
                mode="lines+markers", name="Signals"
            ))
            chart_layout(fig_scan, height=320)
            fig_scan.update_yaxes(title_text="Value")
            st.plotly_chart(fig_scan, use_container_width=True)

    st.markdown("### CFO Scan Summary")
    st.write(
        f"The monitored view contains **{money_usd(scan_actual)} actual cost** against "
        f"**{money_usd(scan_budget)} budget**, a net variance of "
        f"**{money_usd(scan_variance)} ({scan_variance_pct:.2f}%)**."
    )
    st.write(
        f"The scan identified **{len(findings)} management signal(s)** and "
        f"**{scan_anomalies:,} anomaly flag(s)** under the configured thresholds."
    )
    st.caption(
        "Autonomous AI CFO is a prototype decision-support layer. It performs proactive analysis "
        "when invoked; scheduled execution, live ERP connectivity and governed notifications are "
        "required for production-grade continuous autonomy."
    )


# ============================================================
# AI CFO ACTION CENTER
# ============================================================
elif page == "AI CFO Executive Brief":
    actions_for_brief = pd.DataFrame(load_cfo_actions()) if load_cfo_actions() else pd.DataFrame()
    _v20_render_cfo_brief(view, actions_for_brief)


# ============================================================
# AI CFO ACTION CENTER
# ============================================================
elif page == "AI CFO Action Center":
    st.subheader("🎯 AI CFO Action Center")
    st.caption("Turn AI CFO findings into owned, trackable management actions.")

    open_count, progress_count, resolved_count = action_status_counts()
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Open", open_count)
    a2.metric("In Progress", progress_count)
    a3.metric("Resolved", resolved_count)
    a4.metric("Total Actions", len(load_cfo_actions()))

    st.markdown("### Create Action")
    with st.form("manual_action_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            priority = st.selectbox("Priority", ["High", "Medium", "Low"])
            area = st.selectbox("Area", ["Cost Control", "Profitability", "Cash & Liquidity", "Risk & Controls", "Forecast", "Working Capital", "Other"])
        with c2:
            owner = st.text_input("Owner", value="Finance / Cost Owner")
            due_date = st.date_input("Due Date", value=(pd.Timestamp.now() + pd.Timedelta(days=14)).date())
        with c3:
            issue = st.text_input("Issue", placeholder="e.g. EMEA payroll over budget")
            impact = st.text_input("Financial Impact", placeholder="$5.2M unfavorable")
        recommendation = st.text_area("Recommended Action", placeholder="Describe the corrective action...")
        submitted = st.form_submit_button("Create Action")
        if submitted:
            if not issue.strip() or not recommendation.strip():
                st.error("Issue and Recommended Action are required.")
            else:
                created = create_cfo_action(priority, area, issue.strip(), impact.strip() or "Not quantified", recommendation.strip(), owner.strip() or "Finance / Cost Owner")
                if created:
                    set_cfo_action_due_date(load_cfo_actions()[0]["Action ID"], due_date)
                    st.success("Action created successfully.")
                else:
                    st.info("An open action with the same issue already exists.")

    st.markdown("### Action Register")
    if not load_cfo_actions():
        st.info("No management actions yet. Create one from an AI CFO finding or use the form above.")
    else:
        for i, action in enumerate(load_cfo_actions()):
            status = action.get("Status", "Open")
            priority_label = action.get("Priority", "Medium")
            icon = "🔴" if priority_label == "High" else "🟠" if priority_label == "Medium" else "🟢"
            with st.expander(f"{icon} {action['Action ID']} · {action['Area']} · {action['Issue']} · {status}", expanded=(i == 0 and status != "Resolved")):
                c1, c2 = st.columns([1.15, 1])
                with c1:
                    st.write(f"**Issue:** {action['Issue']}")
                    st.write(f"**Financial Impact:** {action['Financial Impact']}")
                    st.write(f"**Recommendation:** {action['Recommendation']}")
                with c2:
                    st.write(f"**Owner:** {action['Owner']}")
                    st.write(f"**Due Date:** {action['Due Date']}")
                    st.write(f"**Created:** {action['Created']}")
                    new_status = st.selectbox(
                        "Status",
                        ["Open", "In Progress", "Resolved"],
                        index=["Open", "In Progress", "Resolved"].index(status) if status in ["Open", "In Progress", "Resolved"] else 0,
                        key=f"status_{action['Action ID']}",
                    )
                    if st.button("Update Status", key=f"update_{action['Action ID']}"):
                        update_cfo_action_status(action["Action ID"], new_status)
                        st.success(f"Action updated to {new_status}.")
                        st.rerun()

        st.markdown("### Action Register Export")
        export_df = pd.DataFrame(load_cfo_actions())
        st.download_button(
            "Download Action Register CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="finsight_ai_cfo_action_register.csv",
            mime="text/csv",
        )

    st.markdown("### Workflow")
    st.info("Detect → Explain → Recommend → Assign → Track → Resolve. Management actions and audit events are persisted in the prototype database.")


# ============================================================
# REPORTS LIBRARY — CFO MONTHLY BUSINESS REVIEW V16
# ============================================================
elif page == "Reports Library":
    st.subheader("Reports Library")
    st.caption("Management-ready reporting pack generated from the current ERP view and AI CFO workflow.")

    # -----------------------------
    # Report controls
    # -----------------------------
    c1, c2, c3 = st.columns([1.2, 1.2, 1.6])
    with c1:
        report_type = st.selectbox(
            "Report",
            ["CFO Monthly Business Review", "Executive KPI Pack", "Budget vs Actual Pack", "Risk & Action Register"],
            key="report_type_v16",
        )
    with c2:
        report_period = st.selectbox(
            "Period",
            ["Selected Range", "Latest Month", "Latest Quarter", "YTD"],
            key="report_period_v16",
        )
    with c3:
        report_title = st.text_input("Report title", value="FinSight AI — CFO Monthly Business Review", key="report_title_v16")

    report_view = view.copy()
    report_view["Date"] = pd.to_datetime(report_view["Date"], errors="coerce")
    report_view = report_view.dropna(subset=["Date"]).copy()

    max_report_date = report_view["Date"].max()
    if pd.notna(max_report_date):
        if report_period == "Latest Month":
            report_view = report_view[report_view["Date"].dt.to_period("M") == max_report_date.to_period("M")]
        elif report_period == "Latest Quarter":
            report_view = report_view[report_view["Date"].dt.to_period("Q") == max_report_date.to_period("Q")]
        elif report_period == "YTD":
            report_view = report_view[report_view["Date"].dt.year == max_report_date.year]

    # -----------------------------
    # Core management metrics
    # -----------------------------
    revenue_r = float(report_view.get("Revenue USD", pd.Series(dtype=float)).sum())
    actual_r = float(report_view.get("Actual USD", pd.Series(dtype=float)).sum())
    budget_r = float(report_view.get("Budget USD", pd.Series(dtype=float)).sum())
    variance_r = actual_r - budget_r

    def first_metric(col, default=np.nan):
        if col not in report_view.columns:
            return default
        s = pd.to_numeric(report_view[col], errors="coerce").dropna()
        return float(s.iloc[0]) if not s.empty else default

    def latest_metric(col, default=np.nan):
        if col not in report_view.columns:
            return default
        tmp = report_view[["Date", col]].copy()
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        tmp = tmp.dropna(subset=[col]).sort_values("Date")
        return float(tmp.iloc[-1][col]) if not tmp.empty else default

    gross_profit_r = first_metric("Gross Profit", revenue_r - actual_r)
    ebitda_r = first_metric("EBITDA", np.nan)
    pat_r = first_metric("PAT", np.nan)
    cash_r = latest_metric("Closing Cash", np.nan)

    if not np.isfinite(ebitda_r):
        cogs_r = float(report_view.loc[report_view.get("Cost Type", "").astype(str).eq("COGS"), "Actual USD"].sum()) if "Cost Type" in report_view.columns else 0.0
        opex_r = float(report_view.loc[report_view.get("Cost Type", "").astype(str).eq("Opex"), "Actual USD"].sum()) if "Cost Type" in report_view.columns else 0.0
        ebitda_r = revenue_r - cogs_r - opex_r
    if not np.isfinite(pat_r):
        pat_r = ebitda_r
    if not np.isfinite(cash_r):
        cash_r = latest_metric("Closing Cash USD", np.nan)

    gross_margin_r = gross_profit_r / revenue_r * 100 if revenue_r else 0
    ebitda_margin_r = ebitda_r / revenue_r * 100 if revenue_r else 0
    pat_margin_r = pat_r / revenue_r * 100 if revenue_r else 0
    variance_pct_r = variance_r / budget_r * 100 if budget_r else 0
    anomaly_r = int(pd.to_numeric(report_view.get("Anomaly Flag", 0), errors="coerce").fillna(0).sum())
    risk_score_r = min(100, max(0, int(round(abs(variance_pct_r) * 100 + anomaly_r * 0.15))))

    # -----------------------------
    # Executive summary
    # -----------------------------
    if variance_r > 0:
        budget_comment = f"Costs are {money_usd(variance_r)} above budget ({variance_pct_r:+.2f}%)."
    elif variance_r < 0:
        budget_comment = f"Costs are {money_usd(abs(variance_r))} below budget ({variance_pct_r:+.2f}%)."
    else:
        budget_comment = "Costs are on budget for the selected reporting period."

    if ebitda_margin_r >= 20:
        profit_comment = f"Operating profitability is strong at {ebitda_margin_r:.1f}% EBITDA margin."
    elif ebitda_margin_r >= 15:
        profit_comment = f"Operating profitability is healthy at {ebitda_margin_r:.1f}% EBITDA margin."
    else:
        profit_comment = f"Operating profitability requires attention at {ebitda_margin_r:.1f}% EBITDA margin."

    risk_comment = "Risk profile is low." if risk_score_r < 15 else ("Risk profile requires monitoring." if risk_score_r < 35 else "Risk profile requires management attention.")

    # -----------------------------
    # Management P&L
    # -----------------------------
    cogs_r = float(report_view.loc[report_view["Cost Type"].astype(str).eq("COGS"), "Actual USD"].sum()) if "Cost Type" in report_view.columns else max(0.0, revenue_r - gross_profit_r)
    opex_r = float(report_view.loc[report_view["Cost Type"].astype(str).eq("Opex"), "Actual USD"].sum()) if "Cost Type" in report_view.columns else max(0.0, revenue_r - gross_profit_r - ebitda_r + 0.0)
    da_r = first_metric("D&A", 0.0)
    interest_r = first_metric("Interest / Finance Cost", 0.0)
    tax_r = first_metric("Tax", 0.0)
    ebit_r = ebitda_r - da_r
    ebt_r = ebit_r - interest_r

    pnl = pd.DataFrame([
        ["Revenue", revenue_r, "N/A", np.nan],
        ["COGS", cogs_r, budget_r, cogs_r - budget_r],
        ["Gross Profit", gross_profit_r, "N/A", np.nan],
        ["Opex", opex_r, budget_r, opex_r - budget_r],
        ["EBITDA", ebitda_r, "N/A", np.nan],
        ["D&A", da_r, "N/A", np.nan],
        ["EBIT", ebit_r, "N/A", np.nan],
        ["Interest / Finance Cost", interest_r, "N/A", np.nan],
        ["EBT", ebt_r, "N/A", np.nan],
        ["Tax", tax_r, "N/A", np.nan],
        ["PAT", pat_r, "N/A", np.nan],
    ], columns=["Metric", "Actual", "Budget", "Variance"])

    # -----------------------------
    # Variance drivers
    # -----------------------------
    driver_col = "Cost Category" if "Cost Category" in report_view.columns else "Account / Cost Category"
    drivers = report_view.groupby(driver_col, as_index=False).agg(Budget=("Budget USD", "sum"), Actual=("Actual USD", "sum"))
    drivers["Variance"] = drivers["Actual"] - drivers["Budget"]
    drivers["Variance %"] = np.where(drivers["Budget"].abs() > 0, drivers["Variance"] / drivers["Budget"].abs() * 100, 0)
    drivers = drivers.sort_values("Variance", ascending=False)
    top_drivers = drivers.head(8).copy()

    # -----------------------------
    # Regional performance
    # -----------------------------
    if "Region" in report_view.columns:
        regional = report_view.groupby("Region", as_index=False).agg(
            Revenue=("Revenue USD", "sum"),
            Actual=("Actual USD", "sum"),
            Budget=("Budget USD", "sum"),
        )
        regional["Variance"] = regional["Actual"] - regional["Budget"]
        regional["Profit"] = regional["Revenue"] - regional["Actual"]
        regional["Margin %"] = np.where(regional["Revenue"].abs() > 0, regional["Profit"] / regional["Revenue"].abs() * 100, 0)
        regional = regional.sort_values("Revenue", ascending=False)
    else:
        regional = pd.DataFrame()

    # -----------------------------
    # Actions and report catalog
    # -----------------------------
    actions_df = pd.DataFrame(load_cfo_actions()) if load_cfo_actions() else pd.DataFrame()

    st.markdown("### CFO Monthly Business Review")
    st.caption(f"{report_title} • {report_period} • Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Revenue", money_usd(revenue_r))
    k2.metric("Gross Profit", money_usd(gross_profit_r))
    k3.metric("EBITDA", money_usd(ebitda_r))
    k4.metric("PAT", money_usd(pat_r))
    k5.metric("Cash", money_usd(cash_r) if np.isfinite(cash_r) else "N/A")
    k6.metric("Risk Score", f"{risk_score_r}/100")

    st.markdown("### Executive Summary")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.info(f"**Budget:** {budget_comment}")
    with s2:
        st.success(f"**Profitability:** {profit_comment}")
    with s3:
        st.warning(f"**Risk:** {risk_comment} {anomaly_r:,} anomaly flags in scope.")

    st.markdown("### Management P&L")
    pnl_display = pnl.copy()
    for col in ["Actual", "Budget", "Variance"]:
        pnl_display[col] = pnl_display[col].apply(lambda x: money_usd(x) if isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(x) else ("N/A" if pd.isna(x) else x))
    st.dataframe(pnl_display, use_container_width=True, hide_index=True)

    st.markdown("### Top Variance Drivers")
    if not top_drivers.empty:
        driver_display = top_drivers.copy()
        for col in ["Budget", "Actual", "Variance"]:
            driver_display[col] = driver_display[col].apply(money_usd)
        driver_display["Variance %"] = driver_display["Variance %"].map(lambda x: f"{x:+.2f}%")
        st.dataframe(driver_display, use_container_width=True, hide_index=True)
    else:
        st.info("No variance drivers available.")

    st.markdown("### Regional Performance")
    if not regional.empty:
        regional_display = regional.copy()
        for col in ["Revenue", "Actual", "Budget", "Variance", "Profit"]:
            regional_display[col] = regional_display[col].apply(money_usd)
        regional_display["Margin %"] = regional_display["Margin %"].map(lambda x: f"{x:.1f}%")
        st.dataframe(regional_display, use_container_width=True, hide_index=True)

    st.markdown("### Management Actions")
    if actions_df.empty:
        st.info("No management actions have been created in the current session.")
    else:
        st.dataframe(actions_df, use_container_width=True, hide_index=True)

    # -----------------------------
    # Downloadable management pack
    # -----------------------------
    def html_table(frame):
        return frame.to_html(index=False, border=0, classes="data-table", escape=True)

    report_html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{report_title}</title>
<style>
body{{font-family:Arial,sans-serif;background:#f4f7fb;color:#172033;margin:0;padding:32px}}
.container{{max-width:1100px;margin:auto;background:white;padding:36px;border-radius:16px}}
h1{{margin-bottom:4px}} h2{{margin-top:30px;border-bottom:2px solid #d9e2ef;padding-bottom:8px}}
.meta{{color:#64748b;margin-bottom:24px}}
.kpis{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
.kpi{{background:#eef5fb;padding:18px;border-radius:10px}} .label{{font-size:12px;color:#64748b}} .value{{font-size:24px;font-weight:700;margin-top:6px}}
.summary{{padding:14px;background:#f8fafc;border-left:4px solid #3b82f6;margin:8px 0}}
.data-table{{width:100%;border-collapse:collapse;font-size:13px}} .data-table th,.data-table td{{padding:8px;border-bottom:1px solid #e2e8f0;text-align:left}} .data-table th{{background:#f1f5f9}}
.footer{{margin-top:30px;color:#64748b;font-size:12px}}
</style></head><body><div class='container'>
<h1>{report_title}</h1><div class='meta'>{report_period} • Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</div>
<div class='kpis'>
<div class='kpi'><div class='label'>Revenue</div><div class='value'>{money_usd(revenue_r)}</div></div>
<div class='kpi'><div class='label'>Gross Profit</div><div class='value'>{money_usd(gross_profit_r)}</div></div>
<div class='kpi'><div class='label'>EBITDA</div><div class='value'>{money_usd(ebitda_r)}</div></div>
<div class='kpi'><div class='label'>PAT</div><div class='value'>{money_usd(pat_r)}</div></div>
<div class='kpi'><div class='label'>Cash Balance</div><div class='value'>{money_usd(cash_r) if np.isfinite(cash_r) else 'N/A'}</div></div>
<div class='kpi'><div class='label'>Risk Score</div><div class='value'>{risk_score_r}/100</div></div>
</div>
<h2>Executive Summary</h2>
<div class='summary'><b>Budget:</b> {budget_comment}</div>
<div class='summary'><b>Profitability:</b> {profit_comment}</div>
<div class='summary'><b>Risk:</b> {risk_comment} {anomaly_r:,} anomaly flags in scope.</div>
<h2>Management P&amp;L</h2>{html_table(pnl_display)}
<h2>Top Variance Drivers</h2>{html_table(driver_display if not top_drivers.empty else pd.DataFrame({'Message':['No variance drivers available.']}))}
<h2>Regional Performance</h2>{html_table(regional_display if not regional.empty else pd.DataFrame({'Message':['No regional data available.']}))}
<h2>Management Actions</h2>{html_table(actions_df if not actions_df.empty else pd.DataFrame({'Message':['No management actions created in the current session.']}))}
<div class='footer'>FinSight AI — AI CFO / Agentic FP&amp;A prototype. USD is the group consolidation base. Cash-flow outputs may be modeled when live treasury data is unavailable.</div>
</div></body></html>"""

    st.markdown("### Report Downloads")
    st.download_button(
        "Download CFO Business Review (HTML)",
        data=report_html.encode("utf-8"),
        file_name="finsight_cfo_monthly_business_review.html",
        mime="text/html",
        key="download_cfo_report_v16",
    )

    st.download_button(
        "Download P&L (CSV)",
        data=pnl.to_csv(index=False).encode("utf-8"),
        file_name="finsight_management_pnl.csv",
        mime="text/csv",
        key="download_pnl_v16",
    )

    st.download_button(
        "Download Variance Drivers (CSV)",
        data=top_drivers.to_csv(index=False).encode("utf-8"),
        file_name="finsight_variance_drivers.csv",
        mime="text/csv",
        key="download_variance_v16",
    )

    st.markdown("### Report Catalog")
    reports = pd.DataFrame({
        "Report": [
            "CFO Monthly Business Review",
            "Executive KPI Pack",
            "Budget vs Actual Pack",
            "Regional Performance",
            "Cost Intelligence Pack",
            "Risk & Anomaly Register",
            "Forecast & Scenario Pack",
            "Management Action Register",
        ],
        "Frequency": ["Monthly", "Monthly", "Monthly", "Monthly", "Monthly", "Weekly", "Monthly", "Weekly"],
        "Status": ["Ready", "Ready", "Ready", "Ready", "Ready", "Ready", "Ready", "Ready"],
    })
    st.dataframe(reports, use_container_width=True, hide_index=True)


# ============================================================
# UPLOAD DATA
# ============================================================
elif page == "Upload Data":
    st.subheader("Upload Data")
    st.caption("Current ERP source and schema inspection.")

    st.write(f"**Rows:** {len(view):,}")
    st.write(f"**Columns:** {len(view.columns):,}")

    st.dataframe(view.head(100), use_container_width=True, hide_index=True)


# ============================================================
# ERP CONNECTIONS
# ============================================================
elif page == "ERP Connections":
    st.subheader("ERP Connections")
    st.caption("Product architecture / integration roadmap.")

    connections = pd.DataFrame({
        "System": ["SAP", "Oracle ERP", "Microsoft Dynamics", "NetSuite", "CSV / Excel"],
        "Connector": ["Roadmap", "Roadmap", "Roadmap", "Roadmap", "Available"],
        "Purpose": ["GL / P&L / AP / AR", "GL / Planning", "Finance / Operations", "SMB ERP", "Prototype ingestion"],
        "Status": ["Planned", "Planned", "Planned", "Planned", "Live"],
    })
    st.dataframe(connections, use_container_width=True, hide_index=True)


# ============================================================
# DATA MAPPING
# ============================================================
elif page == "Data Mapping":
    st.subheader("Data Mapping")
    st.caption("Canonical FinSight ERP hierarchy.")

    hierarchy = pd.DataFrame({
        "Level": range(1, 11),
        "Dimension": [
            "Region",
            "Country",
            "Legal Entity",
            "Business Unit",
            "Department",
            "Cost Center",
            "Profit Center",
            "GL Account",
            "Cost Category",
            "Transaction",
        ],
        "Purpose": [
            "Global reporting",
            "Country reporting",
            "Statutory entity",
            "Operating unit",
            "Functional ownership",
            "Cost ownership",
            "Profit ownership",
            "Accounting classification",
            "Management reporting",
            "Source transaction",
        ],
    })
    st.dataframe(hierarchy, use_container_width=True, hide_index=True)

    st.markdown("### Detected Source Fields")
    st.write(", ".join(view.columns))


# ============================================================
# DATA QUALITY & CONTROLS
# ============================================================
elif page == "Data Quality":
    st.subheader("Data Quality & Controls")
    st.caption("Pre-reporting controls across completeness, duplicates, hierarchy, dates, currency and financial integrity.")

    q = view.copy()
    n = max(len(q), 1)

    # ----------------------------
    # Control calculations
    # ----------------------------
    key_fields = [
        "Transaction ID", "Date", "Region", "Country", "Legal Entity",
        "Department", "Cost Center", "Profit Center", "GL Account",
        "Cost Category", "Currency", "Budget USD", "Actual USD", "Revenue USD"
    ]
    available_keys = [c for c in key_fields if c in q.columns]
    missing_cells = int(q[available_keys].isna().sum().sum()) if available_keys else 0
    total_key_cells = max(n * max(len(available_keys), 1), 1)
    completeness = max(0.0, 100.0 * (1 - missing_cells / total_key_cells))

    duplicate_rows = int(q.duplicated().sum())
    duplicate_ids = int(q["Transaction ID"].duplicated(keep=False).sum()) if "Transaction ID" in q.columns else 0

    dates = pd.to_datetime(q["Date"], errors="coerce") if "Date" in q.columns else pd.Series(pd.NaT, index=q.index)
    invalid_dates = int(dates.isna().sum())

    hierarchy_fields = ["Region", "Country", "Legal Entity", "Department", "Cost Center", "Profit Center", "GL Account", "Cost Category"]
    hierarchy_missing = int(q[[c for c in hierarchy_fields if c in q.columns]].isna().sum().sum()) if any(c in q.columns for c in hierarchy_fields) else 0

    # Financial integrity: Actual - Budget should reconcile to Variance USD.
    variance_mismatch = 0
    if all(c in q.columns for c in ["Actual USD", "Budget USD", "Variance USD"]):
        actual = pd.to_numeric(q["Actual USD"], errors="coerce")
        budget = pd.to_numeric(q["Budget USD"], errors="coerce")
        variance = pd.to_numeric(q["Variance USD"], errors="coerce")
        variance_mismatch = int((actual.notna() & budget.notna() & variance.notna() & ((actual - budget - variance).abs() > 0.01)).sum())

    margin_mismatch = 0
    if all(c in q.columns for c in ["Revenue USD", "Profit USD", "Margin %"]):
        revenue = pd.to_numeric(q["Revenue USD"], errors="coerce")
        profit = pd.to_numeric(q["Profit USD"], errors="coerce")
        stored_margin = pd.to_numeric(q["Margin %"], errors="coerce")
        expected_margin = np.where(revenue.abs() > 0, profit / revenue.abs() * 100, 0)
        margin_mismatch = int((np.isfinite(expected_margin) & stored_margin.notna() & (np.abs(expected_margin - stored_margin) > 0.1)).sum())

    currency_issues = 0
    if "Currency" in q.columns:
        currency_issues += int(q["Currency"].isna().sum())
    if "FX Rate to USD" in q.columns:
        fx = pd.to_numeric(q["FX Rate to USD"], errors="coerce")
        currency_issues += int((fx.isna() | (fx <= 0)).sum())

    anomaly_issues = 0
    if "Anomaly Flag" in q.columns:
        flags = pd.to_numeric(q["Anomaly Flag"], errors="coerce")
        anomaly_issues = int((flags.notna() & ~flags.isin([0, 1])).sum())

    payment_date_issues = 0
    if "Payment Due Date" in q.columns:
        payment_dates = pd.to_datetime(q["Payment Due Date"], errors="coerce")
        payment_date_issues = int((q["Payment Due Date"].notna() & payment_dates.isna()).sum())

    # Weighted score. This is a control score, not a statistical confidence score.
    duplicate_issue_rate = min(100.0, ((duplicate_rows + duplicate_ids) / max(n, 1)) * 100)
    date_issue_rate = min(100.0, invalid_dates / n * 100)
    hierarchy_issue_rate = min(100.0, hierarchy_missing / max(n * max(len([c for c in hierarchy_fields if c in q.columns]), 1), 1) * 100)
    financial_issue_rate = min(100.0, (variance_mismatch + margin_mismatch) / max(2 * n, 1) * 100)
    currency_issue_rate = min(100.0, currency_issues / n * 100)
    anomaly_issue_rate = min(100.0, anomaly_issues / n * 100)
    payment_issue_rate = min(100.0, payment_date_issues / n * 100)

    score = round(
        completeness * 0.25
        + (100 - duplicate_issue_rate) * 0.15
        + (100 - date_issue_rate) * 0.10
        + (100 - hierarchy_issue_rate) * 0.15
        + (100 - financial_issue_rate) * 0.20
        + (100 - currency_issue_rate) * 0.05
        + (100 - anomaly_issue_rate) * 0.05
        + (100 - payment_issue_rate) * 0.05,
        1,
    )

    total_exceptions = (missing_cells + duplicate_rows + invalid_dates + hierarchy_missing + variance_mismatch + margin_mismatch + currency_issues + anomaly_issues + payment_date_issues)

    status = "Excellent" if score >= 98 else "Good" if score >= 90 else "Needs Review" if score >= 75 else "Critical"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Data Quality Score", f"{score:.1f}/100")
    with c2:
        st.metric("Rows Reviewed", f"{len(q):,}")
    with c3:
        st.metric("Exceptions", f"{total_exceptions:,}")
    with c4:
        st.metric("Control Status", status)

    if score >= 98:
        st.success(f"Data Quality: {score:.1f}/100 — {status}. Dataset is ready for management reporting.")
    elif score >= 90:
        st.info(f"Data Quality: {score:.1f}/100 — {status}. Review the exceptions below before final reporting.")
    else:
        st.warning(f"Data Quality: {score:.1f}/100 — {status}. Resolve material control exceptions before relying on the output.")

    # ----------------------------
    # Control matrix
    # ----------------------------
    controls = pd.DataFrame([
        ["Completeness", missing_cells, f"{completeness:.1f}%", "Pass" if missing_cells == 0 else "Review", "Required reporting fields are populated"],
        ["Duplicate rows", duplicate_rows, "100% unique" if duplicate_rows == 0 else "Duplicates found", "Pass" if duplicate_rows == 0 else "Review", "No identical transaction rows"],
        ["Duplicate Transaction IDs", duplicate_ids, "Unique IDs" if duplicate_ids == 0 else "Repeated IDs", "Pass" if duplicate_ids == 0 else "Review", "Transaction IDs should be unique"],
        ["Date validity", invalid_dates, "Valid dates" if invalid_dates == 0 else "Invalid dates", "Pass" if invalid_dates == 0 else "Review", "Transaction dates parse correctly"],
        ["ERP hierarchy", hierarchy_missing, "Complete" if hierarchy_missing == 0 else "Missing mappings", "Pass" if hierarchy_missing == 0 else "Review", "Region → Entity → Department → GL mapping"],
        ["Variance reconciliation", variance_mismatch, "Reconciled" if variance_mismatch == 0 else "Mismatch", "Pass" if variance_mismatch == 0 else "Review", "Actual − Budget = Variance"],
        ["Margin reconciliation", margin_mismatch, "Reconciled" if margin_mismatch == 0 else "Mismatch", "Pass" if margin_mismatch == 0 else "Review", "Profit / Revenue = Margin"],
        ["Currency / FX", currency_issues, "Valid" if currency_issues == 0 else "Issues", "Pass" if currency_issues == 0 else "Review", "Currency and FX rates are usable"],
        ["Anomaly flags", anomaly_issues, "Valid" if anomaly_issues == 0 else "Invalid flags", "Pass" if anomaly_issues == 0 else "Review", "Anomaly Flag is 0/1"],
        ["Payment due dates", payment_date_issues, "Valid" if payment_date_issues == 0 else "Issues", "Pass" if payment_date_issues == 0 else "Review", "Due dates parse correctly"],
    ], columns=["Control", "Exceptions", "Result", "Status", "Control Objective"])

    st.markdown("### Control Matrix")
    st.dataframe(controls, use_container_width=True, hide_index=True)

    # ----------------------------
    # Field-level completeness
    # ----------------------------
    st.markdown("### Field Completeness")
    field_rows = []
    for col in key_fields:
        if col in q.columns:
            nulls = int(q[col].isna().sum())
            field_rows.append({
                "Field": col,
                "Rows": len(q),
                "Missing": nulls,
                "Completeness": f"{(1 - nulls / n) * 100:.1f}%",
                "Status": "Pass" if nulls == 0 else "Review",
            })
    st.dataframe(pd.DataFrame(field_rows), use_container_width=True, hide_index=True)

    # ----------------------------
    # Exceptions / remediation
    # ----------------------------
    st.markdown("### Control Guidance")
    guidance = []
    if missing_cells:
        guidance.append("Complete missing mandatory ERP fields before management reporting.")
    if duplicate_rows or duplicate_ids:
        guidance.append("Investigate duplicate transaction records or repeated transaction IDs.")
    if hierarchy_missing:
        guidance.append("Review ERP master-data mappings across Region, Entity, Department, Cost Center, Profit Center and GL.")
    if variance_mismatch:
        guidance.append("Reconcile transactions where Actual − Budget does not equal Variance USD.")
    if margin_mismatch:
        guidance.append("Reconcile stored margin percentages against Profit / Revenue.")
    if currency_issues:
        guidance.append("Validate currency codes and positive FX rates before USD consolidation.")
    if anomaly_issues:
        guidance.append("Normalize anomaly flags to 0/1 before downstream risk analytics.")
    if payment_date_issues:
        guidance.append("Correct invalid payment due dates before liquidity analysis.")
    if not guidance:
        guidance.append("No control exceptions detected in the current filtered dataset.")

    for item in guidance:
        st.write("• " + item)


# ============================================================
# ALERTS
# ============================================================
elif page == "Alerts & Notifications":
    st.subheader("Alerts & Notifications")

    alerts = []

    if variance > 0:
        alerts.append(["High", "Budget overrun", f"Actual cost is {money_usd(variance)} above budget.", "Open"])

    anomaly_count = int(view["Anomaly Flag"].sum())
    if anomaly_count:
        alerts.append(["High", "Anomaly detection", f"{anomaly_count:,} source anomalies require review.", "Open"])

    if margin < 60:
        alerts.append(["High", "Margin pressure", f"Margin is {pct(margin)}.", "Open"])

    if not alerts:
        alerts.append(["Low", "No critical alerts", "No major alert condition detected.", "Monitoring"])

    st.dataframe(
        pd.DataFrame(alerts, columns=["Priority", "Alert", "Description", "Status"]),
        use_container_width=True,
        hide_index=True,
    )



# ============================================================
# RATIO ANALYSIS
# ============================================================
elif page == "Ratio Analysis":
    st.subheader("Ratio Analysis")
    st.caption("CFO-grade profitability and operating ratios, with separate market and technical analytics.")

    # --------------------------------------------------------
    # Management metrics: V3 stores monthly P&L measures repeated
    # across transaction rows, so use one value per month.
    # --------------------------------------------------------
    ratio_df = view.copy()
    ratio_df["Month"] = pd.to_datetime(ratio_df["Date"], errors="coerce").dt.to_period("M").dt.to_timestamp()

    revenue_r = float(ratio_df["Revenue USD"].sum())
    actual_cost_r = float(ratio_df["Actual USD"].sum())

    if "Cost Type" in ratio_df.columns:
        cogs_r = float(ratio_df.loc[ratio_df["Cost Type"].eq("COGS"), "Actual USD"].sum())
        opex_r = float(ratio_df.loc[ratio_df["Cost Type"].eq("Opex"), "Actual USD"].sum())
    else:
        cogs_r = 0.0
        opex_r = actual_cost_r

    if "Gross Profit" in ratio_df.columns:
        gross_profit_r = float(ratio_df.groupby("Month")["Gross Profit"].first().sum())
    else:
        gross_profit_r = revenue_r - cogs_r

    if "EBITDA" in ratio_df.columns:
        ebitda_r = float(ratio_df.groupby("Month")["EBITDA"].first().sum())
    else:
        ebitda_r = gross_profit_r - opex_r

    if "EBIT" in ratio_df.columns:
        ebit_r = float(ratio_df.groupby("Month")["EBIT"].first().sum())
    else:
        ebit_r = ebitda_r

    if "PAT" in ratio_df.columns:
        pat_r = float(ratio_df.groupby("Month")["PAT"].first().sum())
    else:
        pat_r = ebitda_r - revenue_r * 0.04

    tabs = st.tabs([
        "Financial Ratios",
        "Liquidity & Leverage",
        "Working Capital",
        "Market Analytics",
        "Technical Analytics",
    ])

    # --------------------------------------------------------
    # FINANCIAL RATIOS
    # --------------------------------------------------------
    with tabs[0]:
        st.markdown("### Profitability & Operating Ratios")
        ratios = pd.DataFrame([
            ["Gross Margin", gross_profit_r / revenue_r * 100 if revenue_r else 0, "Gross Profit / Revenue", "ERP P&L"],
            ["EBITDA Margin", ebitda_r / revenue_r * 100 if revenue_r else 0, "EBITDA / Revenue", "ERP P&L"],
            ["EBIT Margin", ebit_r / revenue_r * 100 if revenue_r else 0, "EBIT / Revenue", "ERP P&L"],
            ["PAT Margin", pat_r / revenue_r * 100 if revenue_r else 0, "PAT / Revenue", "ERP P&L"],
            ["COGS / Revenue", cogs_r / revenue_r * 100 if revenue_r else 0, "COGS / Revenue", "ERP P&L"],
            ["Opex / Revenue", opex_r / revenue_r * 100 if revenue_r else 0, "Opex / Revenue", "ERP P&L"],
        ], columns=["Ratio", "Value", "Formula", "Data Source"])

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Gross Margin", pct(ratios.loc[0, "Value"]))
        k2.metric("EBITDA Margin", pct(ratios.loc[1, "Value"]))
        k3.metric("PAT Margin", pct(ratios.loc[3, "Value"]))
        k4.metric("Opex / Revenue", pct(ratios.loc[5, "Value"]))

        display = ratios.copy()
        display["Value"] = display["Value"].map(pct)
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.markdown("### Margin Trend")
        trend = ratio_df.groupby("Month", as_index=False).agg(Revenue=("Revenue USD", "sum"))
        if "Gross Profit" in ratio_df.columns:
            gp = ratio_df.groupby("Month")["Gross Profit"].first()
            trend["Gross Profit"] = trend["Month"].map(gp)
        else:
            trend["Gross Profit"] = trend["Revenue"] - ratio_df.loc[ratio_df["Cost Type"].eq("COGS")].groupby("Month")["Actual USD"].sum().reindex(trend["Month"]).fillna(0).values if "Cost Type" in ratio_df.columns else 0
        if "EBITDA" in ratio_df.columns:
            trend["EBITDA"] = trend["Month"].map(ratio_df.groupby("Month")["EBITDA"].first())
        else:
            trend["EBITDA"] = np.nan
        trend["Gross Margin %"] = np.where(trend["Revenue"] != 0, trend["Gross Profit"] / trend["Revenue"] * 100, 0)
        trend["EBITDA Margin %"] = np.where(trend["Revenue"] != 0, trend["EBITDA"] / trend["Revenue"] * 100, 0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend["Month"], y=trend["Gross Margin %"], mode="lines+markers", name="Gross Margin"))
        fig.add_trace(go.Scatter(x=trend["Month"], y=trend["EBITDA Margin %"], mode="lines+markers", name="EBITDA Margin"))
        chart_layout(fig, height=330)
        fig.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # LIQUIDITY & LEVERAGE
    # --------------------------------------------------------
    with tabs[1]:
        st.markdown("### Liquidity & Leverage")
        st.caption("These ratios require balance-sheet accounts. FinSight does not fabricate Current Ratio, Quick Ratio or Debt-to-Equity from P&L data.")

        cash_available = "Closing Cash" in ratio_df.columns
        interest_available = "Interest / Finance Cost" in ratio_df.columns

        if cash_available:
            cash_monthly = ratio_df.groupby("Month")["Closing Cash"].first().sort_index()
            latest_cash = float(cash_monthly.iloc[-1]) if len(cash_monthly) else 0
        else:
            latest_cash = 0

        interest_cost = float(ratio_df.groupby("Month")["Interest / Finance Cost"].first().sum()) if interest_available else 0
        interest_coverage = ebitda_r / interest_cost if interest_cost > 0 else np.nan

        lk1, lk2, lk3 = st.columns(3)
        lk1.metric("Cash Balance", money_usd(latest_cash) if cash_available else "N/A")
        lk2.metric("Interest Coverage", f"{interest_coverage:,.1f}x" if np.isfinite(interest_coverage) else "N/A")
        lk3.metric("Balance Sheet Coverage", "Partial")

        liquidity_rows = [
            ["Current Ratio", "N/A", "Current Assets / Current Liabilities", "Requires balance sheet"],
            ["Quick Ratio", "N/A", "Quick Assets / Current Liabilities", "Requires balance sheet"],
            ["Debt-to-Equity", "N/A", "Total Debt / Equity", "Requires balance sheet"],
            ["Interest Coverage", f"{interest_coverage:,.1f}x" if np.isfinite(interest_coverage) else "N/A", "EBITDA / Interest", "ERP P&L"],
            ["Cash Balance", money_usd(latest_cash) if cash_available else "N/A", "Latest Closing Cash", "ERP cash layer" if cash_available else "Unavailable"],
        ]
        st.dataframe(pd.DataFrame(liquidity_rows, columns=["Metric", "Value", "Formula", "Availability"]), use_container_width=True, hide_index=True)

        st.info("Production enhancement: connect balance-sheet and treasury data to unlock Current Ratio, Quick Ratio, Debt-to-Equity, ROA and ROE.")

    # --------------------------------------------------------
    # WORKING CAPITAL
    # --------------------------------------------------------
    with tabs[2]:
        st.markdown("### Working Capital Analytics")
        has_receivables = "Receivables" in ratio_df.columns
        has_payables = "Payables" in ratio_df.columns
        has_due = "Payment Due Date" in ratio_df.columns
        has_status = "Payment Status" in ratio_df.columns

        if has_receivables or has_payables:
            receivables = float(ratio_df["Receivables"].sum()) if has_receivables else np.nan
            payables = float(ratio_df["Payables"].sum()) if has_payables else np.nan
            st.dataframe(pd.DataFrame([
                ["Receivables", money_usd(receivables) if np.isfinite(receivables) else "N/A", "Source ERP balance"],
                ["Payables", money_usd(payables) if np.isfinite(payables) else "N/A", "Source ERP balance"],
            ], columns=["Metric", "Value", "Availability"]), use_container_width=True, hide_index=True)
        else:
            st.warning("Working-capital balances are not present in the current V3 ERP export, so DSO, DPO and Cash Conversion Cycle are not calculated.")

        if has_due and has_status:
            st.markdown("### Payment Timeliness")
            pay = ratio_df[["Payment Due Date", "Payment Status"]].copy()
            pay["Payment Due Date"] = pd.to_datetime(pay["Payment Due Date"], errors="coerce")
            status_counts = pay["Payment Status"].fillna("Unknown").value_counts().reset_index()
            status_counts.columns = ["Payment Status", "Transactions"]
            st.dataframe(status_counts, use_container_width=True, hide_index=True)
        else:
            st.info("Payment-status analytics become available when payment due date/status fields are supplied.")

        st.markdown("### Ratio Roadmap")
        roadmap = pd.DataFrame([
            ["DSO", "Receivables / Revenue × Days", "Pending balance-sheet data"],
            ["DPO", "Payables / COGS × Days", "Pending balance-sheet data"],
            ["Cash Conversion Cycle", "DSO + DIO − DPO", "Pending inventory + AP/AR data"],
        ], columns=["Metric", "Formula", "Status"])
        st.dataframe(roadmap, use_container_width=True, hide_index=True)

    # --------------------------------------------------------
    # MARKET ANALYTICS
    # --------------------------------------------------------
    def load_market_data(uploaded_market):
        if uploaded_market is not None:
            m = pd.read_csv(uploaded_market)
        else:
            m = pd.DataFrame()
        if m.empty:
            return m
        m.columns = [str(c).strip() for c in m.columns]
        rename = {}
        for c in m.columns:
            cl = c.lower().replace("_", " ")
            if cl in {"date", "datetime", "timestamp"}:
                rename[c] = "Date"
            elif cl in {"asset price", "asset", "price", "close", "asset close"}:
                rename[c] = "Asset Price"
            elif cl in {"benchmark price", "benchmark", "benchmark close", "index price"}:
                rename[c] = "Benchmark Price"
        m = m.rename(columns=rename)
        if "Date" not in m.columns or "Asset Price" not in m.columns:
            return pd.DataFrame()
        m["Date"] = pd.to_datetime(m["Date"], errors="coerce")
        m["Asset Price"] = pd.to_numeric(m["Asset Price"], errors="coerce")
        if "Benchmark Price" in m.columns:
            m["Benchmark Price"] = pd.to_numeric(m["Benchmark Price"], errors="coerce")
        return m.dropna(subset=["Date", "Asset Price"]).sort_values("Date")

    market = load_market_data(market_uploaded)

    if market.empty:
        st.markdown("### Market Analytics")
        st.info("Upload the demo market CSV or your own market price history to activate the market charts and analytics.")
        st.caption("Expected columns: Date, Asset Price, Benchmark Price. Market analytics remain separate from ERP accounting data.")
    else:
        market["Asset Return"] = market["Asset Price"].pct_change()
        has_benchmark = "Benchmark Price" in market.columns
        if has_benchmark:
            market["Benchmark Return"] = market["Benchmark Price"].pct_change()
        ret = market.dropna(subset=["Asset Return"]).copy()

        annual_factor = 252
        rf = st.number_input(
            "Annual risk-free rate (%)",
            min_value=0.0, max_value=20.0, value=5.0, step=0.25,
            key="v25_rf"
        ) / 100
        daily_rf = (1 + rf) ** (1 / annual_factor) - 1

        beta = alpha = corr = r2 = tracking_error = information_ratio = np.nan
        if has_benchmark:
            pair = ret.dropna(subset=["Benchmark Return"])
            if len(pair) > 1 and pair["Benchmark Return"].var() > 0:
                beta = pair["Asset Return"].cov(pair["Benchmark Return"]) / pair["Benchmark Return"].var()
                corr = pair["Asset Return"].corr(pair["Benchmark Return"])
                r2 = corr ** 2 if np.isfinite(corr) else np.nan
                alpha_daily = pair["Asset Return"].mean() - daily_rf - beta * (pair["Benchmark Return"].mean() - daily_rf)
                alpha = (1 + alpha_daily) ** annual_factor - 1
                active = pair["Asset Return"] - pair["Benchmark Return"]
                tracking_error = active.std() * np.sqrt(annual_factor)
                information_ratio = active.mean() / active.std() * np.sqrt(annual_factor) if active.std() > 0 else np.nan

        asset_mean = ret["Asset Return"].mean()
        asset_std = ret["Asset Return"].std()
        asset_vol = asset_std * np.sqrt(annual_factor)
        downside = ret.loc[ret["Asset Return"] < daily_rf, "Asset Return"] - daily_rf
        sharpe = ((asset_mean - daily_rf) / asset_std * np.sqrt(annual_factor)) if asset_std > 0 else np.nan
        sortino = ((asset_mean - daily_rf) / downside.std() * np.sqrt(annual_factor)) if len(downside) > 1 and downside.std() > 0 else np.nan
        treynor = ((asset_mean * annual_factor - rf) / beta) if np.isfinite(beta) and beta != 0 else np.nan
        var_95 = ret["Asset Return"].quantile(0.05)
        cvar_95 = ret.loc[ret["Asset Return"] <= var_95, "Asset Return"].mean()
        wealth = (1 + ret["Asset Return"].fillna(0)).cumprod()
        drawdown = wealth / wealth.cummax() - 1
        max_drawdown = drawdown.min()

        st.markdown("### 📈 Market Performance")
        mk1, mk2, mk3, mk4 = st.columns(4)
        mk1.metric("Alpha", f"{alpha*100:.2f}%" if np.isfinite(alpha) else "N/A")
        mk2.metric("Beta", f"{beta:.2f}" if np.isfinite(beta) else "N/A")
        mk3.metric("R²", f"{r2:.2f}" if np.isfinite(r2) else "N/A")
        mk4.metric("Sharpe", f"{sharpe:.2f}" if np.isfinite(sharpe) else "N/A")

        # Normalized performance: both series start at 100.
        perf = market[["Date", "Asset Price"]].copy()
        perf["Asset"] = perf["Asset Price"] / perf["Asset Price"].iloc[0] * 100
        if has_benchmark:
            bench = market["Benchmark Price"].dropna()
            if len(bench):
                perf["Benchmark"] = market["Benchmark Price"] / bench.iloc[0] * 100

        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(x=perf["Date"], y=perf["Asset"], mode="lines", name="Asset"))
        if has_benchmark:
            fig_perf.add_trace(go.Scatter(x=perf["Date"], y=perf["Benchmark"], mode="lines", name="Benchmark"))
        chart_layout(fig_perf, height=380)
        fig_perf.update_yaxes(title_text="Indexed performance")
        fig_perf.update_xaxes(title_text="Date")
        st.plotly_chart(fig_perf, use_container_width=True)

        # Drawdown chart.
        dd = pd.DataFrame({"Date": ret["Date"], "Drawdown": drawdown.values})
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd["Date"], y=dd["Drawdown"] * 100,
            mode="lines", name="Drawdown", fill="tozeroy"
        ))
        chart_layout(fig_dd, height=300)
        fig_dd.update_yaxes(title_text="Drawdown (%)")
        fig_dd.update_xaxes(title_text="Date")
        st.plotly_chart(fig_dd, use_container_width=True)

        market_metrics = pd.DataFrame([
            ["Alpha", f"{alpha*100:.2f}%" if np.isfinite(alpha) else "N/A", "Annualized risk-adjusted excess return vs benchmark"],
            ["Beta", f"{beta:.2f}" if np.isfinite(beta) else "N/A", "Sensitivity to benchmark"],
            ["R / Correlation", f"{corr:.2f}" if np.isfinite(corr) else "N/A", "Asset vs benchmark correlation"],
            ["R²", f"{r2:.2f}" if np.isfinite(r2) else "N/A", "Variance explained by benchmark"],
            ["Sharpe Ratio", f"{sharpe:.2f}" if np.isfinite(sharpe) else "N/A", "Return per unit of total risk"],
            ["Sortino Ratio", f"{sortino:.2f}" if np.isfinite(sortino) else "N/A", "Return per unit of downside risk"],
            ["Treynor Ratio", f"{treynor:.2f}" if np.isfinite(treynor) else "N/A", "Return per unit of systematic risk"],
            ["Information Ratio", f"{information_ratio:.2f}" if np.isfinite(information_ratio) else "N/A", "Active return / tracking error"],
            ["Annualized Volatility", f"{asset_vol*100:.2f}%", "Standard deviation of returns"],
            ["VaR 95%", f"{var_95*100:.2f}%", "One-period 95% loss threshold"],
            ["CVaR 95%", f"{cvar_95*100:.2f}%" if np.isfinite(cvar_95) else "N/A", "Average return beyond VaR"],
            ["Max Drawdown", f"{max_drawdown*100:.2f}%", "Peak-to-trough decline"],
        ], columns=["Metric", "Value", "Interpretation"])
        st.dataframe(market_metrics, use_container_width=True, hide_index=True)

    # --------------------------------------------------------
    # TECHNICAL ANALYTICS
    # --------------------------------------------------------
    with tabs[4]:
        st.markdown("### 📊 Technical Analytics")
        if market.empty:
            st.info("Upload market price history to unlock technical charts.")
        else:
            tech = market.copy().sort_values("Date")
            price = tech["Asset Price"].astype(float)

            tech["SMA 20"] = price.rolling(20).mean()
            tech["SMA 50"] = price.rolling(50).mean()

            delta = price.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            tech["RSI 14"] = 100 - (100 / (1 + rs))

            ema12 = price.ewm(span=12, adjust=False).mean()
            ema26 = price.ewm(span=26, adjust=False).mean()
            tech["MACD"] = ema12 - ema26
            tech["Signal"] = tech["MACD"].ewm(span=9, adjust=False).mean()

            mid = price.rolling(20).mean()
            std = price.rolling(20).std()
            tech["Upper BB"] = mid + 2 * std
            tech["Lower BB"] = mid - 2 * std

            high = float(price.max())
            low = float(price.min())
            diff = high - low
            fib_levels = [
                ("0.0%", high),
                ("23.6%", high - 0.236 * diff),
                ("38.2%", high - 0.382 * diff),
                ("50.0%", high - 0.500 * diff),
                ("61.8%", high - 0.618 * diff),
                ("78.6%", high - 0.786 * diff),
                ("100.0%", low),
            ]

            latest = tech.iloc[-1]
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Latest Price", f"{latest['Asset Price']:,.2f}")
            t2.metric("RSI 14", "N/A" if pd.isna(latest["RSI 14"]) else f"{latest['RSI 14']:.1f}")
            t3.metric("SMA 20", "N/A" if pd.isna(latest["SMA 20"]) else f"{latest['SMA 20']:,.2f}")
            t4.metric("SMA 50", "N/A" if pd.isna(latest["SMA 50"]) else f"{latest['SMA 50']:,.2f}")

            # Price + moving averages + Bollinger Bands.
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(x=tech["Date"], y=tech["Asset Price"], mode="lines", name="Price"))
            fig_price.add_trace(go.Scatter(x=tech["Date"], y=tech["SMA 20"], mode="lines", name="SMA 20"))
            fig_price.add_trace(go.Scatter(x=tech["Date"], y=tech["SMA 50"], mode="lines", name="SMA 50"))
            fig_price.add_trace(go.Scatter(x=tech["Date"], y=tech["Upper BB"], mode="lines", name="Upper BB"))
            fig_price.add_trace(go.Scatter(x=tech["Date"], y=tech["Lower BB"], mode="lines", name="Lower BB"))
            chart_layout(fig_price, height=420)
            fig_price.update_yaxes(title_text="Price")
            fig_price.update_xaxes(title_text="Date")
            st.plotly_chart(fig_price, use_container_width=True)

            # RSI.
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=tech["Date"], y=tech["RSI 14"], mode="lines", name="RSI 14"))
            fig_rsi.add_hline(y=70, line_dash="dash", annotation_text="Overbought 70")
            fig_rsi.add_hline(y=30, line_dash="dash", annotation_text="Oversold 30")
            chart_layout(fig_rsi, height=280)
            fig_rsi.update_yaxes(title_text="RSI", range=[0, 100])
            fig_rsi.update_xaxes(title_text="Date")
            st.plotly_chart(fig_rsi, use_container_width=True)

            # MACD.
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=tech["Date"], y=tech["MACD"], mode="lines", name="MACD"))
            fig_macd.add_trace(go.Scatter(x=tech["Date"], y=tech["Signal"], mode="lines", name="Signal"))
            chart_layout(fig_macd, height=280)
            fig_macd.update_yaxes(title_text="MACD")
            fig_macd.update_xaxes(title_text="Date")
            st.plotly_chart(fig_macd, use_container_width=True)

            # Fibonacci levels over the price series.
            fig_fib = go.Figure()
            fig_fib.add_trace(go.Scatter(x=tech["Date"], y=tech["Asset Price"], mode="lines", name="Price"))
            for level, value in fib_levels:
                fig_fib.add_hline(y=value, annotation_text=f"Fib {level}", line_dash="dot")
            chart_layout(fig_fib, height=420)
            fig_fib.update_yaxes(title_text="Price")
            fig_fib.update_xaxes(title_text="Date")
            st.plotly_chart(fig_fib, use_container_width=True)

            st.markdown("### Fibonacci Levels")
            st.dataframe(
                pd.DataFrame(fib_levels, columns=["Level", "Price"]),
                use_container_width=True,
                hide_index=True,
            )

            # Simple support/resistance reference levels from rolling extrema.
            support = float(price.rolling(20).min().iloc[-1])
            resistance = float(price.rolling(20).max().iloc[-1])
            sr1, sr2 = st.columns(2)
            sr1.metric("20-Day Support", f"{support:,.2f}")
            sr2.metric("20-Day Resistance", f"{resistance:,.2f}")

            st.caption("Technical indicators are analytical tools, not investment recommendations. The market series used here is separate from the ERP dataset.")

    # --------------------------------------------------------
    with tabs[4]:
        st.markdown("### Technical Analytics")
        if market.empty:
            st.info("Upload market price history to unlock Fibonacci, RSI, MACD, moving averages, Bollinger Bands and support/resistance.")
        else:
            tech = market[["Date", "Asset Price"]].copy().dropna().sort_values("Date")
            price = tech["Asset Price"]
            tech["SMA 20"] = price.rolling(20).mean()
            tech["SMA 50"] = price.rolling(50).mean()
            ema12 = price.ewm(span=12, adjust=False).mean()
            ema26 = price.ewm(span=26, adjust=False).mean()
            tech["MACD"] = ema12 - ema26
            tech["Signal"] = tech["MACD"].ewm(span=9, adjust=False).mean()
            delta = price.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            tech["RSI 14"] = 100 - (100 / (1 + rs))
            mid = price.rolling(20).mean()
            std = price.rolling(20).std()
            tech["Upper Band"] = mid + 2 * std
            tech["Lower Band"] = mid - 2 * std

            latest = tech.iloc[-1]
            tk1, tk2, tk3 = st.columns(3)
            tk1.metric("Latest Price", f"{latest['Asset Price']:,.2f}")
            tk2.metric("RSI 14", f"{latest['RSI 14']:.1f}" if np.isfinite(latest['RSI 14']) else "N/A")
            tk3.metric("MACD", f"{latest['MACD']:.2f}" if np.isfinite(latest['MACD']) else "N/A")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tech["Date"], y=tech["Asset Price"], mode="lines", name="Price"))
            fig.add_trace(go.Scatter(x=tech["Date"], y=tech["SMA 20"], mode="lines", name="SMA 20"))
            fig.add_trace(go.Scatter(x=tech["Date"], y=tech["SMA 50"], mode="lines", name="SMA 50"))
            fig.add_trace(go.Scatter(x=tech["Date"], y=tech["Upper Band"], mode="lines", name="Upper Band"))
            fig.add_trace(go.Scatter(x=tech["Date"], y=tech["Lower Band"], mode="lines", name="Lower Band"))
            chart_layout(fig, height=390)
            st.plotly_chart(fig, use_container_width=True)

            low = float(price.min())
            high = float(price.max())
            diff = high - low
            fib = pd.DataFrame([
                ["0.0%", high],
                ["23.6%", high - diff * 0.236],
                ["38.2%", high - diff * 0.382],
                ["50.0%", high - diff * 0.500],
                ["61.8%", high - diff * 0.618],
                ["78.6%", high - diff * 0.786],
                ["100.0%", low],
                ["161.8% Extension", low - diff * 0.618],
            ], columns=["Fibonacci Level", "Price"])
            st.markdown("### Fibonacci Retracement & Extension")
            fib["Price"] = fib["Price"].map(lambda x: f"{x:,.2f}")
            st.dataframe(fib, use_container_width=True, hide_index=True)
            st.caption("Fibonacci levels are calculated from the selected series' observed high/low range; they are technical reference levels, not guaranteed support or resistance.")

# ============================================================
# SETTINGS
# ============================================================
elif page == "Settings":
    st.subheader("Settings")
    st.caption("Prototype application configuration.")

    st.toggle("Enable AI CFO recommendations", value=True)
    st.toggle("Enable anomaly detection", value=True)
    st.toggle("Enable management alerts", value=True)
    st.selectbox("Default consolidation currency", ["USD"], index=0)
    st.selectbox("Fiscal year", ["Calendar Year", "April–March"], index=0)

    st.info("Production roadmap: authentication, role-based access, ERP connectors, audit controls, governed FX rates and live treasury integrations.")


# ============================================================
# AUDIT LOGS
# ============================================================
elif page == "Audit Logs":
    st.subheader("Audit Logs")
    st.caption("Prototype audit trail for AI CFO and management-action events.")
    if not load_cfo_audit():
        st.info("No action events recorded yet. Create or update an AI CFO action to populate the audit trail.")
    else:
        st.dataframe(pd.DataFrame(load_cfo_audit()), use_container_width=True, hide_index=True)
        st.download_button(
            "Download Audit Log CSV",
            data=pd.DataFrame(load_cfo_audit()).to_csv(index=False).encode("utf-8"),
            file_name="finsight_ai_audit_log.csv",
            mime="text/csv",
        )


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "FinSight AI is an AI CFO / agentic FP&A prototype. "
    "USD is the consolidation base; local entity currencies are retained for drill-down. "
    "Cash-flow outputs are modeled when live treasury data is unavailable."
)
