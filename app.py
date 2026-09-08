
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="FinSight AI — CFO Command Center",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# THEME
# ============================================================
st.markdown("""
<style>
.stApp {
    background: #07111d;
    color: #e8eef7;
}
[data-testid="stSidebar"] {
    background: #091522;
    border-right: 1px solid #20364d;
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}
.hero {
    background: linear-gradient(135deg,#123451,#0b1d30);
    border: 1px solid #2b506e;
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 22px;
}
.hero-title {
    font-size: 31px;
    font-weight: 800;
    color: #f4f8fc;
    margin-bottom: 7px;
}
.hero-subtitle {
    font-size: 15px;
    color: #9fc1df;
}
.pill {
    display: inline-block;
    border: 1px solid #2f607f;
    border-radius: 999px;
    padding: 5px 11px;
    margin-right: 7px;
    margin-top: 12px;
    color: #8bd8c4;
    background: #0b2635;
    font-size: 12px;
}
.kpi {
    background: linear-gradient(180deg,#102238,#0d1c2c);
    border: 1px solid #284864;
    border-radius: 16px;
    padding: 18px 19px;
    min-height: 126px;
}
.kpi-label {
    color: #91aec8;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .08em;
}
.kpi-value {
    color: #f7fafc;
    font-size: 25px;
    font-weight: 800;
    margin-top: 9px;
}
.kpi-note {
    color: #54d6b1;
    font-size: 12px;
    margin-top: 8px;
}
.section-card {
    background: #0b1827;
    border: 1px solid #20384f;
    border-radius: 16px;
    padding: 18px 20px;
}
.insight {
    background: #0d2330;
    border-left: 4px solid #41c7a5;
    padding: 13px 16px;
    border-radius: 9px;
    margin: 7px 0;
}
.warning {
    background: #2a2110;
    border-left: 4px solid #e4ad49;
    padding: 13px 16px;
    border-radius: 9px;
    margin: 7px 0;
}
.danger {
    background: #2b1519;
    border-left: 4px solid #e26a75;
    padding: 13px 16px;
    border-radius: 9px;
    margin: 7px 0;
}
.small {
    color: #8ea8bd;
    font-size: 12px;
}
div[data-testid="stMetric"] {
    background: #0d1d2d;
    border: 1px solid #20384f;
    padding: 13px;
    border-radius: 12px;
}
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
    if ax >= 1_000_000_000:
        return f"${x/1_000_000_000:,.2f}B"
    if ax >= 1_000_000:
        return f"${x/1_000_000:,.2f}M"
    if ax >= 1_000:
        return f"${x/1_000:,.1f}K"
    return f"${x:,.0f}"


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
        "Financial Performance",
        "Cash Flow Center",
        "Budget vs Actuals",
        "Forecasting & Planning",
        "What-if Scenarios",
        "Risk & Anomaly Detection",
        "Cost Intelligence",
        "Management Actions",
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
            help="Optional price history with Date, Asset Price and Benchmark Price columns for Alpha, Beta and technical analytics.",
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
      <div class="hero-subtitle">Agentic FP&A • ERP Analytics • Forecasting • Risk • Cash & Liquidity • Management Actions</div>
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
        paper_bgcolor="#0b1827",
        plot_bgcolor="#0b1827",
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================
if page == "Executive Dashboard":
    st.subheader("Executive Dashboard")
    st.caption("CFO view of revenue, profitability, cash, budget performance and enterprise risk.")

    # Management-layer fields in V3 are monthly values repeated across transaction rows.
    # Never sum those repeated columns and never take only the first transaction row.
    mgmt = view.copy()
    mgmt["Month"] = pd.to_datetime(mgmt["Date"], errors="coerce").dt.to_period("M").dt.to_timestamp()

    if "Cost Type" in mgmt.columns:
        cogs_total = float(mgmt.loc[mgmt["Cost Type"].eq("COGS"), "Actual USD"].sum())
        opex_total = float(mgmt.loc[mgmt["Cost Type"].eq("Opex"), "Actual USD"].sum())
    else:
        cogs_total = 0.0
        opex_total = float(actual)

    # V3 management measures are stored once per month (but repeated per row).
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
        cash_balance = float(mgmt.groupby("Month")["Closing Cash"].first().sort_index().iloc[-1])
    else:
        cash_balance = float((mgmt.groupby("Month")["Revenue USD"].sum() - mgmt.groupby("Month")["Actual USD"].sum()).cumsum().iloc[-1])

    gross_margin = gross_profit / revenue * 100 if revenue else 0
    ebitda_margin = ebitda / revenue * 100 if revenue else 0
    pat_margin = pat / revenue * 100 if revenue else 0

    anomaly_count = int(view["Anomaly Flag"].sum())
    variance_risk = min(abs(variance_pct) * 8, 55)
    anomaly_risk = min(anomaly_count / max(len(view), 1) * 100 * 1.8, 35)
    risk_score = min(max(variance_risk + anomaly_risk, 0), 100)

    # Six CFO-level KPIs
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

    # Monthly management view. Use first value for management metrics because
    # V3 stores monthly P&L measures on each transaction row.
    monthly_base = view.copy()
    monthly_base["Month"] = pd.to_datetime(monthly_base["Date"]).dt.to_period("M").dt.to_timestamp()

    # Build the monthly management view defensively. Some uploaded V3 files
    # may not contain every management-layer column.
    monthly = monthly_base.groupby("Month", as_index=False).agg(
        Revenue=("Revenue USD", "sum"),
        Budget=("Budget USD", "sum"),
        Actual_Cost=("Actual USD", "sum"),
    )

    if "Cost Type" in monthly_base.columns:
        cogs_month = (
            monthly_base.loc[monthly_base["Cost Type"].eq("COGS")]
            .groupby("Month")["Actual USD"].sum()
        )
        opex_month = (
            monthly_base.loc[monthly_base["Cost Type"].eq("Opex")]
            .groupby("Month")["Actual USD"].sum()
        )
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

    monthly["PAT"] = monthly["PAT"].fillna(
        monthly["EBITDA"] - monthly["Revenue"] * 0.04
    )

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
        regional = view.groupby("Region", as_index=False).agg(
            Revenue=("Revenue USD", "sum"),
            Budget=("Budget USD", "sum"),
            Actual=("Actual USD", "sum"),
        )
        regional["Variance"] = regional["Actual"] - regional["Budget"]

        # Regional EBITDA is taken from the monthly management layer and
        # allocated by each region's revenue share for a useful management view.
        regional["Revenue Share"] = regional["Revenue"] / max(regional["Revenue"].sum(), 1)
        regional["EBITDA"] = ebitda * regional["Revenue Share"]
        regional["EBITDA Margin %"] = regional["EBITDA"] / regional["Revenue"] * 100

        display = regional.drop(columns=["Revenue Share"]).copy()
        for c in ["Revenue", "Budget", "Actual", "Variance", "EBITDA"]:
            display[c] = display[c].map(money_usd)
        display["EBITDA Margin %"] = regional["EBITDA Margin %"].map(pct)
        st.dataframe(display, use_container_width=True, hide_index=True)

    cost_col, action_col = st.columns([1.15, 1])

    with cost_col:
        st.subheader("Top Cost Drivers")
        cost = view.groupby("Cost Category", as_index=False).agg(
            Budget=("Budget USD", "sum"),
            Actual=("Actual USD", "sum"),
        )
        cost["Variance"] = cost["Actual"] - cost["Budget"]
        top_cost = cost.sort_values("Variance", ascending=False).head(8)

        fig = px.bar(
            top_cost.sort_values("Variance"),
            x="Variance",
            y="Cost Category",
            orientation="h",
            title="Unfavorable Variance",
        )
        chart_layout(fig, height=330)
        fig.update_xaxes(tickprefix="$", tickformat="~s")
        st.plotly_chart(fig, use_container_width=True)

    with action_col:
        st.subheader("AI CFO Recommended Actions")

        if variance > 0:
            html_block(
                f'<div class="danger"><b>🔴 Cost Control</b><br>'
                f'Actual cost is {money_usd(variance)} above budget.<br>'
                f'<span class="small">Owner: Finance Controller · Investigate the largest cost drivers.</span></div>'
            )
        else:
            html_block(
                f'<div class="insight"><b>🟢 Cost Performance</b><br>'
                f'Actual cost is {money_usd(abs(variance))} below budget.<br>'
                f'<span class="small">Owner: FP&A · Validate sustainability of the savings.</span></div>'
            )

        if anomaly_count:
            html_block(
                f'<div class="warning"><b>🟠 Risk Investigation</b><br>'
                f'{anomaly_count:,} transactions are flagged for review.<br>'
                f'<span class="small">Owner: Controller · Prioritize high-value anomalies.</span></div>'
            )

        html_block(
            f'<div class="insight"><b>🟢 Profitability</b><br>'
            f'Gross margin is {pct(gross_margin)}, EBITDA margin is {pct(ebitda_margin)}, '
            f'and PAT margin is {pct(pat_margin)}.<br>'
            f'<span class="small">CFO focus: protect profitable growth and cash generation.</span></div>'
        )


# ============================================================
# AI CFO
# ============================================================
elif page == "AI CFO":
    st.subheader("🤖 AI CFO")
    st.caption("Decision-support prototype combining deterministic FP&A rules, anomaly signals and management actions.")

    findings = []

    if variance > 0:
        findings.append(("High", "Cost over budget", f"Actual cost exceeds budget by {money_usd(variance)}.", "Finance Controller", "Review unfavorable cost centers and approve corrective actions."))
    else:
        findings.append(("Low", "Cost under budget", f"Actual cost is favorable by {money_usd(abs(variance))}.", "FP&A", "Validate whether savings are structural or timing-related."))

    if margin < 60:
        findings.append(("High", "Margin pressure", f"Current consolidated margin is {pct(margin)}.", "CFO / FP&A", "Review pricing, delivery economics and major cost drivers."))
    elif margin < 70:
        findings.append(("Medium", "Margin watch", f"Current margin is {pct(margin)}.", "FP&A", "Monitor margin trend and high-growth cost categories."))
    else:
        findings.append(("Low", "Healthy margin", f"Current margin is {pct(margin)}.", "CFO", "Protect the current margin while funding growth."))

    anomalies = int(view["Anomaly Flag"].sum())
    if anomalies:
        findings.append(("High", "Anomaly exposure", f"{anomalies:,} transaction(s) are flagged for investigation.", "Controller", "Prioritize the highest-value anomalous transactions."))
    else:
        findings.append(("Low", "No flagged anomalies", "No source-level anomaly flags are present in the current view.", "Controller", "Continue routine monitoring."))

    for priority, title, finding, owner, action in findings:
        cls = "danger" if priority == "High" else "warning" if priority == "Medium" else "insight"
        html_block(f'<div class="{cls}"><b>{priority} · {title}</b><br>{finding}<br><span class="small">Owner: {owner} · Recommended action: {action}</span></div>')

    st.subheader("CFO Executive Brief")
    st.info(
        f"Revenue is {money_usd(revenue)}, actual cost is {money_usd(actual)}, "
        f"budget is {money_usd(budget)}, and profit margin is {pct(margin)}. "
        f"The current cost variance is {money_usd(variance)}."
    )

    st.markdown("### Ask the CFO")
    question = st.text_input(
        "Ask a finance question",
        placeholder="Why is cost above budget? Which region is driving the variance?",
    )

    if question:
        q = question.lower()
        q_view = view.copy()
        applied_filters = []

        # Detect named dimensions in the user's question and apply them
        # before calculating the requested answer. This enables chained
        # investigations such as Region -> Cost Category -> Department.
        for region_name in sorted(df["Region"].dropna().astype(str).unique(), key=len, reverse=True):
            if region_name.lower() in q:
                q_view = q_view[q_view["Region"].astype(str).str.lower() == region_name.lower()]
                applied_filters.append(f"Region={region_name}")
                break

        for country_name in sorted(df["Country"].dropna().astype(str).unique(), key=len, reverse=True):
            if country_name.lower() in q:
                q_view = q_view[q_view["Country"].astype(str).str.lower() == country_name.lower()]
                applied_filters.append(f"Country={country_name}")
                break

        for category_name in sorted(df["Cost Category"].dropna().astype(str).unique(), key=len, reverse=True):
            if category_name.lower() in q:
                q_view = q_view[q_view["Cost Category"].astype(str).str.lower() == category_name.lower()]
                applied_filters.append(f"Cost Category={category_name}")
                break

        if q_view.empty:
            st.warning("No transactions matched the dimensions detected in the question.")
        else:
            if "department" in q:
                temp = q_view.groupby("Department", as_index=False).agg(
                    Budget=("Budget USD", "sum"),
                    Actual=("Actual USD", "sum"),
                )
                temp["Variance"] = temp["Actual"] - temp["Budget"]
                top = temp.sort_values("Variance", ascending=False).iloc[0]

                scope = ", ".join(applied_filters) if applied_filters else "current filtered view"
                st.success(
                    f"Within {scope}, {top['Department']} is the largest unfavorable "
                    f"departmental variance at {money_usd(top['Variance'])}."
                )

            elif "region" in q or "business unit" in q:
                temp = q_view.groupby("Region", as_index=False).agg(
                    Budget=("Budget USD", "sum"),
                    Actual=("Actual USD", "sum"),
                )
                temp["Variance"] = temp["Actual"] - temp["Budget"]
                top = temp.sort_values("Variance", ascending=False).iloc[0]

                scope = ", ".join(applied_filters) if applied_filters else "current filtered view"
                st.success(
                    f"Within {scope}, {top['Region']} is the largest unfavorable "
                    f"regional variance at {money_usd(top['Variance'])}."
                )

            elif "cost" in q or "cost category" in q or "budget" in q or "variance" in q:
                temp = q_view.groupby("Cost Category", as_index=False).agg(
                    Budget=("Budget USD", "sum"),
                    Actual=("Actual USD", "sum"),
                )
                temp["Variance"] = temp["Actual"] - temp["Budget"]
                top = temp.sort_values("Variance", ascending=False).iloc[0]

                scope = ", ".join(applied_filters) if applied_filters else "current filtered view"
                st.success(
                    f"Within {scope}, {top['Cost Category']} is the largest unfavorable "
                    f"cost driver at {money_usd(top['Variance'])}."
                )

            else:
                st.success(
                    f"The current investigation scope has {money_usd(q_view['Revenue USD'].sum())} "
                    f"revenue, {money_usd(q_view['Actual USD'].sum())} actual cost and "
                    f"{pct(q_view['Profit USD'].sum() / q_view['Revenue USD'].sum() * 100 if q_view['Revenue USD'].sum() else 0)} margin."
                )


# ============================================================
# FINANCIAL PERFORMANCE
# ============================================================
elif page == "Financial Performance":
    st.subheader("Financial Performance")
    st.caption("Global P&L performance by region, department and cost category.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue", money_usd(revenue))
    c2.metric("Actual Cost", money_usd(actual))
    c3.metric("Profit", money_usd(profit))
    c4.metric("Margin", pct(margin))

    tab1, tab2, tab3 = st.tabs(["Region", "Department", "Cost Category"])

    for tab, field in [(tab1, "Region"), (tab2, "Department"), (tab3, "Cost Category")]:
        with tab:
            g = view.groupby(field, as_index=False).agg(
                Revenue=("Revenue USD", "sum"),
                Budget=("Budget USD", "sum"),
                Actual=("Actual USD", "sum"),
                Profit=("Profit USD", "sum"),
            )
            g["Variance"] = g["Actual"] - g["Budget"]
            g["Margin %"] = np.where(g["Revenue"] != 0, g["Profit"] / g["Revenue"] * 100, 0)

            fig = px.bar(g.sort_values("Variance"), x="Variance", y=field, orientation="h", title=f"Variance by {field}")
            chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

            display = g.copy()
            for c in ["Revenue", "Budget", "Actual", "Profit", "Variance"]:
                display[c] = display[c].map(money_usd)
            display["Margin %"] = g["Margin %"].map(pct)
            st.dataframe(display, use_container_width=True, hide_index=True)


# ============================================================
# CASH FLOW
# ============================================================
elif page == "Cash Flow Center":
    st.subheader("💧 Cash Flow Center")
    st.caption("Liquidity command center. Cash-flow fields are modeled when the ERP source does not contain treasury transactions.")

    cash = view.copy()
    cash["Month"] = pd.to_datetime(cash["Date"]).dt.to_period("M").dt.to_timestamp()

    # Model a transparent cash layer from revenue/cost/payment status.
    status = cash["Payment Status"].astype(str).str.lower() if "Payment Status" in cash.columns else pd.Series("On Time", index=cash.index)
    delay_factor = np.where(status.str.contains("30"), 0.72, np.where(status.str.contains("15"), 0.82, np.where(status.str.contains("7"), 0.92, 1.0)))

    cash["Modeled Inflow"] = cash["Revenue USD"] * delay_factor
    cash["Modeled Outflow"] = cash["Actual USD"]

    monthly_cash = cash.groupby("Month", as_index=False).agg(
        Inflow=("Modeled Inflow", "sum"),
        Outflow=("Modeled Outflow", "sum"),
    )
    monthly_cash["Net Cash Flow"] = monthly_cash["Inflow"] - monthly_cash["Outflow"]
    monthly_cash["Closing Cash"] = monthly_cash["Net Cash Flow"].cumsum()

    a, b, c, d = st.columns(4)
    a.metric("Modeled Inflow", money_usd(monthly_cash["Inflow"].sum()))
    b.metric("Modeled Outflow", money_usd(monthly_cash["Outflow"].sum()))
    c.metric("Net Cash Flow", money_usd(monthly_cash["Net Cash Flow"].sum()))
    d.metric("Closing Cash", money_usd(monthly_cash["Closing Cash"].iloc[-1]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_cash["Month"], y=monthly_cash["Inflow"], mode="lines+markers", name="Inflow"))
    fig.add_trace(go.Scatter(x=monthly_cash["Month"], y=monthly_cash["Outflow"], mode="lines+markers", name="Outflow"))
    fig.add_trace(go.Scatter(x=monthly_cash["Month"], y=monthly_cash["Closing Cash"], mode="lines", name="Closing Cash"))
    chart_layout(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.warning("Cash Flow Center is a modeled prototype unless the uploaded ERP dataset contains actual receivables, payables and cash transactions.")


# ============================================================
# BUDGET VS ACTUAL
# ============================================================
elif page == "Budget vs Actuals":
    st.subheader("Budget vs Actuals")

    field = st.selectbox("Analyze by", ["Region", "Department", "Cost Category", "GL Account", "Cost Center", "Profit Center"])

    if field not in view.columns:
        st.info(f"{field} is not present in the uploaded ERP dataset.")
    else:
        g = view.groupby(field, as_index=False).agg(
            Budget=("Budget USD", "sum"),
            Actual=("Actual USD", "sum"),
        )
        g["Variance"] = g["Actual"] - g["Budget"]
        g["Variance %"] = np.where(g["Budget"] != 0, g["Variance"] / g["Budget"] * 100, 0)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=g[field], y=g["Budget"], name="Budget"))
        fig.add_trace(go.Bar(x=g[field], y=g["Actual"], name="Actual"))
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

        display = g.copy()
        for c in ["Budget", "Actual", "Variance"]:
            display[c] = display[c].map(money_usd)
        display["Variance %"] = g["Variance %"].map(pct)
        st.dataframe(display, use_container_width=True, hide_index=True)


# ============================================================
# FORECASTING
# ============================================================
elif page == "Forecasting & Planning":
    st.subheader("Forecasting & Planning")
    st.caption("Revenue forecast using historical trend plus recent growth trajectory.")

    monthly = view.copy()
    monthly["Month"] = pd.to_datetime(monthly["Date"]).dt.to_period("M").dt.to_timestamp()
    monthly = monthly.groupby("Month", as_index=False)["Revenue USD"].sum().sort_values("Month")

    if len(monthly) < 3:
        st.warning("At least three monthly periods are required for forecasting.")
    else:
        x = np.arange(len(monthly)).reshape(-1, 1)
        y = monthly["Revenue USD"].values
        model = LinearRegression().fit(x, y)

        horizon = st.slider("Forecast horizon (months)", 3, 12, 6)
        future_x = np.arange(len(monthly), len(monthly) + horizon).reshape(-1, 1)
        trend_forecast = model.predict(future_x)

        recent_growth = monthly["Revenue USD"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        avg_growth = recent_growth.tail(6).mean() if len(recent_growth) else 0
        base = trend_forecast * (1 + avg_growth * 0.35)
        optimistic = base * 1.05
        downside = base * 0.95

        future_dates = pd.date_range(
            monthly["Month"].max() + pd.offsets.MonthBegin(1),
            periods=horizon,
            freq="MS",
        )

        forecast = pd.DataFrame({
            "Month": future_dates,
            "Base": base,
            "Optimistic": optimistic,
            "Downside": downside,
        })

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly["Revenue USD"], mode="lines+markers", name="Historical"))
        fig.add_trace(go.Scatter(x=forecast["Month"], y=forecast["Base"], mode="lines+markers", name="Base"))
        fig.add_trace(go.Scatter(x=forecast["Month"], y=forecast["Optimistic"], mode="lines", name="Optimistic"))
        fig.add_trace(go.Scatter(x=forecast["Month"], y=forecast["Downside"], mode="lines", name="Downside"))
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

        a, b, c = st.columns(3)
        a.metric("Historical Avg Growth", pct(avg_growth * 100))
        b.metric("Next Period Base", money_usd(base[0]))
        c.metric("Forecast Horizon", f"{horizon} months")

        display = forecast.copy()
        for c in ["Base", "Optimistic", "Downside"]:
            display[c] = display[c].map(money_usd)
        st.dataframe(display, use_container_width=True, hide_index=True)


# ============================================================
# WHAT-IF
# ============================================================
elif page == "What-if Scenarios":
    st.subheader("What-if Scenario Planner")

    cost_change = st.slider("Change in actual cost", -30, 30, 0, 1)
    revenue_change = st.slider("Change in revenue", -20, 30, 0, 1)

    scenario_revenue = revenue * (1 + revenue_change / 100)
    scenario_cost = actual * (1 + cost_change / 100)
    scenario_profit = scenario_revenue - scenario_cost
    scenario_margin = scenario_profit / scenario_revenue * 100 if scenario_revenue else 0

    a, b, c = st.columns(3)
    a.metric("Scenario Revenue", money_usd(scenario_revenue), f"{revenue_change:+.0f}%")
    b.metric("Scenario Cost", money_usd(scenario_cost), f"{cost_change:+.0f}%")
    c.metric("Scenario Margin", pct(scenario_margin), f"{scenario_margin - margin:+.1f} pts")

    st.info("Use this planner to test cost-control and growth assumptions before management review.")


# ============================================================
# RISK
# ============================================================
elif page == "Risk & Anomaly Detection":
    st.subheader("Risk & Anomaly Detection")

    risk = view.copy()
    numeric = risk[["Budget USD", "Actual USD", "Revenue USD", "Variance USD"]].fillna(0)

    if len(risk) >= 20:
        model = IsolationForest(contamination=0.03, random_state=42)
        prediction = model.fit_predict(numeric)
        risk["AI Anomaly"] = np.where(prediction == -1, "AI Flag", "Normal")
    else:
        risk["AI Anomaly"] = "Insufficient sample"

    risk["Risk Score"] = (
        risk["Variance %"].abs().clip(0, 100) * 0.6
        + risk["Anomaly Flag"] * 40
    ).clip(0, 100)

    a, b, c = st.columns(3)
    a.metric("Source Anomalies", f"{int(risk['Anomaly Flag'].sum()):,}")
    b.metric("AI Flags", f"{int((risk['AI Anomaly'] == 'AI Flag').sum()):,}")
    c.metric("High Risk Transactions", f"{int((risk['Risk Score'] >= 70).sum()):,}")

    top = risk.sort_values("Risk Score", ascending=False).head(25).copy()
    for c in ["Budget USD", "Actual USD", "Revenue USD", "Variance USD"]:
        top[c] = top[c].map(money_usd)
    top["Risk Score"] = risk.sort_values("Risk Score", ascending=False).head(25)["Risk Score"].map(lambda x: f"{x:.0f}")
    st.dataframe(top, use_container_width=True, hide_index=True)


# ============================================================
# COST INTELLIGENCE
# ============================================================
elif page == "Cost Intelligence":
    st.subheader("Cost Intelligence")

    cost = view.groupby("Cost Category", as_index=False).agg(
        Budget=("Budget USD", "sum"),
        Actual=("Actual USD", "sum"),
    )
    cost["Variance"] = cost["Actual"] - cost["Budget"]
    cost["Variance %"] = np.where(cost["Budget"] != 0, cost["Variance"] / cost["Budget"] * 100, 0)

    left, right = st.columns(2)

    with left:
        fig = px.bar(cost.sort_values("Variance"), x="Variance", y="Cost Category", orientation="h", title="Top Cost Variances")
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.pie(cost, names="Cost Category", values="Actual", title="Actual Cost Mix", hole=.45)
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    display = cost.copy()
    for c in ["Budget", "Actual", "Variance"]:
        display[c] = display[c].map(money_usd)
    display["Variance %"] = cost["Variance %"].map(pct)
    st.dataframe(display, use_container_width=True, hide_index=True)


# ============================================================
# MANAGEMENT ACTIONS
# ============================================================
elif page == "Management Actions":
    st.subheader("Management Actions")
    st.caption("AI CFO recommended actions generated from current financial signals.")

    cost = view.groupby("Cost Category", as_index=False).agg(
        Budget=("Budget USD", "sum"),
        Actual=("Actual USD", "sum"),
    )
    cost["Variance"] = cost["Actual"] - cost["Budget"]
    cost = cost.sort_values("Variance", ascending=False)

    actions = []
    for _, r in cost.head(10).iterrows():
        if r["Variance"] > 0:
            actions.append({
                "Priority": "High" if r["Variance"] / max(r["Budget"], 1) > .10 else "Medium",
                "Area": r["Cost Category"],
                "Issue": f"Over budget by {money_usd(r['Variance'])}",
                "Action": "Investigate root cause and create corrective action.",
                "Owner": "Finance / Cost Owner",
                "Status": "Open",
            })

    if actions:
        st.dataframe(pd.DataFrame(actions), use_container_width=True, hide_index=True)
    else:
        st.success("No unfavorable cost actions were generated for the current filters.")


# ============================================================
# REPORTS
# ============================================================
elif page == "Reports Library":
    st.subheader("Reports Library")

    reports = pd.DataFrame({
        "Report": [
            "Executive KPI Pack",
            "Budget vs Actual",
            "Regional Performance",
            "Cost Intelligence",
            "Risk & Anomaly Register",
            "Forecast & Scenario Pack",
            "Management Action Register",
        ],
        "Frequency": ["Monthly", "Monthly", "Monthly", "Monthly", "Weekly", "Monthly", "Weekly"],
        "Status": ["Ready", "Ready", "Ready", "Ready", "Ready", "Ready", "Ready"],
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
# DATA QUALITY
# ============================================================
elif page == "Data Quality":
    st.subheader("Data Quality")

    quality = []
    for col in ["Date", "Region", "Country", "Legal Entity", "Department", "Cost Category", "Budget USD", "Actual USD", "Revenue USD"]:
        if col in view.columns:
            nulls = int(view[col].isna().sum())
            quality.append({
                "Field": col,
                "Rows": len(view),
                "Missing": nulls,
                "Completeness": f"{(1 - nulls / max(len(view), 1)) * 100:.1f}%",
            })

    st.dataframe(pd.DataFrame(quality), use_container_width=True, hide_index=True)


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
        st.info("Upload a market CSV in the sidebar to calculate Alpha, Beta, R, R², Sharpe, Sortino, Treynor, Information Ratio, volatility, VaR and maximum drawdown.")
        st.caption("Expected columns: Date, Asset Price, Benchmark Price. Market analytics are intentionally kept separate from ERP accounting data.")
    else:
        market["Asset Return"] = market["Asset Price"].pct_change()
        has_benchmark = "Benchmark Price" in market.columns
        if has_benchmark:
            market["Benchmark Return"] = market["Benchmark Price"].pct_change()
        ret = market.dropna(subset=["Asset Return"]).copy()

        annual_factor = 252
        asset_mean = ret["Asset Return"].mean()
        asset_vol = ret["Asset Return"].std() * np.sqrt(annual_factor)
        rf = st.number_input("Annual risk-free rate (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.25) / 100
        daily_rf = (1 + rf) ** (1 / annual_factor) - 1

        beta = np.nan
        alpha = np.nan
        corr = np.nan
        r2 = np.nan
        tracking_error = np.nan
        information_ratio = np.nan
        if has_benchmark:
            pair = ret.dropna(subset=["Benchmark Return"])
            if len(pair) > 1 and pair["Benchmark Return"].var() > 0:
                beta = pair["Asset Return"].cov(pair["Benchmark Return"]) / pair["Benchmark Return"].var()
                corr = pair["Asset Return"].corr(pair["Benchmark Return"])
                r2 = corr ** 2 if np.isfinite(corr) else np.nan
                alpha_daily = pair["Asset Return"].mean() - rf / annual_factor - beta * (pair["Benchmark Return"].mean() - rf / annual_factor)
                alpha = (1 + alpha_daily) ** annual_factor - 1
                active = pair["Asset Return"] - pair["Benchmark Return"]
                tracking_error = active.std() * np.sqrt(annual_factor)
                information_ratio = active.mean() / active.std() * np.sqrt(annual_factor) if active.std() > 0 else np.nan

        downside = ret.loc[ret["Asset Return"] < daily_rf, "Asset Return"] - daily_rf
        downside_dev = downside.std() * np.sqrt(annual_factor) if len(downside) > 1 else np.nan
        sharpe = ((asset_mean - daily_rf) / ret["Asset Return"].std() * np.sqrt(annual_factor)) if ret["Asset Return"].std() > 0 else np.nan
        sortino = ((asset_mean - daily_rf) / (downside.std()) * np.sqrt(annual_factor)) if len(downside) > 1 and downside.std() > 0 else np.nan
        treynor = ((asset_mean * annual_factor - rf) / beta) if np.isfinite(beta) and beta != 0 else np.nan
        var_95 = ret["Asset Return"].quantile(0.05)
        cvar_95 = ret.loc[ret["Asset Return"] <= var_95, "Asset Return"].mean()
        wealth = (1 + ret["Asset Return"].fillna(0)).cumprod()
        drawdown = wealth / wealth.cummax() - 1
        max_drawdown = drawdown.min()

        mk1, mk2, mk3, mk4 = st.columns(4)
        mk1.metric("Alpha", f"{alpha*100:.2f}%" if np.isfinite(alpha) else "N/A")
        mk2.metric("Beta", f"{beta:.2f}" if np.isfinite(beta) else "N/A")
        mk3.metric("R²", f"{r2:.2f}" if np.isfinite(r2) else "N/A")
        mk4.metric("Sharpe", f"{sharpe:.2f}" if np.isfinite(sharpe) else "N/A")

        market_metrics = pd.DataFrame([
            ["Alpha", f"{alpha*100:.2f}%" if np.isfinite(alpha) else "N/A", "Risk-adjusted excess return vs benchmark"],
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

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=market["Date"], y=market["Asset Price"], mode="lines", name="Asset"))
        if has_benchmark:
            bench_norm = market["Benchmark Price"] / market["Benchmark Price"].dropna().iloc[0] * market["Asset Price"].dropna().iloc[0]
            fig.add_trace(go.Scatter(x=market["Date"], y=bench_norm, mode="lines", name="Benchmark (normalized)"))
        chart_layout(fig, height=340)
        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # TECHNICAL ANALYTICS
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
    st.caption("Prototype audit trail for finance actions.")

    logs = pd.DataFrame({
        "Timestamp": [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")],
        "User": ["Demo CFO"],
        "Action": ["Dashboard viewed"],
        "Module": ["Executive Dashboard"],
        "Status": ["Success"],
    })
    st.dataframe(logs, use_container_width=True, hide_index=True)


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "FinSight AI is an AI CFO / agentic FP&A prototype. "
    "USD is the consolidation base; local entity currencies are retained for drill-down. "
    "Cash-flow outputs are modeled when live treasury data is unavailable."
)
