
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
        pathlib.Path("synthetic_erp_financials_global_v2.csv"),
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
        "Settings",
        "Audit Logs",
    ]

    page = st.radio("Command Center", modules, index=0)

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

    cash_balance = (
        float(view["Closing Cash"].iloc[0])
        if "Closing Cash" in view.columns
        else float((view["Revenue USD"] - view["Actual USD"]).sum())
    )
    gross_profit = float(view["Gross Profit"].iloc[0]) if "Gross Profit" in view.columns else 0.0
    ebitda = float(view["EBITDA"].iloc[0]) if "EBITDA" in view.columns else 0.0
    pat = float(view["PAT"].iloc[0]) if "PAT" in view.columns else 0.0

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
