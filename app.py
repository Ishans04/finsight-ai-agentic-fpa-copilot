
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
# AI CFO ACTION WORKFLOW STATE
# ============================================================
if "cfo_actions" not in st.session_state:
    st.session_state.cfo_actions = []
if "cfo_audit" not in st.session_state:
    st.session_state.cfo_audit = []


def log_cfo_event(action, module="AI CFO", status="Success"):
    st.session_state.cfo_audit.insert(0, {
        "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "User": "Demo CFO",
        "Action": action,
        "Module": module,
        "Status": status,
    })
    st.session_state.cfo_audit = st.session_state.cfo_audit[:100]


def create_cfo_action(priority, area, issue, impact, recommendation, owner):
    existing = [a for a in st.session_state.cfo_actions if a.get("Issue") == issue and a.get("Status") != "Resolved"]
    if existing:
        return False
    action_id = f"ACT-{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}-{len(st.session_state.cfo_actions)+1:03d}"
    st.session_state.cfo_actions.insert(0, {
        "Action ID": action_id,
        "Created": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "Priority": priority,
        "Area": area,
        "Issue": issue,
        "Financial Impact": impact,
        "Recommendation": recommendation,
        "Owner": owner,
        "Due Date": (pd.Timestamp.now() + pd.Timedelta(days=14)).strftime("%Y-%m-%d"),
        "Status": "Open",
    })
    log_cfo_event(f"Created management action {action_id}: {area}")
    return True


def action_status_counts():
    if not st.session_state.cfo_actions:
        return 0, 0, 0
    statuses = pd.Series([a.get("Status", "Open") for a in st.session_state.cfo_actions])
    return int((statuses == "Open").sum()), int((statuses == "In Progress").sum()), int((statuses == "Resolved").sum())


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
# EXECUTIVE DASHBOARD — CFO-GRADE V7
# ============================================================
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
# AI CFO
# ============================================================
elif page == "AI CFO":
    st.subheader("🤖 AI CFO")
    st.caption("Decision-support engine that converts financial signals into explainable management actions.")

    findings = []
    cost_overrun = float(max(variance, 0))
    if cost_overrun > 0:
        findings.append({
            "Priority": "High", "Area": "Cost Control",
            "Issue": "Cost over budget",
            "Finding": f"Actual cost exceeds budget by {money_usd(cost_overrun)}.",
            "Impact": money_usd(cost_overrun),
            "Risk": "Margin and cash pressure if the overrun persists.",
            "Recommendation": "Review the largest unfavorable cost categories, cost centers and transaction-level drivers.",
            "Owner": "Finance Controller",
        })

    if margin < 60:
        findings.append({
            "Priority": "High", "Area": "Profitability",
            "Issue": "Margin pressure",
            "Finding": f"Current consolidated margin is {pct(margin)}.",
            "Impact": "Profitability risk",
            "Risk": "Lower earnings conversion and reduced headroom for investment.",
            "Recommendation": "Review pricing, delivery economics and high-growth cost categories.",
            "Owner": "CFO / FP&A",
        })
    elif margin < 70:
        findings.append({
            "Priority": "Medium", "Area": "Profitability",
            "Issue": "Margin watch",
            "Finding": f"Current consolidated margin is {pct(margin)}.",
            "Impact": "Margin monitoring",
            "Risk": "Further cost inflation could compress profitability.",
            "Recommendation": "Monitor margin trend and investigate unfavorable cost movement.",
            "Owner": "FP&A",
        })

    anomalies = int(view["Anomaly Flag"].sum())
    if anomalies:
        findings.append({
            "Priority": "High", "Area": "Risk & Controls",
            "Issue": "Anomaly exposure",
            "Finding": f"{anomalies:,} transaction(s) are flagged for investigation.",
            "Impact": f"{anomalies:,} transactions",
            "Risk": "Potential control, posting or unusual-spend risk.",
            "Recommendation": "Prioritize the highest-value anomalies and document investigation outcomes.",
            "Owner": "Controller",
        })

    if not findings:
        findings.append({
            "Priority": "Low", "Area": "Financial Health",
            "Issue": "No material exception detected",
            "Finding": "No high-priority management exception was generated from the current view.",
            "Impact": "Monitoring",
            "Risk": "Low",
            "Recommendation": "Continue routine monitoring and periodic forecast review.",
            "Owner": "FP&A",
        })

    open_count, progress_count, resolved_count = action_status_counts()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("AI Findings", len(findings))
    k2.metric("Open Actions", open_count)
    k3.metric("In Progress", progress_count)
    k4.metric("Resolved", resolved_count)

    st.markdown("### AI CFO Findings")
    for idx, f in enumerate(findings):
        cls = "danger" if f["Priority"] == "High" else "warning" if f["Priority"] == "Medium" else "insight"
        html_block(
            f'<div class="{cls}"><b>{f["Priority"]} · {f["Area"]} · {f["Issue"]}</b><br>'
            f'{f["Finding"]}<br><span class="small">Financial impact: {f["Impact"]} · Owner: {f["Owner"]}</span></div>'
        )
        with st.expander(f"🔎 Investigate: {f['Issue']}"):
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Finding**  ")
                st.write(f["Finding"])
                st.write(f"**Financial impact**  ")
                st.write(f["Impact"])
                st.write(f"**Risk**  ")
                st.write(f["Risk"])
            with c2:
                st.write(f"**Recommended action**  ")
                st.write(f["Recommendation"])
                st.write(f"**Owner**  ")
                st.write(f["Owner"])
                if st.button("Create Management Action", key=f"create_ai_action_{idx}"):
                    created = create_cfo_action(
                        f["Priority"], f["Area"], f["Issue"], f["Impact"], f["Recommendation"], f["Owner"]
                    )
                    if created:
                        st.success("Management action created and added to the Action Center.")
                    else:
                        st.info("An open action for this issue already exists.")

    st.subheader("CFO Executive Brief")
    st.info(
        f"Revenue is {money_usd(revenue)}, actual cost is {money_usd(actual)}, "
        f"budget is {money_usd(budget)}, and profit margin is {pct(margin)}. "
        f"The current cost variance is {money_usd(variance)}. "
        f"The AI CFO has generated {len(findings)} management signal(s)."
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
                temp = q_view.groupby("Department", as_index=False).agg(Budget=("Budget USD", "sum"), Actual=("Actual USD", "sum"))
                temp["Variance"] = temp["Actual"] - temp["Budget"]
                top = temp.sort_values("Variance", ascending=False).iloc[0]
                scope = ", ".join(applied_filters) if applied_filters else "current filtered view"
                st.success(f"Within {scope}, {top['Department']} is the largest unfavorable departmental variance at {money_usd(top['Variance'])}.")
            elif "region" in q or "business unit" in q:
                temp = q_view.groupby("Region", as_index=False).agg(Budget=("Budget USD", "sum"), Actual=("Actual USD", "sum"))
                temp["Variance"] = temp["Actual"] - temp["Budget"]
                top = temp.sort_values("Variance", ascending=False).iloc[0]
                scope = ", ".join(applied_filters) if applied_filters else "current filtered view"
                st.success(f"Within {scope}, {top['Region']} is the largest unfavorable regional variance at {money_usd(top['Variance'])}.")
            elif "cost" in q or "category" in q or "driver" in q:
                temp = q_view.groupby("Cost Category", as_index=False).agg(Budget=("Budget USD", "sum"), Actual=("Actual USD", "sum"))
                temp["Variance"] = temp["Actual"] - temp["Budget"]
                top = temp.sort_values("Variance", ascending=False).iloc[0]
                scope = ", ".join(applied_filters) if applied_filters else "current filtered view"
                st.success(f"Within {scope}, {top['Cost Category']} is the largest unfavorable cost driver at {money_usd(top['Variance'])}.")
            else:
                st.info("Try asking about a region, department, cost category, budget variance, or driver.")


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
    k3.metric("Net Variance", money_usd(total_variance), delta=pct(total_variance / total_budget * 100) if total_budget else "0.0%")
    k4.metric("Unfavorable Exposure", money_usd(total_unfav))

    st.markdown("### Cost Driver Analysis")
    left, right = st.columns(2)
    with left:
        driver = cost.sort_values("Variance", ascending=True).copy()
        fig = px.bar(driver, x="Variance", y="Cost Category", orientation="h", title="Budget vs Actual Variance")
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
    a4.metric("Total Actions", len(st.session_state.cfo_actions))

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
                    st.session_state.cfo_actions[0]["Due Date"] = str(due_date)
                    st.success("Action created successfully.")
                else:
                    st.info("An open action with the same issue already exists.")

    st.markdown("### Action Register")
    if not st.session_state.cfo_actions:
        st.info("No management actions yet. Create one from an AI CFO finding or use the form above.")
    else:
        for i, action in enumerate(st.session_state.cfo_actions):
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
                        action["Status"] = new_status
                        log_cfo_event(f"Updated {action['Action ID']} to {new_status}", module="AI CFO Action Center")
                        st.success(f"Action updated to {new_status}.")
                        st.rerun()

        st.markdown("### Action Register Export")
        export_df = pd.DataFrame(st.session_state.cfo_actions)
        st.download_button(
            "Download Action Register CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="finsight_ai_cfo_action_register.csv",
            mime="text/csv",
        )

    st.markdown("### Workflow")
    st.info("Detect → Explain → Recommend → Assign → Track → Resolve. Actions are stored for the current Streamlit session in this prototype.")


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
    st.caption("Prototype audit trail for AI CFO and management-action events.")
    if not st.session_state.cfo_audit:
        st.info("No action events recorded yet. Create or update an AI CFO action to populate the audit trail.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.cfo_audit), use_container_width=True, hide_index=True)
        st.download_button(
            "Download Audit Log CSV",
            data=pd.DataFrame(st.session_state.cfo_audit).to_csv(index=False).encode("utf-8"),
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
