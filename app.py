import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from textwrap import dedent


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FinSight AI | Finance Command Center",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# ENTERPRISE DARK THEME
# ============================================================

st.markdown(
    """
<style>
.stApp {
    background:
        radial-gradient(circle at 85% 5%, rgba(37,99,235,.10), transparent 25%),
        radial-gradient(circle at 15% 90%, rgba(124,58,237,.08), transparent 30%),
        #050b14;
    color: #e8eef8;
}

.block-container {
    max-width: 1500px;
    padding-top: 1rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#07101d 0%,#081321 55%,#050b14 100%);
    border-right: 1px solid #17263b;
}

section[data-testid="stSidebar"] hr {
    border-color: #1a2940;
}

h1,h2,h3,h4 {
    color: #f4f7fb !important;
    font-weight: 750 !important;
}

p {
    color: #9aa9bd;
}

/* BRAND */

.brand-box {
    padding: .4rem 0 1rem 0;
}

.brand-title {
    font-size: 1.45rem;
    font-weight: 800;
    color: #f8fafc;
}

.brand-title span {
    color: #60a5fa;
}

.brand-subtitle {
    font-size: .72rem;
    color: #8191a8;
}

/* HERO */

.hero {
    padding: 1.45rem 1.8rem;
    border-radius: 16px;
    background:
        radial-gradient(circle at 85% 15%,rgba(6,182,212,.23),transparent 28%),
        linear-gradient(135deg,#0b1730 0%,#10264a 50%,#075d68 100%);
    border: 1px solid #244767;
    margin-bottom: 1.2rem;
    box-shadow: 0 15px 45px rgba(0,0,0,.3);
}

.hero-title {
    color: white;
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: .35rem;
}

.hero-subtitle {
    color: #c9d6e6;
    font-size: .95rem;
}

/* SEARCH */

.search-box {
    background: #081321;
    border: 1px solid #1c3048;
    border-radius: 9px;
    padding: .7rem 1rem;
    color: #75869d;
    margin-bottom: 1.1rem;
}

/* KPI */

.kpi-card {
    background: linear-gradient(145deg,#0c1727,#091321);
    border: 1px solid #1b3049;
    border-radius: 13px;
    padding: 1rem;
    min-height: 112px;
    box-shadow: 0 7px 22px rgba(0,0,0,.22);
}

.kpi-label {
    color: #91a2b8;
    font-size: .76rem;
    margin-bottom: .4rem;
}

.kpi-value {
    color: #f8fafc;
    font-size: 1.42rem;
    font-weight: 800;
    line-height: 1.2;
    white-space: nowrap;
}

.kpi-value.blue {
    color: #60a5fa;
}

.kpi-value.green {
    color: #4ade80;
}

.kpi-value.purple {
    color: #a78bfa;
}

.kpi-value.cyan {
    color: #22d3ee;
}

.kpi-value.orange {
    color: #fb923c;
}

.kpi-value.yellow {
    color: #facc15;
}

.kpi-delta {
    margin-top: .4rem;
    font-size: .68rem;
    color: #6ee7b7;
}

.kpi-delta.warn {
    color: #fbbf24;
}

.kpi-delta.bad {
    color: #fb7185;
}

/* PANELS */

.panel {
    background: linear-gradient(145deg,#0b1625,#08121f);
    border: 1px solid #1a2c43;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
}

.panel-title {
    color: #edf3fa;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: .15rem;
}

/* AI */

.ai-box {
    background: linear-gradient(145deg,#101638,#0b1227);
    border: 1px solid #303b76;
    border-left: 4px solid #6366f1;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-bottom: .8rem;
}

.ai-title {
    color: #c4b5fd;
    font-weight: 750;
    font-size: .95rem;
}

.ai-label {
    color: #8ea0b9;
    font-size: .68rem;
    text-transform: uppercase;
    letter-spacing: .6px;
    margin-top: .5rem;
}

.ai-text {
    color: #dce5f1;
    font-size: .8rem;
    line-height: 1.5;
}

/* ALERTS */

.alert-box {
    padding: .85rem 1rem;
    border-radius: 9px;
    margin-bottom: .6rem;
    border: 1px solid #26364d;
    background: #0b1523;
}

.alert-critical {
    border-left: 4px solid #ef4444;
}

.alert-warning {
    border-left: 4px solid #f59e0b;
}

.alert-good {
    border-left: 4px solid #22c55e;
}

.alert-title {
    color: #e7edf6;
    font-weight: 700;
    font-size: .8rem;
}

.alert-text {
    color: #8292a8;
    font-size: .72rem;
}

/* ACTION */

.action-card {
    background: #0a1523;
    border: 1px solid #1d3047;
    border-radius: 9px;
    padding: .8rem .9rem;
    margin-bottom: .55rem;
}

.priority-p1 {
    color: #fbbf24;
    font-weight: 800;
}

.priority-p2 {
    color: #4ade80;
    font-weight: 800;
}

/* INPUTS */

div[data-baseweb="select"] > div {
    background-color: #081321 !important;
    border-color: #1c3048 !important;
}

.stTextInput input {
    background: #081321 !important;
    color: #e2e8f0 !important;
    border: 1px solid #1c3048 !important;
}

/* BUTTON */

.stButton > button {
    background: linear-gradient(135deg,#2563eb,#4f46e5) !important;
    color: white !important;
    border: 1px solid #4f7cff !important;
    border-radius: 8px !important;
    font-weight: 650 !important;
}

/* FOOTER */

.finsight-footer {
    margin-top: 2rem;
    padding: 1rem;
    border-top: 1px solid #1a2940;
    text-align: center;
    color: #5f7087;
    font-size: .72rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# FORMATTING
# ============================================================

def money(value):
    try:
        value = float(value)

        if abs(value) >= 10_000_000:
            return f"₹{value / 10_000_000:,.2f} Cr"

        if abs(value) >= 100_000:
            return f"₹{value / 100_000:,.2f} L"

        return f"₹{value:,.0f}"

    except Exception:
        return "₹0"


def money_full(value):
    try:
        return f"₹{float(value):,.0f}"
    except Exception:
        return "₹0"


def pct(value):
    try:
        return f"{float(value):.2f}%"
    except Exception:
        return "0.00%"


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_name(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def normalize_columns(df):

    df = df.copy()

    aliases = {

        "Date": [
            "date",
            "transaction_date",
            "posting_date",
            "period",
            "month",
        ],

        "Business Unit": [
            "business_unit",
            "businessunit",
            "bu",
            "entity",
            "region",
        ],

        "Department": [
            "department",
            "dept",
            "department_name",
            "function",
        ],

        "Account / Cost Category": [
            "account",
            "account_name",
            "cost_category",
            "costcategory",
            "account_cost_category",
            "gl_account",
            "category",
        ],

        "Budget": [
            "budget",
            "budget_amount",
            "planned",
            "plan",
        ],

        "Actual": [
            "actual",
            "actual_amount",
            "actuals",
            "expense",
            "cost",
        ],

        "Revenue": [
            "revenue",
            "sales",
            "revenue_amount",
            "income",
        ],

        "Variance": [
            "variance",
            "budget_variance",
            "variance_amount",
        ],

        "Variance %": [
            "variance_pct",
            "variance_percent",
            "variance_percentage",
        ],

        "Profit": [
            "profit",
            "gross_profit",
            "operating_profit",
        ],

        "Transaction ID": [
            "transaction_id",
            "transactionid",
            "txn_id",
            "id",
        ],

        "Anomaly Flag": [
            "anomaly_flag",
            "anomalyflag",
        ],

        "Anomaly": [
            "anomaly",
            "anomaly_status",
        ],
    }

    normalized = {
        normalize_name(col): col
        for col in df.columns
    }

    rename_map = {}

    for target, candidates in aliases.items():

        if target in df.columns:
            continue

        for candidate in candidates:

            if candidate in normalized:

                rename_map[
                    normalized[candidate]
                ] = target

                break

    df = df.rename(
        columns=rename_map
    )

    return df


# ============================================================
# DEPARTMENT INTELLIGENCE
# ============================================================

def derive_department(df):

    df = df.copy()

    # First use an existing Department if it contains
    # meaningful information.

    if "Department" in df.columns:

        existing = (
            df["Department"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        valid = (
            existing != ""
        ) & (
            existing.str.lower()
            .isin(
                [
                    "unmapped",
                    "unknown",
                    "na",
                    "nan",
                    "none",
                ]
            ) == False
        )

        # Keep real departments where they exist.
        df.loc[valid, "Department"] = existing[valid]

    else:

        df["Department"] = ""

    # Cost-category based intelligent mapping.

    mapping = {
        "cloud": "Engineering",
        "cloud infrastructure": "Engineering",
        "software": "Engineering",
        "software licenses": "Engineering",
        "facilities": "Operations",
        "office": "Operations",
        "office & facilities": "Operations",
        "marketing": "Marketing",
        "payroll": "HR",
        "professional services": "Finance",
        "travel": "Sales",
        "training": "HR",
    }

    category = (
        df["Account / Cost Category"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    derived = []

    for value in category:

        department = "Finance"

        for key, mapped in mapping.items():

            if key in value:
                department = mapped
                break

        derived.append(department)

    derived = pd.Series(
        derived,
        index=df.index
    )

    current = (
        df["Department"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    bad_department = current.isin(
        [
            "",
            "unmapped",
            "unknown",
            "nan",
            "none",
            "na",
        ]
    )

    df.loc[
        bad_department,
        "Department"
    ] = derived[bad_department]

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    df = normalize_columns(df)

    if "Date" in df.columns:

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

    # Required fields

    for col in [
        "Business Unit",
        "Account / Cost Category",
    ]:

        if col not in df.columns:
            df[col] = "Unspecified"

        df[col] = (
            df[col]
            .fillna("Unspecified")
            .astype(str)
        )

    # Numeric fields

    for col in [
        "Budget",
        "Actual",
        "Revenue",
        "Variance",
        "Variance %",
        "Profit",
    ]:

        if col not in df.columns:
            df[col] = 0.0

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).fillna(0)

    # Calculations

    df["Variance"] = (
        df["Actual"] -
        df["Budget"]
    )

    df["Variance %"] = np.where(
        df["Budget"] != 0,
        df["Variance"] /
        df["Budget"] *
        100,
        0,
    )

    df["Profit"] = (
        df["Revenue"] -
        df["Actual"]
    )

    # Transaction ID

    if "Transaction ID" not in df.columns:

        df["Transaction ID"] = [
            f"TXN-{i + 1:05d}"
            for i in range(len(df))
        ]

    # Anomaly

    if "Anomaly Flag" not in df.columns:
        df["Anomaly Flag"] = 0

    if "Anomaly" not in df.columns:
        df["Anomaly"] = "Normal"

    # IMPORTANT:
    # derive department instead of Unmapped

    df = derive_department(df)

    return df


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_default_data():

    paths = [
        Path("synthetic_erp_financials.csv"),
        Path("data/synthetic_erp_financials.csv"),
    ]

    for path in paths:

        if path.exists():

            return prepare_data(
                pd.read_csv(path)
            )

    return None


default_df = load_default_data()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
<div class="brand-box">
    <div class="brand-title">
        📊 Fin<span>Sight</span> AI
    </div>
    <div class="brand-subtitle">
        AI CFO & Finance Command Center
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Data Source")

    uploaded_file = st.file_uploader(
        "Upload ERP CSV",
        type=["csv"],
    )

    if uploaded_file is not None:

        try:

            df = prepare_data(
                pd.read_csv(
                    uploaded_file
                )
            )

            source_label = (
                "Uploaded ERP Data"
            )

        except Exception as e:

            st.error(
                f"Upload error: {e}"
            )

            df = (
                default_df.copy()
                if default_df is not None
                else None
            )

            source_label = (
                "FinSight Demo ERP Data"
            )

    else:

        df = (
            default_df.copy()
            if default_df is not None
            else None
        )

        source_label = (
            "FinSight Demo ERP Data"
        )

    st.caption(
        f"Source: {source_label}"
    )

    st.divider()

    st.markdown("### Navigation")

    page = st.radio(
        "Navigation",
        [
            "🏠 Executive Dashboard",
            "🤖 AI CFO",
            "📊 Financial Performance",
            "💵 Cash Flow Center",
            "🎯 Budget vs Actuals",
            "📈 Forecasting & Planning",
            "🔮 What-if Scenarios",
            "⚠️ Risk & Anomaly Detection",
            "💡 Cost Intelligence",
            "🎯 Management Actions",
            "📑 Reports Library",
            "📤 Upload Data",
            "🔗 ERP Connections",
            "🧩 Data Mapping",
            "✅ Data Quality",
            "🔔 Alerts & Notifications",
            "⚙️ Settings",
            "🧾 Audit Logs",
            "🔎 Data Explorer",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### Global Filters")

    if df is not None:

        bu_options = (
            ["All"]
            +
            sorted(
                df["Business Unit"]
                .dropna()
                .unique()
                .tolist()
            )
        )

        selected_bu = st.selectbox(
            "Business Unit",
            bu_options,
        )

        dept_options = (
            ["All"]
            +
            sorted(
                df["Department"]
                .dropna()
                .unique()
                .tolist()
            )
        )

        selected_department = st.selectbox(
            "Department",
            dept_options,
        )

    else:

        selected_bu = "All"
        selected_department = "All"


# ============================================================
# VALIDATION
# ============================================================

if df is None:

    st.error(
        "No financial dataset found."
    )

    st.stop()


required_columns = [
    "Date",
    "Business Unit",
    "Department",
    "Account / Cost Category",
    "Budget",
    "Actual",
    "Revenue",
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:

    st.error(
        "Missing required columns: "
        + ", ".join(missing)
    )

    st.stop()


# ============================================================
# FILTER
# ============================================================

filtered_df = df.copy()

if selected_bu != "All":

    filtered_df = filtered_df[
        filtered_df["Business Unit"]
        == selected_bu
    ]

if selected_department != "All":

    filtered_df = filtered_df[
        filtered_df["Department"]
        == selected_department
    ]


# ============================================================
# GLOBAL METRICS
# ============================================================

total_revenue = filtered_df[
    "Revenue"
].sum()

total_actual = filtered_df[
    "Actual"
].sum()

total_budget = filtered_df[
    "Budget"
].sum()

total_variance = filtered_df[
    "Variance"
].sum()

total_profit = filtered_df[
    "Profit"
].sum()

profit_margin = (
    total_profit /
    total_revenue *
    100
    if total_revenue != 0
    else 0
)

variance_pct = (
    total_variance /
    total_budget *
    100
    if total_budget != 0
    else 0
)


# ============================================================
# COMPONENTS
# ============================================================

def render_hero(title, subtitle):

    html = dedent(
        f"""
        <div class="hero">
            <div class="hero-title">
                📊 {title}
            </div>
            <div class="hero-subtitle">
                {subtitle}
            </div>
        </div>
        """
    )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


def render_kpi(
    label,
    value,
    delta,
    color="blue",
    delta_type="good",
):

    html = dedent(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">
                {label}
            </div>

            <div class="kpi-value {color}">
                {value}
            </div>

            <div class="kpi-delta {delta_type}">
                {delta}
            </div>
        </div>
        """
    )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


def render_ai_box(
    title,
    finding,
    evidence,
    impact,
    action,
):

    html = dedent(
        f"""
        <div class="ai-box">

            <div class="ai-title">
                🤖 {title}
            </div>

            <div class="ai-label">
                Management Finding
            </div>

            <div class="ai-text">
                {finding}
            </div>

            <div class="ai-label">
                Evidence
            </div>

            <div class="ai-text">
                {evidence}
            </div>

            <div class="ai-label">
                Financial Impact
            </div>

            <div class="ai-text">
                {impact}
            </div>

            <div class="ai-label">
                Recommended Action
            </div>

            <div class="ai-text">
                {action}
            </div>

        </div>
        """
    )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


def dark_chart(fig):

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#b9c6d8"
        ),
        margin=dict(
            l=10,
            r=10,
            t=35,
            b=10,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    fig.update_xaxes(
        gridcolor="#18283d"
    )

    fig.update_yaxes(
        gridcolor="#18283d"
    )

    return fig


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

if page == "🏠 Executive Dashboard":

    render_hero(
        "FinSight AI",
        "Agentic FP&A & ERP Intelligence Command Center — turning financial data into management decisions.",
    )

    st.markdown(
        """
        <div class="search-box">
            🔍 Search for insights, reports, or ask AI CFO...
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "## Executive Dashboard"
    )

    st.caption(
        "Executive view of revenue, cost, profitability, budget performance and financial risk."
    )

    # --------------------------------------------------------
    # KPI ROW
    # --------------------------------------------------------

    c1, c2, c3, c4, c5, c6 = st.columns(
        6
    )

    with c1:

        render_kpi(
            "Revenue",
            money(total_revenue),
            "▲ Financial performance",
            "blue",
        )

    with c2:

        render_kpi(
            "Gross Profit",
            money(total_profit),
            f"Margin {pct(profit_margin)}",
            "green",
        )

    with c3:

        ebitda = total_profit * 0.72

        render_kpi(
            "EBITDA",
            money(ebitda),
            "Modeled operating indicator",
            "purple",
        )

    with c4:

        cash_balance = max(
            total_revenue * 0.19,
            0
        )

        render_kpi(
            "Cash Balance",
            money(cash_balance),
            "Modeled liquidity",
            "orange",
        )

    with c5:

        render_kpi(
            "Profit Margin",
            pct(profit_margin),
            "Revenue profitability",
            "cyan",
        )

    with c6:

        anomaly_count = int(
            filtered_df[
                "Anomaly Flag"
            ].sum()
        )

        risk_score = min(
            100,
            max(
                0,
                int(
                    abs(variance_pct) * 3
                    +
                    anomaly_count
                )
            )
        )

        risk_level = (
            "High"
            if risk_score >= 70
            else
            "Medium"
            if risk_score >= 35
            else
            "Low"
        )

        risk_color = (
            "orange"
            if risk_level == "High"
            else
            "yellow"
            if risk_level == "Medium"
            else
            "green"
        )

        render_kpi(
            "Risk Score",
            risk_level,
            f"Score {risk_score}/100",
            risk_color,
            "warn"
            if risk_level != "Low"
            else "good",
        )

    st.divider()

    # --------------------------------------------------------
    # REVENUE / BU VARIANCE
    # --------------------------------------------------------

    left, right = st.columns(
        [1.35, 1]
    )

    with left:

        st.markdown(
            "### Revenue vs Budget Trend"
        )

        monthly = (
            filtered_df
            .dropna(
                subset=["Date"]
            )
            .assign(
                Month=lambda x:
                x["Date"]
                .dt
                .to_period("M")
                .astype(str)
            )
            .groupby("Month")
            .agg(
                Revenue=(
                    "Revenue",
                    "sum"
                ),
                Budget=(
                    "Budget",
                    "sum"
                ),
                Actual=(
                    "Actual",
                    "sum"
                ),
            )
            .reset_index()
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=monthly["Month"],
                y=monthly["Revenue"],
                mode="lines+markers",
                name="Revenue",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=monthly["Month"],
                y=monthly["Budget"],
                mode="lines+markers",
                name="Budget",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=monthly["Month"],
                y=monthly["Actual"],
                mode="lines+markers",
                name="Actual Cost",
            )
        )

        fig.update_layout(
            height=370
        )

        st.plotly_chart(
            dark_chart(fig),
            use_container_width=True,
        )

    with right:

        st.markdown(
            "### Business Unit Variance"
        )

        bu = (
            filtered_df
            .groupby(
                "Business Unit"
            )["Variance"]
            .sum()
            .reset_index()
            .sort_values(
                "Variance",
                ascending=False,
            )
        )

        fig = px.bar(
            bu,
            x="Business Unit",
            y="Variance",
            text="Variance",
        )

        fig.update_traces(
            texttemplate="%{text:.3s}",
            textposition="outside",
        )

        fig.update_layout(
            height=370
        )

        st.plotly_chart(
            dark_chart(fig),
            use_container_width=True,
        )

    # --------------------------------------------------------
    # P&L / CASH / AI CFO
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(
        [1.05, 1, .95]
    )

    with col1:

        st.markdown(
            "### P&L Summary"
        )

        pnl = pd.DataFrame(
            {
                "Particulars": [
                    "Revenue",
                    "Cost",
                    "Gross Profit",
                    "EBITDA",
                ],
                "Amount": [
                    money(total_revenue),
                    money(total_actual),
                    money(total_profit),
                    money(ebitda),
                ],
            }
        )

        st.dataframe(
            pnl,
            hide_index=True,
            use_container_width=True,
        )

    with col2:

        st.markdown(
            "### Cash Flow Overview"
        )

        monthly_cash = (
            filtered_df
            .dropna(
                subset=["Date"]
            )
            .assign(
                Month=lambda x:
                x["Date"]
                .dt
                .to_period("M")
                .astype(str)
            )
            .groupby("Month")
            .agg(
                Inflow=(
                    "Revenue",
                    "sum"
                ),
                Outflow=(
                    "Actual",
                    "sum"
                ),
            )
            .reset_index()
        )

        monthly_cash["Net Cash"] = (
            monthly_cash["Inflow"]
            -
            monthly_cash["Outflow"]
        )

        monthly_cash["Closing Cash"] = (
            cash_balance
            +
            monthly_cash[
                "Net Cash"
            ].cumsum()
        )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=monthly_cash["Month"],
                y=monthly_cash["Net Cash"],
                name="Net Cash",
            )
        )

        fig.update_layout(
            height=300
        )

        st.plotly_chart(
            dark_chart(fig),
            use_container_width=True,
        )

        st.caption(
            "Modeled closing cash: "
            +
            money(
                monthly_cash[
                    "Closing Cash"
                ].iloc[-1]
            )
        )

    with col3:

        st.markdown(
            "### AI CFO Assistant"
        )

        cost_summary = (
            filtered_df
            .groupby(
                "Account / Cost Category"
            )["Variance"]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        if not cost_summary.empty:

            driver = (
                cost_summary.index[0]
            )

            driver_value = (
                cost_summary.iloc[0]
            )

        else:

            driver = "No material driver"
            driver_value = 0

        render_ai_box(
            "CFO Assistant",
            f"{driver} is the largest unfavorable cost driver.",
            f"Current variance is {money(driver_value)}.",
            f"Continued overspend could put pressure on operating margin and forecast accuracy.",
            "Review utilization, vendor commitments, discretionary spend and forecast assumptions.",
        )

    # --------------------------------------------------------
    # COST VARIANCES
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(
        [1, 1, .9]
    )

    cost_summary_df = (
        filtered_df
        .groupby(
            "Account / Cost Category"
        )["Variance"]
        .sum()
        .reset_index()
        .sort_values(
            "Variance",
            ascending=False,
        )
    )

    with col1:

        st.markdown(
            "### Top Cost Variances"
        )

        fig = px.bar(
            cost_summary_df.head(8),
            x="Variance",
            y="Account / Cost Category",
            orientation="h",
        )

        fig.update_layout(
            height=340
        )

        st.plotly_chart(
            dark_chart(fig),
            use_container_width=True,
        )

    with col2:

        st.markdown(
            "### Cost Driver Mix"
        )

        positive_costs = (
            cost_summary_df[
                cost_summary_df["Variance"] > 0
            ]
            .head(8)
        )

        if not positive_costs.empty:

            fig = px.pie(
                positive_costs,
                names="Account / Cost Category",
                values="Variance",
                hole=.55,
            )

            fig.update_layout(
                height=340
            )

            st.plotly_chart(
                dark_chart(fig),
                use_container_width=True,
            )

        else:

            st.info(
                "No unfavorable cost drivers."
            )

    with col3:

        st.markdown(
            "### Forecast Summary"
        )

        recent = (
            filtered_df
            .dropna(
                subset=["Date"]
            )
            .assign(
                Month=lambda x:
                x["Date"]
                .dt
                .to_period("M")
            )
            .groupby("Month")[
                "Revenue"
            ]
            .sum()
            .sort_index()
        )

        if len(recent) >= 2:

            growth = (
                recent.iloc[-1]
                /
                recent.iloc[-2]
                -
                1
            )

        else:

            growth = .01

        next_revenue = (
            total_revenue *
            (1 + growth)
        )

        next_profit = (
            next_revenue *
            profit_margin /
            100
        )

        render_kpi(
            "Next Period Revenue",
            money(next_revenue),
            f"Growth assumption {pct(growth * 100)}",
            "blue",
        )

        render_kpi(
            "Forecast Profit",
            money(next_profit),
            f"Margin {pct(profit_margin)}",
            "green",
        )

    # --------------------------------------------------------
    # ACTIONS
    # --------------------------------------------------------

    st.divider()

    col1, col2 = st.columns(
        [1.4, .8]
    )

    with col1:

        st.markdown(
            "### AI CFO Recommended Actions"
        )

        actions = []

        if total_variance > 0:

            actions.append(
                [
                    "P1",
                    "Review unfavorable budget variance",
                    "Finance",
                    money(total_variance),
                ]
            )

        if not cost_summary_df.empty:

            top_category = (
                cost_summary_df.iloc[0][
                    "Account / Cost Category"
                ]
            )

            top_value = (
                cost_summary_df.iloc[0][
                    "Variance"
                ]
            )

            actions.append(
                [
                    "P1",
                    f"Investigate {top_category} variance",
                    "Finance + Operations",
                    money(top_value),
                ]
            )

        actions.append(
            [
                "P2",
                "Refresh rolling forecast",
                "FP&A",
                "Management cycle",
            ]
        )

        action_df = pd.DataFrame(
            actions,
            columns=[
                "Priority",
                "Action",
                "Owner",
                "Impact",
            ],
        )

        st.dataframe(
            action_df,
            hide_index=True,
            use_container_width=True,
        )

    with col2:

        st.markdown(
            "### Recent Alerts"
        )

        if total_variance > 0:

            st.markdown(
                dedent(
                    f"""
                    <div class="alert-box alert-critical">
                        <div class="alert-title">
                            🔴 High budget variance
                        </div>
                        <div class="alert-text">
                            Actual cost is above budget by {money(total_variance)}.
                        </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="alert-box alert-warning">
                <div class="alert-title">
                    🟠 Forecast review
                </div>
                <div class="alert-text">
                    Review assumptions before the next planning cycle.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# AI CFO
# ============================================================

elif page == "🤖 AI CFO":

    render_hero(
        "AI CFO",
        "Evidence-based financial reasoning, risk identification and management actions.",
    )

    st.markdown(
        "## AI CFO Assistant"
    )

    question = st.text_input(
        "Ask a finance question",
        placeholder="Why are costs above budget?",
    )

    if question:

        cost = (
            filtered_df
            .groupby(
                "Account / Cost Category"
            )["Variance"]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        bu = (
            filtered_df
            .groupby(
                "Business Unit"
            )["Variance"]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        driver = (
            cost.index[0]
            if not cost.empty
            else "No material driver"
        )

        driver_value = (
            cost.iloc[0]
            if not cost.empty
            else 0
        )

        top_bu = (
            bu.index[0]
            if not bu.empty
            else "No business unit"
        )

        top_bu_value = (
            bu.iloc[0]
            if not bu.empty
            else 0
        )

        render_ai_box(
            "FinSight AI CFO Analysis",
            f"{driver} is currently the largest unfavorable cost driver.",
            f"{driver} variance is {money(driver_value)}. {top_bu} has the highest business-unit variance at {money(top_bu_value)}.",
            "Persistent overspend can reduce margin and create forecast pressure.",
            "Review cost ownership, vendor commitments, utilization and forecast assumptions.",
        )

    else:

        st.info(
            "Ask the AI CFO a financial question."
        )


# ============================================================
# FINANCIAL PERFORMANCE
# ============================================================

elif page == "📊 Financial Performance":

    render_hero(
        "Financial Performance",
        "Business-unit, department and cost-category performance intelligence.",
    )

    st.markdown(
        "## Financial Performance"
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Business Units",
            "Departments",
            "Cost Categories",
        ]
    )

    with tab1:

        data = (
            filtered_df
            .groupby(
                "Business Unit"
            )
            .agg(
                Revenue=("Revenue", "sum"),
                Budget=("Budget", "sum"),
                Actual=("Actual", "sum"),
                Variance=("Variance", "sum"),
                Profit=("Profit", "sum"),
            )
            .reset_index()
        )

        st.dataframe(
            data.style.format(
                {
                    "Revenue": money_full,
                    "Budget": money_full,
                    "Actual": money_full,
                    "Variance": money_full,
                    "Profit": money_full,
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    with tab2:

        data = (
            filtered_df
            .groupby(
                "Department"
            )
            .agg(
                Revenue=("Revenue", "sum"),
                Budget=("Budget", "sum"),
                Actual=("Actual", "sum"),
                Variance=("Variance", "sum"),
                Profit=("Profit", "sum"),
            )
            .reset_index()
        )

        st.dataframe(
            data.style.format(
                {
                    "Revenue": money_full,
                    "Budget": money_full,
                    "Actual": money_full,
                    "Variance": money_full,
                    "Profit": money_full,
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    with tab3:

        data = (
            filtered_df
            .groupby(
                "Account / Cost Category"
            )
            .agg(
                Budget=("Budget", "sum"),
                Actual=("Actual", "sum"),
                Variance=("Variance", "sum"),
            )
            .reset_index()
            .sort_values(
                "Variance",
                ascending=False,
            )
        )

        st.dataframe(
            data.style.format(
                {
                    "Budget": money_full,
                    "Actual": money_full,
                    "Variance": money_full,
                }
            ),
            hide_index=True,
            use_container_width=True,
        )


# ============================================================
# CASH FLOW
# ============================================================

elif page == "💵 Cash Flow Center":

    render_hero(
        "Cash Flow Center",
        "Modeled liquidity, inflows, outflows and cash runway intelligence.",
    )

    st.warning(
        "Cash-flow values are modeled from the current ERP-style dataset. Live bank, receivables and payables data are not connected in this portfolio prototype."
    )

    opening_cash = (
        total_revenue * .08
    )

    cash = (
        filtered_df
        .dropna(
            subset=["Date"]
        )
        .assign(
            Month=lambda x:
            x["Date"]
            .dt
            .to_period("M")
            .astype(str)
        )
        .groupby("Month")
        .agg(
            Inflow=("Revenue", "sum"),
            Outflow=("Actual", "sum"),
        )
        .reset_index()
    )

    cash["Net Cash"] = (
        cash["Inflow"] -
        cash["Outflow"]
    )

    cash["Closing Cash"] = (
        opening_cash +
        cash["Net Cash"].cumsum()
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        render_kpi(
            "Opening Cash",
            money(opening_cash),
            "Modeled",
            "blue",
        )

    with c2:

        render_kpi(
            "Latest Closing Cash",
            money(
                cash[
                    "Closing Cash"
                ].iloc[-1]
            ),
            "Modeled",
            "green",
        )

    with c3:

        render_kpi(
            "Net Cash Movement",
            money(
                cash["Net Cash"].sum()
            ),
            "Inflow less outflow",
            "cyan",
        )

    st.markdown(
        "### Cash Flow Trend"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=cash["Month"],
            y=cash["Inflow"],
            mode="lines+markers",
            name="Inflow",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=cash["Month"],
            y=cash["Outflow"],
            mode="lines+markers",
            name="Outflow",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=cash["Month"],
            y=cash["Closing Cash"],
            mode="lines+markers",
            name="Closing Cash",
        )
    )

    fig.update_layout(
        height=430
    )

    st.plotly_chart(
        dark_chart(fig),
        use_container_width=True,
    )


# ============================================================
# BUDGET VS ACTUALS
# ============================================================

elif page == "🎯 Budget vs Actuals":

    render_hero(
        "Budget vs Actuals",
        "Variance analysis across business units, departments and cost categories.",
    )

    st.markdown(
        "## Budget vs Actuals"
    )

    data = (
        filtered_df
        .groupby(
            [
                "Business Unit",
                "Department",
                "Account / Cost Category",
            ]
        )
        .agg(
            Budget=("Budget", "sum"),
            Actual=("Actual", "sum"),
            Variance=("Variance", "sum"),
        )
        .reset_index()
    )

    data["Variance %"] = np.where(
        data["Budget"] != 0,
        data["Variance"] /
        data["Budget"] *
        100,
        0,
    )

    st.dataframe(
        data.style.format(
            {
                "Budget": money_full,
                "Actual": money_full,
                "Variance": money_full,
                "Variance %": "{:.2f}%",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# FORECAST
# ============================================================

elif page == "📈 Forecasting & Planning":

    render_hero(
        "Forecasting & Planning",
        "Forward-looking revenue planning and financial scenario intelligence.",
    )

    st.markdown(
        "## Revenue Forecast"
    )

    monthly = (
        filtered_df
        .dropna(
            subset=["Date"]
        )
        .assign(
            Month=lambda x:
            x["Date"]
            .dt
            .to_period("M")
            .astype(str)
        )
        .groupby("Month")[
            "Revenue"
        ]
        .sum()
        .reset_index()
    )

    if len(monthly) >= 3:

        monthly["Index"] = np.arange(
            len(monthly)
        )

        model = LinearRegression()

        model.fit(
            monthly[["Index"]],
            monthly["Revenue"]
        )

        future_index = np.arange(
            len(monthly),
            len(monthly) + 6
        )

        future = pd.DataFrame(
            {
                "Index": future_index
            }
        )

        base = model.predict(
            future
        )

        last_actual = (
            monthly["Revenue"]
            .iloc[-1]
        )

        if len(monthly) >= 2:

            growth = (
                monthly["Revenue"]
                .iloc[-1]
                /
                monthly["Revenue"]
                .iloc[-2]
                -
                1
            )

        else:

            growth = .01

        growth_curve = (
            last_actual *
            (
                1 + growth
            )
            **
            np.arange(
                1,
                7
            )
        )

        base = np.maximum(
            base,
            growth_curve
        )

        optimistic = (
            base * 1.08
        )

        downside = (
            base * .92
        )

        future_dates = pd.date_range(
            pd.to_datetime(
                monthly["Month"]
                .iloc[-1]
            )
            +
            pd.offsets.MonthBegin(1),
            periods=6,
            freq="MS",
        )

        forecast = pd.DataFrame(
            {
                "Month": future_dates,
                "Base": base,
                "Optimistic": optimistic,
                "Downside": downside,
            }
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(
                    monthly["Month"]
                ),
                y=monthly["Revenue"],
                mode="lines+markers",
                name="Historical",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast["Month"],
                y=forecast["Base"],
                mode="lines+markers",
                name="Base Case",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast["Month"],
                y=forecast["Optimistic"],
                mode="lines",
                name="Optimistic",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast["Month"],
                y=forecast["Downside"],
                mode="lines",
                name="Downside",
            )
        )

        fig.update_layout(
            height=450,
            title="Revenue Forecast — 6 Month Horizon",
        )

        st.plotly_chart(
            dark_chart(fig),
            use_container_width=True,
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            render_kpi(
                "Base Case",
                money(
                    forecast["Base"].sum()
                ),
                "6-month revenue",
                "blue",
            )

        with c2:

            render_kpi(
                "Optimistic",
                money(
                    forecast[
                        "Optimistic"
                    ].sum()
                ),
                "+8% scenario",
                "green",
            )

        with c3:

            render_kpi(
                "Downside",
                money(
                    forecast[
                        "Downside"
                    ].sum()
                ),
                "-8% scenario",
                "orange",
            )

    else:

        st.warning(
            "Not enough monthly data for forecasting."
        )


# ============================================================
# WHAT IF
# ============================================================

elif page == "🔮 What-if Scenarios":

    render_hero(
        "What-if Scenarios",
        "Management planning simulator for revenue, cost and margin decisions.",
    )

    st.markdown(
        "## Scenario Planner"
    )

    revenue_change = st.slider(
        "Revenue change",
        -20,
        20,
        0,
        1,
        format="%d%%",
    )

    cost_change = st.slider(
        "Cost change",
        -20,
        20,
        0,
        1,
        format="%d%%",
    )

    scenario_revenue = (
        total_revenue *
        (
            1 +
            revenue_change /
            100
        )
    )

    scenario_cost = (
        total_actual *
        (
            1 +
            cost_change /
            100
        )
    )

    scenario_profit = (
        scenario_revenue -
        scenario_cost
    )

    scenario_margin = (
        scenario_profit /
        scenario_revenue *
        100
        if scenario_revenue
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        render_kpi(
            "Scenario Revenue",
            money(
                scenario_revenue
            ),
            f"{revenue_change:+d}% change",
            "blue",
        )

    with c2:

        render_kpi(
            "Scenario Cost",
            money(
                scenario_cost
            ),
            f"{cost_change:+d}% change",
            "orange",
        )

    with c3:

        render_kpi(
            "Scenario Profit",
            money(
                scenario_profit
            ),
            "Revenue minus cost",
            "green",
        )

    with c4:

        render_kpi(
            "Scenario Margin",
            pct(
                scenario_margin
            ),
            "Projected margin",
            "purple",
        )

    comparison = pd.DataFrame(
        {
            "Scenario": [
                "Current",
                "What-if",
            ],
            "Revenue": [
                total_revenue,
                scenario_revenue,
            ],
            "Cost": [
                total_actual,
                scenario_cost,
            ],
            "Profit": [
                total_profit,
                scenario_profit,
            ],
        }
    )

    fig = go.Figure()

    for metric in [
        "Revenue",
        "Cost",
        "Profit",
    ]:

        fig.add_trace(
            go.Bar(
                x=comparison["Scenario"],
                y=comparison[metric],
                name=metric,
            )
        )

    fig.update_layout(
        barmode="group",
        height=400,
    )

    st.plotly_chart(
        dark_chart(fig),
        use_container_width=True,
    )


# ============================================================
# RISK
# ============================================================

elif page == "⚠️ Risk & Anomaly Detection":

    render_hero(
        "Risk & Anomaly Detection",
        "Statistical anomaly detection and financial risk monitoring.",
    )

    risk_df = filtered_df.copy()

    if len(risk_df) >= 10:

        model = IsolationForest(
            contamination=.03,
            random_state=42,
        )

        prediction = model.fit_predict(
            risk_df[
                [
                    "Budget",
                    "Actual",
                    "Revenue",
                ]
            ].fillna(0)
        )

        risk_df["Model Anomaly"] = np.where(
            prediction == -1,
            "Potential Anomaly",
            "Normal",
        )

    else:

        risk_df[
            "Model Anomaly"
        ] = "Insufficient Data"

    anomalies = risk_df[
        risk_df[
            "Model Anomaly"
        ]
        == "Potential Anomaly"
    ]

    c1, c2, c3 = st.columns(3)

    with c1:

        render_kpi(
            "Transactions",
            f"{len(risk_df):,}",
            "Selected population",
            "blue",
        )

    with c2:

        render_kpi(
            "Potential Anomalies",
            f"{len(anomalies):,}",
            "Isolation Forest",
            "orange",
        )

    with c3:

        render_kpi(
            "Anomaly Rate",
            pct(
                len(anomalies) /
                len(risk_df) *
                100
                if len(risk_df)
                else 0
            ),
            "Statistical screening",
            "yellow",
        )

    st.dataframe(
        anomalies[
            [
                "Transaction ID",
                "Date",
                "Business Unit",
                "Department",
                "Account / Cost Category",
                "Budget",
                "Actual",
                "Variance",
                "Variance %",
                "Model Anomaly",
            ]
        ].head(100),
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# COST INTELLIGENCE
# ============================================================

elif page == "💡 Cost Intelligence":

    render_hero(
        "Cost Intelligence",
        "Identify the categories, departments and business units driving financial variance.",
    )

    data = (
        filtered_df
        .groupby(
            "Account / Cost Category"
        )
        .agg(
            Budget=("Budget", "sum"),
            Actual=("Actual", "sum"),
            Variance=("Variance", "sum"),
        )
        .reset_index()
        .sort_values(
            "Variance",
            ascending=False,
        )
    )

    data["Variance %"] = np.where(
        data["Budget"] != 0,
        data["Variance"] /
        data["Budget"] *
        100,
        0,
    )

    st.dataframe(
        data.style.format(
            {
                "Budget": money_full,
                "Actual": money_full,
                "Variance": money_full,
                "Variance %": "{:.2f}%",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    fig = px.bar(
        data.head(10),
        x="Variance",
        y="Account / Cost Category",
        orientation="h",
    )

    fig.update_layout(
        height=450
    )

    st.plotly_chart(
        dark_chart(fig),
        use_container_width=True,
    )


# ============================================================
# MANAGEMENT ACTIONS
# ============================================================

elif page == "🎯 Management Actions":

    render_hero(
        "Management Actions",
        "Translate financial findings into accountable management actions.",
    )

    st.markdown(
        "## AI CFO Action Center"
    )

    data = (
        filtered_df
        .groupby(
            "Account / Cost Category"
        )["Variance"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    for category, variance in data.head(8).items():

        priority = (
            "P1"
            if variance > 0
            else "P2"
        )

        cls = (
            "priority-p1"
            if priority == "P1"
            else "priority-p2"
        )

        st.markdown(
            dedent(
                f"""
                <div class="action-card">

                    <span class="{cls}">
                        {priority}
                    </span>

                    &nbsp;&nbsp;

                    <strong style="color:#e5edf7;">
                        Review {category} variance
                    </strong>

                    <br>

                    <span style="color:#8191a8;font-size:.75rem;">
                        Owner: Finance / FP&A
                        &nbsp; | &nbsp;
                        Impact: {money(variance)}
                    </span>

                </div>
                """
            ),
            unsafe_allow_html=True,
        )


# ============================================================
# REPORTS
# ============================================================

elif page == "📑 Reports Library":

    render_hero(
        "Reports Library",
        "Management-ready financial reports and analysis views.",
    )

    reports = pd.DataFrame(
        {
            "Report": [
                "Executive Financial Summary",
                "Budget vs Actuals",
                "Business Unit Performance",
                "Department Performance",
                "Cost Intelligence",
                "Risk & Anomaly Report",
                "Forecast & Scenario Report",
            ],
            "Status": [
                "Available",
                "Available",
                "Available",
                "Available",
                "Available",
                "Available",
                "Available",
            ],
        }
    )

    st.dataframe(
        reports,
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "⬇️ Download Financial Data",
        filtered_df.to_csv(
            index=False
        ).encode("utf-8"),
        "finsight_financial_data.csv",
        "text/csv",
    )


# ============================================================
# UPLOAD
# ============================================================

elif page == "📤 Upload Data":

    render_hero(
        "Upload Data",
        "Connect ERP-style CSV financial data to the FinSight intelligence layer.",
    )

    st.markdown(
        "## ERP Data Upload"
    )

    st.info(
        "Upload your ERP CSV using the sidebar. FinSight automatically normalizes common finance column names and derives department ownership where department data is unavailable."
    )

    st.dataframe(
        df.head(25),
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# ERP CONNECTIONS
# ============================================================

elif page == "🔗 ERP Connections":

    render_hero(
        "ERP Connections",
        "Enterprise integration architecture for future ERP and finance-system connectivity.",
    )

    systems = pd.DataFrame(
        {
            "System": [
                "ERP CSV",
                "SAP",
                "Oracle ERP",
                "Microsoft Dynamics",
                "QuickBooks",
                "Zoho Books",
            ],
            "Status": [
                "Active Demo",
                "Architecture Ready",
                "Architecture Ready",
                "Architecture Ready",
                "Architecture Ready",
                "Architecture Ready",
            ],
        }
    )

    st.dataframe(
        systems,
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# DATA MAPPING
# ============================================================

elif page == "🧩 Data Mapping":

    render_hero(
        "Data Mapping",
        "Normalize ERP data structures into a consistent FP&A analytical model.",
    )

    mapping = pd.DataFrame(
        {
            "FinSight Field": required_columns,
            "Detected": [
                "Yes"
                if col in df.columns
                else "Derived"
                if col == "Department"
                else "No"
                for col in required_columns
            ],
        }
    )

    st.dataframe(
        mapping,
        hide_index=True,
        use_container_width=True,
    )

    st.markdown(
        "### Department Intelligence"
    )

    dept_mapping = pd.DataFrame(
        {
            "Cost Category": [
                "Cloud",
                "Software",
                "Facilities",
                "Marketing",
                "Payroll",
                "Professional Services",
                "Travel",
            ],
            "Mapped Department": [
                "Engineering",
                "Engineering",
                "Operations",
                "Marketing",
                "HR",
                "Finance",
                "Sales",
            ],
        }
    )

    st.dataframe(
        dept_mapping,
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# DATA QUALITY
# ============================================================

elif page == "✅ Data Quality":

    render_hero(
        "Data Quality",
        "Financial data validation and analytical readiness checks.",
    )

    quality = []

    for col in required_columns:

        missing_count = int(
            df[col].isna().sum()
        )

        completeness = (
            100 -
            (
                missing_count /
                len(df) *
                100
            )
            if len(df)
            else 0
        )

        quality.append(
            {
                "Field": col,
                "Missing Records": missing_count,
                "Completeness %": completeness,
            }
        )

    quality_df = pd.DataFrame(
        quality
    )

    st.dataframe(
        quality_df.style.format(
            {
                "Completeness %":
                "{:.2f}%"
            }
        ),
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# ALERTS
# ============================================================

elif page == "🔔 Alerts & Notifications":

    render_hero(
        "Alerts & Notifications",
        "Monitor financial conditions requiring management attention.",
    )

    if total_variance > 0:

        st.markdown(
            dedent(
                f"""
                <div class="alert-box alert-critical">
                    <div class="alert-title">
                        🔴 Unfavorable budget variance
                    </div>
                    <div class="alert-text">
                        Actual cost exceeds budget by {money(total_variance)}.
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="alert-box alert-good">
                <div class="alert-title">
                    🟢 Budget position stable
                </div>
                <div class="alert-text">
                    No aggregate unfavorable variance detected.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if profit_margin < 50:

        st.markdown(
            """
            <div class="alert-box alert-warning">
                <div class="alert-title">
                    🟠 Margin pressure
                </div>
                <div class="alert-text">
                    Profit margin is below 50%.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# SETTINGS
# ============================================================

elif page == "⚙️ Settings":

    render_hero(
        "Settings",
        "FinSight AI application and analytical configuration.",
    )

    st.markdown(
        "## Application Settings"
    )

    st.checkbox(
        "Enable AI CFO recommendations",
        value=True,
    )

    st.checkbox(
        "Enable anomaly detection",
        value=True,
    )

    st.checkbox(
        "Enable forecast analysis",
        value=True,
    )

    st.checkbox(
        "Show modeled cash-flow indicators",
        value=True,
    )

    st.info(
        "These are portfolio-prototype controls. Production authentication, role-based access and enterprise configuration can be added later."
    )


# ============================================================
# AUDIT LOGS
# ============================================================

elif page == "🧾 Audit Logs":

    render_hero(
        "Audit Logs",
        "Prototype activity and analytical traceability view.",
    )

    audit = pd.DataFrame(
        {
            "Timestamp": [
                pd.Timestamp.now(),
                pd.Timestamp.now(),
                pd.Timestamp.now(),
            ],
            "Event": [
                "Financial dataset loaded",
                "Executive dashboard analyzed",
                "AI CFO analysis generated",
            ],
            "Status": [
                "Success",
                "Success",
                "Success",
            ],
        }
    )

    st.dataframe(
        audit,
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "🔎 Data Explorer":

    render_hero(
        "Data Explorer",
        "Explore the normalized financial dataset powering FinSight AI.",
    )

    st.markdown(
        f"### {len(filtered_df):,} records"
    )

    st.dataframe(
        filtered_df,
        hide_index=True,
        use_container_width=True,
        height=600,
    )

    st.download_button(
        "⬇️ Download Selected Data",
        filtered_df.to_csv(
            index=False
        ).encode("utf-8"),
        "finsight_selected_data.csv",
        "text/csv",
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="finsight-footer">
        <strong>FinSight AI</strong>
        &nbsp; • &nbsp;
        Agentic FP&A & ERP Intelligence
        <br>
        Portfolio prototype using synthetic ERP-style financial data.
        Modeled components are identified where source data is unavailable.
    </div>
    """,
    unsafe_allow_html=True,
)
