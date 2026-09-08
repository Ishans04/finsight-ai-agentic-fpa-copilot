import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression


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
# PREMIUM DARK ENTERPRISE THEME
# ============================================================

st.markdown(
    """
<style>

/* ============================================================
   GLOBAL
   ============================================================ */

.stApp {
    background:
        radial-gradient(
            circle at 85% 5%,
            rgba(37, 99, 235, 0.10),
            transparent 24%
        ),
        radial-gradient(
            circle at 15% 90%,
            rgba(124, 58, 237, 0.08),
            transparent 28%
        ),
        #050b14;

    color: #e8eef8;
}

.main {
    background: transparent;
}

.block-container {
    max-width: 1500px;
    padding-top: 1rem;
    padding-bottom: 3rem;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #07101d 0%,
            #081321 55%,
            #050b14 100%
        );

    border-right: 1px solid #17263b;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1rem;
}

section[data-testid="stSidebar"] hr {
    border-color: #1a2940;
}

section[data-testid="stSidebar"] label {
    color: #b8c5d8 !important;
}

section[data-testid="stSidebar"] .stRadio label {
    color: #d7e0ee !important;
}


/* ============================================================
   TYPOGRAPHY
   ============================================================ */

h1, h2, h3, h4 {
    color: #f4f7fb !important;
    font-weight: 750 !important;
    letter-spacing: -0.4px;
}

p {
    color: #9aa9bd;
}

[data-testid="stCaptionContainer"] {
    color: #718198 !important;
}


/* ============================================================
   FIN SIGHT BRAND
   ============================================================ */

.brand-box {
    padding: 0.25rem 0 0.7rem 0;
}

.brand-title {
    font-size: 1.45rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.6px;
}

.brand-title span {
    color: #60a5fa;
}

.brand-subtitle {
    font-size: 0.72rem;
    color: #8191a8;
    margin-top: -2px;
}


/* ============================================================
   TOP SEARCH
   ============================================================ */

.search-box {
    background: #081321;
    border: 1px solid #1c3048;
    border-radius: 9px;
    padding: 0.65rem 0.9rem;
    color: #75869d;
    font-size: 0.85rem;
    margin-bottom: 1rem;
}


/* ============================================================
   HERO
   ============================================================ */

.hero {
    position: relative;

    padding: 1.45rem 1.8rem;

    border-radius: 16px;

    background:
        radial-gradient(
            circle at 85% 15%,
            rgba(6, 182, 212, 0.23),
            transparent 28%
        ),
        linear-gradient(
            135deg,
            #0b1730 0%,
            #10264a 50%,
            #075d68 100%
        );

    border: 1px solid #244767;

    margin-bottom: 1.25rem;

    box-shadow:
        0 15px 45px rgba(0, 0, 0, 0.30);
}

.hero-title {
    color: white;
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.35rem;
}

.hero-subtitle {
    color: #c9d6e6;
    font-size: 0.95rem;
}


/* ============================================================
   KPI CARDS
   ============================================================ */

.kpi-card {
    background:
        linear-gradient(
            145deg,
            #0c1727,
            #091321
        );

    border: 1px solid #1b3049;

    border-radius: 13px;

    padding: 1rem 1.05rem;

    min-height: 112px;

    box-shadow:
        0 7px 22px rgba(0, 0, 0, 0.22);

    transition: 0.18s ease;
}

.kpi-card:hover {
    transform: translateY(-2px);
    border-color: #315d91;
}

.kpi-label {
    color: #91a2b8;
    font-size: 0.78rem;
    margin-bottom: 0.35rem;
}

.kpi-value {
    color: #f8fafc;
    font-size: 1.55rem;
    font-weight: 800;
    line-height: 1.15;
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
    margin-top: 0.35rem;
    font-size: 0.7rem;
    color: #6ee7b7;
}

.kpi-delta.warn {
    color: #fbbf24;
}

.kpi-delta.bad {
    color: #fb7185;
}


/* ============================================================
   SECTION CARDS
   ============================================================ */

.panel {
    background:
        linear-gradient(
            145deg,
            #0b1625,
            #08121f
        );

    border: 1px solid #1a2c43;

    border-radius: 12px;

    padding: 1rem 1.1rem;

    margin-bottom: 1rem;

    box-shadow:
        0 7px 22px rgba(0, 0, 0, 0.18);
}

.panel-title {
    color: #edf3fa;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.15rem;
}

.panel-subtitle {
    color: #718198;
    font-size: 0.72rem;
    margin-bottom: 0.6rem;
}


/* ============================================================
   AI CFO BOX
   ============================================================ */

.ai-box {
    background:
        linear-gradient(
            145deg,
            #101638,
            #0b1227
        );

    border: 1px solid #303b76;

    border-left: 4px solid #6366f1;

    border-radius: 12px;

    padding: 1rem 1.15rem;

    margin-bottom: 0.8rem;

    box-shadow:
        0 8px 24px rgba(49, 46, 129, 0.18);
}

.ai-title {
    color: #c4b5fd;
    font-weight: 750;
    font-size: 0.95rem;
}

.ai-label {
    color: #8ea0b9;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

.ai-text {
    color: #dce5f1;
    font-size: 0.82rem;
    line-height: 1.5;
}


/* ============================================================
   ALERTS
   ============================================================ */

.alert-box {
    padding: 0.85rem 1rem;
    border-radius: 9px;
    margin-bottom: 0.6rem;
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
    font-size: 0.8rem;
}

.alert-text {
    color: #8292a8;
    font-size: 0.72rem;
    margin-top: 0.15rem;
}


/* ============================================================
   ACTION CENTER
   ============================================================ */

.action-card {
    background: #0a1523;
    border: 1px solid #1d3047;
    border-radius: 9px;
    padding: 0.8rem 0.9rem;
    margin-bottom: 0.55rem;
}

.priority-p0 {
    color: #fb7185;
    font-weight: 800;
}

.priority-p1 {
    color: #fbbf24;
    font-weight: 800;
}

.priority-p2 {
    color: #4ade80;
    font-weight: 800;
}


/* ============================================================
   STREAMLIT METRIC FALLBACK
   ============================================================ */

div[data-testid="stMetric"] {
    background: #0b1625 !important;
    border: 1px solid #1b3049 !important;
    border-radius: 12px !important;
    padding: 0.85rem !important;
    overflow: visible !important;
}

div[data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-size: 1.5rem !important;
    font-weight: 750 !important;
    white-space: nowrap !important;
}


/* ============================================================
   INPUTS
   ============================================================ */

div[data-baseweb="select"] > div {
    background-color: #081321 !important;
    border-color: #1c3048 !important;
    color: #e2e8f0 !important;
}

div[data-baseweb="input"] {
    background-color: #081321 !important;
}

div[data-baseweb="input"] input {
    color: #e2e8f0 !important;
}

.stTextInput input,
.stTextArea textarea {
    background: #081321 !important;
    color: #e2e8f0 !important;
    border: 1px solid #1c3048 !important;
}


/* ============================================================
   BUTTONS
   ============================================================ */

.stButton > button {
    background:
        linear-gradient(
            135deg,
            #2563eb,
            #4f46e5
        ) !important;

    color: white !important;

    border: 1px solid #4f7cff !important;

    border-radius: 8px !important;

    font-weight: 650 !important;
}

.stButton > button:hover {
    border-color: #7dd3fc !important;
}


/* ============================================================
   FILE UPLOADER
   ============================================================ */

section[data-testid="stFileUploaderDropzone"] {
    background: #081321 !important;
    border: 1px dashed #29415f !important;
    border-radius: 10px !important;
}


/* ============================================================
   DATAFRAME
   ============================================================ */

div[data-testid="stDataFrame"] {
    border: 1px solid #1b3049;
    border-radius: 10px;
    overflow: hidden;
}


/* ============================================================
   EXPANDERS
   ============================================================ */

div[data-testid="stExpander"] {
    background: #081321 !important;
    border: 1px solid #1b3049 !important;
    border-radius: 10px !important;
}


/* ============================================================
   FOOTER
   ============================================================ */

.finsight-footer {
    margin-top: 2rem;
    padding: 1rem;
    border-top: 1px solid #1a2940;
    text-align: center;
    color: #5f7087;
    font-size: 0.72rem;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# FORMATTING FUNCTIONS
# ============================================================

def money(value):
    """Compact INR formatting."""
    try:
        value = float(value)

        abs_value = abs(value)

        if abs_value >= 10_000_000:
            return f"₹{value / 10_000_000:,.2f} Cr"

        if abs_value >= 100_000:
            return f"₹{value / 100_000:,.2f} L"

        return f"₹{value:,.0f}"

    except Exception:
        return "₹0"


def money_full(value):
    """Full INR formatting."""
    try:
        return f"₹{float(value):,.0f}"
    except Exception:
        return "₹0"


def pct(value):
    try:
        return f"{float(value):.2f}%"
    except Exception:
        return "0.00%"


def safe_num(value):
    try:
        return float(value)
    except Exception:
        return 0.0


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


def smart_column_mapping(df):
    """
    Maps common ERP / finance column variations
    to FinSight's normalized schema.
    """

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
            "business_unit_name",
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
            "account_cost_category",
            "cost_category",
            "costcategory",
            "gl_account",
            "gl_account_name",
            "gl",
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
            "outlier_flag",
        ],

        "Anomaly": [
            "anomaly",
            "anomaly_status",
            "risk_flag",
        ],
    }

    normalized = {
        normalize_name(c): c
        for c in df.columns
    }

    rename_map = {}

    for target, candidates in aliases.items():

        if target in df.columns:
            continue

        for candidate in candidates:

            if candidate in normalized:

                source = normalized[candidate]

                rename_map[source] = target

                break

    df = df.rename(columns=rename_map)

    return df


# ============================================================
# DATA LOADER
# ============================================================

@st.cache_data
def load_default_data():

    possible_files = [
        Path("synthetic_erp_financials.csv"),
        Path("data/synthetic_erp_financials.csv"),
    ]

    file_path = None

    for path in possible_files:
        if path.exists():
            file_path = path
            break

    if file_path is None:
        return None

    df = pd.read_csv(file_path)

    return prepare_data(df)


def prepare_data(df):

    df = df.copy()

    df = smart_column_mapping(df)

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Required text fields
    # --------------------------------------------------------

    for col in [
        "Business Unit",
        "Department",
        "Account / Cost Category",
    ]:

        if col not in df.columns:

            if col == "Department":
                df[col] = "Unmapped"

            else:
                df[col] = "Unmapped"

        df[col] = (
            df[col]
            .fillna("Unmapped")
            .astype(str)
        )

    # --------------------------------------------------------
    # Numeric fields
    # --------------------------------------------------------

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
        ).fillna(0.0)

    # --------------------------------------------------------
    # Calculations
    # --------------------------------------------------------

    if (
        "Variance" not in df.columns
        or df["Variance"].eq(0).all()
    ):
        df["Variance"] = (
            df["Actual"] -
            df["Budget"]
        )

    if (
        "Variance %" not in df.columns
        or df["Variance %"].eq(0).all()
    ):
        df["Variance %"] = np.where(
            df["Budget"] != 0,
            (
                df["Variance"] /
                df["Budget"]
            ) * 100,
            0,
        )

    if (
        "Profit" not in df.columns
        or df["Profit"].eq(0).all()
    ):
        df["Profit"] = (
            df["Revenue"] -
            df["Actual"]
        )

    # --------------------------------------------------------
    # Transaction ID
    # --------------------------------------------------------

    if "Transaction ID" not in df.columns:

        df["Transaction ID"] = [
            f"TXN-{i + 1:05d}"
            for i in range(len(df))
        ]

    # --------------------------------------------------------
    # Anomaly fields
    # --------------------------------------------------------

    if "Anomaly Flag" not in df.columns:
        df["Anomaly Flag"] = 0

    if "Anomaly" not in df.columns:
        df["Anomaly"] = "Normal"

    return df


# ============================================================
# LOAD DATA
# ============================================================

default_df = load_default_data()


# ============================================================
# SIDEBAR BRAND
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

    # --------------------------------------------------------
    # DATA SOURCE
    # --------------------------------------------------------

    st.markdown("### Data Source")

    uploaded_file = st.file_uploader(
        "Upload ERP CSV",
        type=["csv"],
        help="Upload ERP-style financial data in CSV format.",
    )

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)
            df = prepare_data(df)

            source_label = "Uploaded ERP Data"

        except Exception as e:

            st.error(
                f"Unable to read uploaded file: {e}"
            )

            df = default_df.copy() if default_df is not None else None

            source_label = "FinSight Demo ERP Data"

    else:

        df = (
            default_df.copy()
            if default_df is not None
            else None
        )

        source_label = "FinSight Demo ERP Data"

    st.caption(f"Source: {source_label}")

    st.divider()

    # --------------------------------------------------------
    # NAVIGATION
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # GLOBAL FILTERS
    # --------------------------------------------------------

    st.markdown("### Global Filters")

    if df is not None:

        bu_options = ["All"] + sorted(
            df["Business Unit"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_bu = st.selectbox(
            "Business Unit",
            bu_options,
        )

        dept_options = ["All"] + sorted(
            df["Department"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_department = st.selectbox(
            "Department",
            dept_options,
        )

    else:

        selected_bu = "All"
        selected_department = "All"


# ============================================================
# DATA VALIDATION
# ============================================================

if df is None:

    st.error(
        "No financial dataset was found. "
        "Please upload an ERP CSV."
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

    st.info(
        "FinSight accepts common ERP column names such as "
        "date, business_unit, department, account, "
        "cost_category, budget, actual and revenue."
    )

    st.stop()


# ============================================================
# GLOBAL FILTER
# ============================================================

filtered_df = df.copy()

if selected_bu != "All":

    filtered_df = filtered_df[
        filtered_df["Business Unit"] ==
        selected_bu
    ]

if selected_department != "All":

    filtered_df = filtered_df[
        filtered_df["Department"] ==
        selected_department
    ]


# ============================================================
# COMMON CALCULATIONS
# ============================================================

total_revenue = filtered_df["Revenue"].sum()

total_actual = filtered_df["Actual"].sum()

total_budget = filtered_df["Budget"].sum()

total_variance = filtered_df["Variance"].sum()

total_profit = filtered_df["Profit"].sum()

profit_margin = (
    total_profit / total_revenue * 100
    if total_revenue != 0
    else 0
)

budget_variance_pct = (
    total_variance / total_budget * 100
    if total_budget != 0
    else 0
)


# ============================================================
# COMMON HERO
# ============================================================

def render_hero(title, subtitle):

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">
                📊 {title}
            </div>
            <div class="hero-subtitle">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# KPI CARD
# ============================================================

def kpi_card(
    label,
    value,
    delta="",
    color="blue",
    delta_type="good",
):

    delta_class = ""

    if delta_type == "warn":
        delta_class = "warn"

    elif delta_type == "bad":
        delta_class = "bad"

    st.markdown(
        f"""
        <div class="kpi-card">

            <div class="kpi-label">
                {label}
            </div>

            <div class="kpi-value {color}">
                {value}
            </div>

            <div class="kpi-delta {delta_class}">
                {delta}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CHART THEME
# ============================================================

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
            t=25,
            b=10,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(
                color="#aebbd0"
            ),
        ),
    )

    fig.update_xaxes(
        gridcolor="#18283d",
        zerolinecolor="#18283d",
    )

    fig.update_yaxes(
        gridcolor="#18283d",
        zerolinecolor="#18283d",
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

    # --------------------------------------------------------
    # TOP SEARCH
    # --------------------------------------------------------

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

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        kpi_card(
            "Revenue",
            money(total_revenue),
            "▲ Financial performance",
            "blue",
        )

    with c2:
        gross_profit = total_profit

        kpi_card(
            "Gross Profit",
            money(gross_profit),
            f"Margin {pct(profit_margin)}",
            "green",
        )

    with c3:
        ebitda = total_profit * 0.72

        kpi_card(
            "EBITDA",
            money(ebitda),
            "Modeled from operating profit",
            "purple",
        )

    with c4:
        modeled_cash = max(
            total_revenue * 0.19,
            0
        )

        kpi_card(
            "Cash Balance",
            money(modeled_cash),
            "Modeled liquidity view",
            "orange",
        )

    with c5:

        kpi_card(
            "Profit Margin",
            pct(profit_margin),
            "Revenue profitability",
            "cyan",
        )

    with c6:

        risk_score = min(
            100,
            max(
                0,
                int(
                    abs(budget_variance_pct) * 3
                    + filtered_df["Anomaly Flag"].sum()
                    if "Anomaly Flag" in filtered_df.columns
                    else abs(budget_variance_pct) * 3
                ),
            ),
        )

        risk_level = (
            "High"
            if risk_score >= 70
            else "Medium"
            if risk_score >= 35
            else "Low"
        )

        risk_color = (
            "yellow"
            if risk_level == "Medium"
            else "orange"
            if risk_level == "High"
            else "green"
        )

        kpi_card(
            "Risk Score",
            f"{risk_level}",
            f"Score {risk_score}/100",
            risk_color,
            "warn" if risk_level != "Low" else "good",
        )

    st.divider()

    # --------------------------------------------------------
    # TOP CHARTS
    # --------------------------------------------------------

    left, right = st.columns([1.35, 1])

    with left:

        st.markdown(
            '<div class="panel-title">Revenue vs Budget Trend</div>',
            unsafe_allow_html=True,
        )

        monthly = (
            filtered_df
            .dropna(subset=["Date"])
            .assign(
                Month=lambda x:
                x["Date"].dt.to_period("M").astype(str)
            )
            .groupby("Month")
            .agg(
                Revenue=("Revenue", "sum"),
                Budget=("Budget", "sum"),
                Actual=("Actual", "sum"),
            )
            .reset_index()
        )

        if not monthly.empty:

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
                height=360,
                title="",
            )

            st.plotly_chart(
                dark_chart(fig),
                use_container_width=True,
            )

    with right:

        st.markdown(
            '<div class="panel-title">Budget Variance by Business Unit</div>',
            unsafe_allow_html=True,
        )

        bu_summary = (
            filtered_df
            .groupby("Business Unit")
            .agg(
                Budget=("Budget", "sum"),
                Actual=("Actual", "sum"),
                Variance=("Variance", "sum"),
            )
            .reset_index()
        )

        bu_summary["Variance %"] = np.where(
            bu_summary["Budget"] != 0,
            bu_summary["Variance"] /
            bu_summary["Budget"] * 100,
            0,
        )

        if not bu_summary.empty:

            fig = px.bar(
                bu_summary,
                x="Business Unit",
                y="Variance",
                text="Variance",
            )

            fig.update_traces(
                texttemplate="%{text:.2s}",
                textposition="outside",
            )

            fig.update_layout(
                height=360,
                title="",
            )

            st.plotly_chart(
                dark_chart(fig),
                use_container_width=True,
            )

    # --------------------------------------------------------
    # P&L / CASH / CFO
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(
        [1.15, 1.0, 0.95]
    )

    with col1:

        st.markdown(
            '<div class="panel-title">P&L Summary</div>',
            unsafe_allow_html=True,
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
                    total_revenue,
                    total_actual,
                    total_profit,
                    ebitda,
                ],
            }
        )

        pnl["Amount"] = pnl["Amount"].apply(
            money
        )

        st.dataframe(
            pnl,
            hide_index=True,
            use_container_width=True,
        )

    with col2:

        st.markdown(
            '<div class="panel-title">Cash Flow Overview</div>',
            unsafe_allow_html=True,
        )

        monthly_cash = (
            filtered_df
            .dropna(subset=["Date"])
            .assign(
                Month=lambda x:
                x["Date"].dt.to_period("M").astype(str)
            )
            .groupby("Month")
            .agg(
                Inflow=("Revenue", "sum"),
                Outflow=("Actual", "sum"),
            )
            .reset_index()
        )

        if not monthly_cash.empty:

            opening = total_revenue * 0.08

            monthly_cash["Net Cash"] = (
                monthly_cash["Inflow"] -
                monthly_cash["Outflow"]
            )

            monthly_cash["Closing"] = (
                opening +
                monthly_cash["Net Cash"].cumsum()
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
                height=300,
                title="",
            )

            st.plotly_chart(
                dark_chart(fig),
                use_container_width=True,
            )

            st.caption(
                f"Modeled closing cash: {money(monthly_cash['Closing'].iloc[-1])}"
            )

    with col3:

        st.markdown(
            '<div class="panel-title">AI CFO Assistant</div>',
            unsafe_allow_html=True,
        )

        top_cost = (
            filtered_df
            .groupby("Account / Cost Category")["Variance"]
            .sum()
            .sort_values(ascending=False)
        )

        if not top_cost.empty:

            driver = top_cost.index[0]
            driver_variance = top_cost.iloc[0]

        else:

            driver = "No material driver"
            driver_variance = 0

        st.markdown(
            f"""
            <div class="ai-box">

                <div class="ai-title">
                    🤖 CFO Assistant
                </div>

                <br>

                <div class="ai-label">
                    Management Attention
                </div>

                <div class="ai-text">
                    {driver} is currently the largest
                    unfavorable cost driver.
                </div>

                <br>

                <div class="ai-label">
                    Financial Impact
                </div>

                <div class="ai-text">
                    Variance: {money(driver_variance)}
                </div>

                <br>

                <div class="ai-label">
                    Recommended Action
                </div>

                <div class="ai-text">
                    Review spend, utilization, contracts,
                    headcount and discretionary expenses.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # SECOND ROW
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(
        [1, 1, 0.9]
    )

    with col1:

        st.markdown(
            '<div class="panel-title">Top Cost Variances</div>',
            unsafe_allow_html=True,
        )

        cost_summary = (
            filtered_df
            .groupby("Account / Cost Category")[
                "Variance"
            ]
            .sum()
            .reset_index()
            .sort_values(
                "Variance",
                ascending=False,
            )
            .head(8)
        )

        if not cost_summary.empty:

            fig = px.bar(
                cost_summary,
                x="Variance",
                y="Account / Cost Category",
                orientation="h",
            )

            fig.update_layout(
                height=330,
                title="",
            )

            st.plotly_chart(
                dark_chart(fig),
                use_container_width=True,
            )

    with col2:

        st.markdown(
            '<div class="panel-title">Cost Driver Mix</div>',
            unsafe_allow_html=True,
        )

        if not cost_summary.empty:

            fig = px.pie(
                cost_summary,
                names="Account / Cost Category",
                values="Variance",
                hole=0.55,
            )

            fig.update_layout(
                height=330,
                title="",
            )

            st.plotly_chart(
                dark_chart(fig),
                use_container_width=True,
            )

    with col3:

        st.markdown(
            '<div class="panel-title">Forecast Summary</div>',
            unsafe_allow_html=True,
        )

        recent_revenue = (
            filtered_df
            .groupby(
                filtered_df["Date"].dt.to_period("M")
            )["Revenue"]
            .sum()
            .sort_index()
        )

        if len(recent_revenue) >= 2:

            growth = (
                recent_revenue.iloc[-1] /
                recent_revenue.iloc[-2] -
                1
            )

        else:

            growth = 0.01

        forecast_revenue = (
            total_revenue *
            (1 + growth)
        )

        forecast_profit = (
            forecast_revenue *
            (
                profit_margin / 100
            )
        )

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-label">
                    Next Period Revenue
                </div>

                <div class="kpi-value blue">
                    {money(forecast_revenue)}
                </div>

                <div class="kpi-delta">
                    Growth assumption {pct(growth * 100)}
                </div>

            </div>

            <br>

            <div class="kpi-card">

                <div class="kpi-label">
                    Forecast Profit
                </div>

                <div class="kpi-value green">
                    {money(forecast_profit)}
                </div>

                <div class="kpi-delta">
                    Margin {pct(profit_margin)}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # ACTIONS + ALERTS
    # --------------------------------------------------------

    st.divider()

    col1, col2 = st.columns(
        [1.4, 0.8]
    )

    with col1:

        st.markdown(
            '<div class="panel-title">AI CFO Recommended Actions</div>',
            unsafe_allow_html=True,
        )

        actions = []

        if total_variance > 0:

            actions.append(
                (
                    "P1",
                    "Review unfavorable budget variance",
                    "Finance",
                    money(total_variance),
                )
            )

        if not cost_summary.empty:

            actions.append(
                (
                    "P1",
                    f"Investigate {cost_summary.iloc[0]['Account / Cost Category']} variance",
                    "Finance + Operations",
                    money(
                        cost_summary.iloc[0]["Variance"]
                    ),
                )
            )

        actions.append(
            (
                "P2",
                "Refresh rolling forecast",
                "FP&A",
                "Management cycle",
            )
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
            '<div class="panel-title">Recent Alerts</div>',
            unsafe_allow_html=True,
        )

        if total_variance > 0:

            st.markdown(
                f"""
                <div class="alert-box alert-critical">
                    <div class="alert-title">
                        🔴 High budget variance
                    </div>
                    <div class="alert-text">
                        Actual cost is above budget by {money(total_variance)}.
                    </div>
                </div>
                """,
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

    st.markdown("## AI CFO Assistant")

    question = st.text_input(
        "Ask a finance question",
        placeholder=(
            "Why are costs above budget? "
            "Which business unit needs attention?"
        ),
    )

    if question:

        cost_summary = (
            filtered_df
            .groupby("Account / Cost Category")[
                "Variance"
            ]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        bu_summary = (
            filtered_df
            .groupby("Business Unit")[
                "Variance"
            ]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        driver = (
            cost_summary.index[0]
            if not cost_summary.empty
            else "No material driver"
        )

        driver_value = (
            cost_summary.iloc[0]
            if not cost_summary.empty
            else 0
        )

        bu = (
            bu_summary.index[0]
            if not bu_summary.empty
            else "No business unit"
        )

        bu_value = (
            bu_summary.iloc[0]
            if not bu_summary.empty
            else 0
        )

        st.markdown(
            f"""
            <div class="ai-box">

                <div class="ai-title">
                    🤖 FinSight AI CFO Analysis
                </div>

                <br>

                <div class="ai-label">
                    Finding
                </div>

                <div class="ai-text">
                    {driver} is the largest unfavorable
                    cost driver in the selected population.
                </div>

                <br>

                <div class="ai-label">
                    Evidence
                </div>

                <div class="ai-text">
                    Cost variance: {money(driver_value)}.
                    Highest business-unit variance:
                    {bu} at {money(bu_value)}.
                </div>

                <br>

                <div class="ai-label">
                    Business Impact
                </div>

                <div class="ai-text">
                    Continued overspend may reduce operating
                    margin and create forecast pressure.
                </div>

                <br>

                <div class="ai-label">
                    Management Action
                </div>

                <div class="ai-text">
                    Review controllable versus committed costs,
                    vendor spend, utilization and forecast assumptions.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.info(
            "Ask the AI CFO a financial question to generate an evidence-based analysis."
        )


# ============================================================
# FINANCIAL PERFORMANCE
# ============================================================

elif page == "📊 Financial Performance":

    render_hero(
        "Financial Performance",
        "Business-unit, department and cost-category performance intelligence.",
    )

    st.markdown("## Financial Performance")

    tab1, tab2, tab3 = st.tabs(
        [
            "Business Units",
            "Departments",
            "Cost Categories",
        ]
    )

    with tab1:

        summary = (
            filtered_df
            .groupby("Business Unit")
            .agg(
                Revenue=("Revenue", "sum"),
                Budget=("Budget", "sum"),
                Actual=("Actual", "sum"),
                Variance=("Variance", "sum"),
                Profit=("Profit", "sum"),
            )
            .reset_index()
        )

        summary["Margin %"] = np.where(
            summary["Revenue"] != 0,
            summary["Profit"] /
            summary["Revenue"] * 100,
            0,
        )

        st.dataframe(
            summary.style.format(
                {
                    "Revenue": money_full,
                    "Budget": money_full,
                    "Actual": money_full,
                    "Variance": money_full,
                    "Profit": money_full,
                    "Margin %": "{:.2f}%",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    with tab2:

        summary = (
            filtered_df
            .groupby("Department")
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
            summary.style.format(
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

        summary = (
            filtered_df
            .groupby("Account / Cost Category")
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
            summary.style.format(
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
# CASH FLOW CENTER
# ============================================================

elif page == "💵 Cash Flow Center":

    render_hero(
        "Cash Flow Center",
        "Modeled liquidity, inflows, outflows and cash runway intelligence.",
    )

    st.warning(
        "Cash-flow values are modeled from the available ERP-style financial dataset because the current demo dataset does not contain transaction-level bank balances, receivables and payables."
    )

    opening_cash = total_revenue * 0.08

    cash_monthly = (
        filtered_df
        .dropna(subset=["Date"])
        .assign(
            Month=lambda x:
            x["Date"].dt.to_period("M").astype(str)
        )
        .groupby("Month")
        .agg(
            Inflow=("Revenue", "sum"),
            Outflow=("Actual", "sum"),
        )
        .reset_index()
    )

    cash_monthly["Net Cash"] = (
        cash_monthly["Inflow"] -
        cash_monthly["Outflow"]
    )

    cash_monthly["Closing Cash"] = (
        opening_cash +
        cash_monthly["Net Cash"].cumsum()
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        kpi_card(
            "Opening Cash",
            money(opening_cash),
            "Modeled",
            "blue",
        )

    with c2:
        kpi_card(
            "Latest Closing Cash",
            money(
                cash_monthly["Closing Cash"].iloc[-1]
                if not cash_monthly.empty
                else opening_cash
            ),
            "Modeled",
            "green",
        )

    with c3:

        net_cash = cash_monthly["Net Cash"].sum()

        kpi_card(
            "Net Cash Movement",
            money(net_cash),
            "Inflow less outflow",
            "cyan",
        )

    st.markdown("### Cash Flow Trend")

    if not cash_monthly.empty:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=cash_monthly["Month"],
                y=cash_monthly["Inflow"],
                mode="lines+markers",
                name="Inflow",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=cash_monthly["Month"],
                y=cash_monthly["Outflow"],
                mode="lines+markers",
                name="Outflow",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=cash_monthly["Month"],
                y=cash_monthly["Closing Cash"],
                mode="lines+markers",
                name="Closing Cash",
            )
        )

        fig.update_layout(
            height=420
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

    st.markdown("## Budget vs Actuals")

    variance_df = (
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

    variance_df["Variance %"] = np.where(
        variance_df["Budget"] != 0,
        variance_df["Variance"] /
        variance_df["Budget"] * 100,
        0,
    )

    st.dataframe(
        variance_df.style.format(
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
# FORECASTING & PLANNING
# ============================================================

elif page == "📈 Forecasting & Planning":

    render_hero(
        "Forecasting & Planning",
        "Forward-looking revenue planning and financial scenario intelligence.",
    )

    st.markdown("## Revenue Forecast")

    monthly = (
        filtered_df
        .dropna(subset=["Date"])
        .assign(
            Month=lambda x:
            x["Date"].dt.to_period("M").astype(str)
        )
        .groupby("Month")["Revenue"]
        .sum()
        .reset_index()
    )

    if len(monthly) >= 3:

        monthly["Index"] = np.arange(
            len(monthly)
        )

        X = monthly[["Index"]]
        y = monthly["Revenue"]

        model = LinearRegression()

        model.fit(X, y)

        future_index = np.arange(
            len(monthly),
            len(monthly) + 6,
        )

        future = pd.DataFrame(
            {
                "Index": future_index
            }
        )

        base = model.predict(future)

        last_actual = (
            monthly["Revenue"].iloc[-1]
        )

        # ----------------------------------------------------
        # Growth adjustment
        # ----------------------------------------------------

        if len(monthly) >= 2:

            historical_growth = (
                monthly["Revenue"].iloc[-1] /
                monthly["Revenue"].iloc[-2] -
                1
            )

        else:

            historical_growth = 0.01

        base = np.maximum(
            base,
            last_actual *
            (1 + historical_growth)
            ** np.arange(
                1,
                7,
            ),
        )

        optimistic = (
            base * 1.08
        )

        downside = (
            base * 0.92
        )

        future_dates = pd.date_range(
            pd.to_datetime(
                monthly["Month"].iloc[-1]
            ) + pd.offsets.MonthBegin(1),
            periods=6,
            freq="MS",
        )

        forecast_df = pd.DataFrame(
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
                x=forecast_df["Month"],
                y=forecast_df["Base"],
                mode="lines+markers",
                name="Base Case",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_df["Month"],
                y=forecast_df["Optimistic"],
                mode="lines",
                name="Optimistic",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_df["Month"],
                y=forecast_df["Downside"],
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

            kpi_card(
                "Base Case",
                money(forecast_df["Base"].sum()),
                "6-month revenue",
                "blue",
            )

        with c2:

            kpi_card(
                "Optimistic",
                money(forecast_df["Optimistic"].sum()),
                "+8% scenario",
                "green",
            )

        with c3:

            kpi_card(
                "Downside",
                money(forecast_df["Downside"].sum()),
                "-8% scenario",
                "orange",
            )

    else:

        st.warning(
            "At least three monthly observations are needed for forecasting."
        )


# ============================================================
# WHAT-IF SCENARIOS
# ============================================================

elif page == "🔮 What-if Scenarios":

    render_hero(
        "What-if Scenarios",
        "Management planning simulator for revenue, cost and margin decisions.",
    )

    st.markdown("## Scenario Planner")

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
        (1 + revenue_change / 100)
    )

    scenario_cost = (
        total_actual *
        (1 + cost_change / 100)
    )

    scenario_profit = (
        scenario_revenue -
        scenario_cost
    )

    scenario_margin = (
        scenario_profit /
        scenario_revenue * 100
        if scenario_revenue != 0
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        kpi_card(
            "Scenario Revenue",
            money(scenario_revenue),
            f"{revenue_change:+d}% change",
            "blue",
        )

    with c2:

        kpi_card(
            "Scenario Cost",
            money(scenario_cost),
            f"{cost_change:+d}% change",
            "orange",
        )

    with c3:

        kpi_card(
            "Scenario Profit",
            money(scenario_profit),
            "Revenue minus cost",
            "green",
        )

    with c4:

        kpi_card(
            "Scenario Margin",
            pct(scenario_margin),
            "Projected margin",
            "purple",
        )

    st.markdown("### Scenario Comparison")

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
# RISK & ANOMALY DETECTION
# ============================================================

elif page == "⚠️ Risk & Anomaly Detection":

    render_hero(
        "Risk & Anomaly Detection",
        "Statistical anomaly detection and financial risk monitoring.",
    )

    risk_df = filtered_df.copy()

    if len(risk_df) >= 10:

        values = risk_df[
            ["Budget", "Actual", "Revenue"]
        ].fillna(0)

        model = IsolationForest(
            contamination=0.03,
            random_state=42,
        )

        predictions = model.fit_predict(
            values
        )

        risk_df["Model Anomaly"] = np.where(
            predictions == -1,
            "Potential Anomaly",
            "Normal",
        )

    else:

        risk_df["Model Anomaly"] = "Insufficient Data"

    anomaly_df = risk_df[
        risk_df["Model Anomaly"] ==
        "Potential Anomaly"
    ]

    c1, c2, c3 = st.columns(3)

    with c1:

        kpi_card(
            "Transactions",
            f"{len(risk_df):,}",
            "Selected population",
            "blue",
        )

    with c2:

        kpi_card(
            "Potential Anomalies",
            f"{len(anomaly_df):,}",
            "Isolation Forest",
            "orange",
        )

    with c3:

        anomaly_rate = (
            len(anomaly_df) /
            len(risk_df) * 100
            if len(risk_df) > 0
            else 0
        )

        kpi_card(
            "Anomaly Rate",
            pct(anomaly_rate),
            "Statistical screening",
            "yellow",
        )

    st.markdown("### Flagged Transactions")

    display_cols = [
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

    display_cols = [
        col
        for col in display_cols
        if col in anomaly_df.columns
    ]

    st.dataframe(
        anomaly_df[
            display_cols
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

    cost = (
        filtered_df
        .groupby("Account / Cost Category")
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

    cost["Variance %"] = np.where(
        cost["Budget"] != 0,
        cost["Variance"] /
        cost["Budget"] * 100,
        0,
    )

    st.dataframe(
        cost.style.format(
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
        cost.head(10),
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

    st.markdown("## AI CFO Action Center")

    top_costs = (
        filtered_df
        .groupby("Account / Cost Category")[
            "Variance"
        ]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    if not top_costs.empty:

        for i, (category, variance) in enumerate(
            top_costs.head(8).items()
        ):

            priority = (
                "P1"
                if variance > 0
                else "P2"
            )

            priority_class = (
                "priority-p1"
                if priority == "P1"
                else "priority-p2"
            )

            st.markdown(
                f"""
                <div class="action-card">

                    <span class="{priority_class}">
                        {priority}
                    </span>

                    &nbsp;&nbsp;

                    <strong style="color:#e5edf7;">
                        Review {category} variance
                    </strong>

                    <br>

                    <span style="color:#8191a8;font-size:0.75rem;">
                        Owner: Finance / FP&A
                        &nbsp; | &nbsp;
                        Impact: {money(variance)}
                    </span>

                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# REPORTS LIBRARY
# ============================================================

elif page == "📑 Reports Library":

    render_hero(
        "Reports Library",
        "Management-ready financial reports and analysis views.",
    )

    st.markdown("## Available Reports")

    reports = pd.DataFrame(
        {
            "Report": [
                "Executive Financial Summary",
                "Budget vs Actuals",
                "Business Unit Performance",
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
            ],
        }
    )

    st.dataframe(
        reports,
        hide_index=True,
        use_container_width=True,
    )

    csv_data = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Financial Data",
        csv_data,
        "finsight_financial_data.csv",
        "text/csv",
    )


# ============================================================
# UPLOAD DATA
# ============================================================

elif page == "📤 Upload Data":

    render_hero(
        "Upload Data",
        "Connect ERP-style CSV financial data to the FinSight intelligence layer.",
    )

    st.markdown("## ERP Data Upload")

    st.info(
        "Use the sidebar uploader to load a CSV. FinSight automatically recognizes common finance and ERP column names."
    )

    st.markdown(
        """
        ### Minimum recommended fields

        - Date
        - Business Unit
        - Department
        - Account / Cost Category
        - Budget
        - Actual
        - Revenue
        """
    )

    st.dataframe(
        df.head(20),
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

    st.markdown("## ERP Connectivity")

    systems = pd.DataFrame(
        {
            "System": [
                "ERP / Financial CSV",
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

    st.info(
        "The current portfolio prototype uses CSV-based ERP-style data. Live ERP APIs can be connected in a production implementation."
    )


# ============================================================
# DATA MAPPING
# ============================================================

elif page == "🧩 Data Mapping":

    render_hero(
        "Data Mapping",
        "Normalize ERP data structures into a consistent FP&A analytical model.",
    )

    st.markdown("## Detected Data Model")

    mapping = pd.DataFrame(
        {
            "FinSight Field": required_columns,
            "Detected": [
                "Yes" if c in df.columns else "No"
                for c in required_columns
            ],
        }
    )

    st.dataframe(
        mapping,
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

        quality.append(
            {
                "Field": col,
                "Missing Records": missing_count,
                "Completeness %": (
                    100 -
                    missing_count /
                    len(df) * 100
                    if len(df) > 0
                    else 0
                ),
            }
        )

    quality_df = pd.DataFrame(
        quality
    )

    st.dataframe(
        quality_df.style.format(
            {
                "Completeness %": "{:.2f}%"
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

    st.markdown("## Current Alerts")

    if total_variance > 0:

        st.markdown(
            f"""
            <div class="alert-box alert-critical">

                <div class="alert-title">
                    🔴 Unfavorable budget variance detected
                </div>

                <div class="alert-text">
                    Aggregate actual cost exceeds budget
                    by {money(total_variance)}.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="alert-box alert-good">

                <div class="alert-title">
                    🟢 No aggregate unfavorable variance
                </div>

                <div class="alert-text">
                    Current selected population is within budget.
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

    st.markdown("## Application Settings")

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
        "Settings shown here are portfolio-prototype controls. Production authentication, role-based access and enterprise configuration can be added later."
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

    csv_data = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Selected Data",
        csv_data,
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

        Portfolio prototype using synthetic ERP-style
        financial data. Modeled components are clearly
        identified where source data is unavailable.

    </div>
    """,
    unsafe_allow_html=True,
)
