import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from textwrap import dedent
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FinSight AI | CFO Command Center",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(circle at top right, rgba(30, 90, 130, 0.18), transparent 35%),
        linear-gradient(135deg, #06101c 0%, #091522 55%, #07111d 100%);
    color: #eaf2f8;
}

[data-testid="stSidebar"] {
    background: #081421;
    border-right: 1px solid #20364e;
}

[data-testid="stSidebar"] * {
    color: #dce8f2;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

h1, h2, h3, h4 {
    color: #f3f8fc !important;
}

.hero {
    padding: 25px 28px;
    border-radius: 18px;
    border: 1px solid #24435e;
    background:
        linear-gradient(
            110deg,
            rgba(19, 48, 76, 0.98),
            rgba(10, 25, 41, 0.98)
        );
    margin-bottom: 20px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.18);
}

.hero-title {
    font-size: 31px;
    font-weight: 800;
    letter-spacing: -0.7px;
}

.hero-subtitle {
    color: #94a9bd;
    margin-top: 7px;
    font-size: 14px;
}

.section-title {
    font-size: 19px;
    font-weight: 750;
    margin-top: 12px;
    margin-bottom: 9px;
}

.section-subtitle {
    color: #8fa3b6;
    font-size: 12px;
    margin-bottom: 12px;
}

.cfo-card {
    background:
        linear-gradient(145deg, #10243a, #0b1828);
    border: 1px solid #27425c;
    border-radius: 14px;
    padding: 17px;
    min-height: 112px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.12);
}

.cfo-label {
    color: #91a6b9;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}

.cfo-value {
    color: #f4f8fb;
    font-size: 23px;
    font-weight: 800;
    margin-top: 8px;
    white-space: nowrap;
}

.cfo-delta {
    color: #6bd3aa;
    font-size: 11px;
    margin-top: 7px;
}

.cfo-delta.bad {
    color: #ff8f8f;
}

.insight-card {
    background: #0d1f31;
    border-left: 4px solid #42b9ff;
    border-radius: 10px;
    padding: 14px 16px;
    margin: 7px 0;
}

.action-card {
    background: #101e2d;
    border: 1px solid #29435d;
    border-radius: 11px;
    padding: 13px 15px;
    margin: 7px 0;
}

.pill {
    display: inline-block;
    padding: 4px 9px;
    border-radius: 999px;
    background: #163650;
    color: #8ed7ff;
    font-size: 10px;
    font-weight: 750;
    margin-right: 5px;
}

.muted {
    color: #8fa3b6;
    font-size: 12px;
}

.footer {
    text-align: center;
    color: #70859a;
    font-size: 11px;
    padding-top: 25px;
}

div[data-testid="stMetric"] {
    background: #102238;
    border: 1px solid #263f58;
    border-radius: 12px;
    padding: 12px;
}

div[data-testid="stMetricLabel"] {
    color: #91a6b9;
}

div[data-testid="stMetricValue"] {
    color: #f4f8fb;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def html_block(text):
    """
    Safely render HTML without Streamlit treating indented
    multiline HTML as a code block.
    """
    st.markdown(
        dedent(text).strip(),
        unsafe_allow_html=True
    )


def money(value):
    if pd.isna(value):
        return "₹0"

    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)

    if value >= 10_000_000:
        return f"{sign}₹{value / 10_000_000:.2f} Cr"

    if value >= 100_000:
        return f"{sign}₹{value / 100_000:.2f} L"

    return f"{sign}₹{value:,.0f}"


def money_full(value):
    if pd.isna(value):
        return "₹0"

    return f"₹{float(value):,.0f}"


def percentage(value):
    if pd.isna(value):
        return "0.0%"

    return f"{float(value):.1f}%"


def kpi_card(label, value, subtitle="", negative=False):
    delta_class = "cfo-delta bad" if negative else "cfo-delta"

    html_block(
        f"""
        <div class="cfo-card">
            <div class="cfo-label">{label}</div>
            <div class="cfo-value">{value}</div>
            <div class="{delta_class}">{subtitle}</div>
        </div>
        """
    )


def section(title, subtitle=None):
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )

    if subtitle:
        st.markdown(
            f'<div class="section-subtitle">{subtitle}</div>',
            unsafe_allow_html=True
        )


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_default_data():

    possible_files = [
        Path("synthetic_erp_financials.csv"),
        Path("data/synthetic_erp_financials.csv"),
    ]

    for file in possible_files:
        if file.exists():
            return pd.read_csv(file)

    return pd.DataFrame()


# ============================================================
# DATA NORMALIZATION
# ============================================================

def normalize_data(raw):

    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()

    rename_map = {
        "transaction_id": "Transaction ID",
        "Transaction ID": "Transaction ID",

        "date": "Date",
        "Date": "Date",

        "business_unit": "Business Unit",
        "Business Unit": "Business Unit",

        "department": "Department",
        "Department": "Department",

        "account": "Account / Cost Category",
        "Account / Cost Category": "Account / Cost Category",

        "budget": "Budget",
        "Budget": "Budget",

        "actual": "Actual",
        "Actual": "Actual",

        "revenue": "Revenue",
        "Revenue": "Revenue",

        "variance": "Variance",
        "Variance": "Variance",

        "variance_pct": "Variance %",
        "Variance %": "Variance %",

        "profit": "Profit",
        "Profit": "Profit",

        "anomaly_flag": "Anomaly Flag",
        "Anomaly Flag": "Anomaly Flag",

        "anomaly": "Anomaly",
        "Anomaly": "Anomaly",
    }

    df.rename(
        columns={
            column: rename_map[column]
            for column in df.columns
            if column in rename_map
        },
        inplace=True,
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    if "Date" not in df.columns:
        df["Date"] = pd.Timestamp.today()

    if "Business Unit" not in df.columns:
        df["Business Unit"] = "Unknown"

    if "Account / Cost Category" not in df.columns:
        df["Account / Cost Category"] = "Other"

    for column in ["Budget", "Actual", "Revenue"]:
        if column not in df.columns:
            df[column] = 0.0

    # --------------------------------------------------------
    # Types
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    numeric_columns = [
        "Budget",
        "Actual",
        "Revenue",
        "Variance",
        "Variance %",
        "Profit",
        "Anomaly Flag",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            ).fillna(0)

    # --------------------------------------------------------
    # Derived financial metrics
    # --------------------------------------------------------

    if "Variance" not in df.columns:
        df["Variance"] = (
            df["Actual"] -
            df["Budget"]
        )

    if "Variance %" not in df.columns:

        df["Variance %"] = np.where(
            df["Budget"].abs() > 0,
            df["Variance"] /
            df["Budget"] *
            100,
            0,
        )

    if "Profit" not in df.columns:

        df["Profit"] = (
            df["Revenue"] -
            df["Actual"]
        )

    if "Transaction ID" not in df.columns:

        df["Transaction ID"] = [
            f"TXN-{i + 1:05d}"
            for i in range(len(df))
        ]

    if "Anomaly" not in df.columns:
        df["Anomaly"] = ""

    # --------------------------------------------------------
    # Department intelligence
    # --------------------------------------------------------

    department_map = {

        "Cloud Infrastructure":
            "Engineering",

        "Software Licenses":
            "Engineering",

        "Office & Facilities":
            "Operations",

        "Marketing":
            "Marketing",

        "Payroll":
            "HR",

        "Professional Services":
            "Finance",

        "Training":
            "HR",

        "Travel":
            "Sales",

        "Cloud":
            "Engineering",

        "Software":
            "Engineering",

        "Facilities":
            "Operations",
    }

    account_series = (
        df["Account / Cost Category"]
        .astype(str)
        .str.strip()
    )

    derived_department = (
        account_series.map(department_map)
    )

    if "Department" not in df.columns:

        df["Department"] = (
            derived_department
            .fillna("Other")
        )

    else:

        existing_department = (
            df["Department"]
            .astype(str)
            .str.strip()
        )

        invalid_department = (
            existing_department.eq("")
            |
            existing_department.str.lower().isin(
                [
                    "nan",
                    "none",
                    "unmapped",
                    "unknown",
                ]
            )
        )

        df.loc[
            invalid_department,
            "Department"
        ] = (
            derived_department[
                invalid_department
            ].fillna("Other")
        )

    # --------------------------------------------------------
    # Clean dimensions
    # --------------------------------------------------------

    for column in [
        "Business Unit",
        "Department",
        "Account / Cost Category",
    ]:

        df[column] = (
            df[column]
            .astype(str)
            .replace(
                {
                    "nan": "Other",
                    "None": "Other",
                }
            )
        )

    df = (
        df
        .dropna(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# LOAD DATA
# ============================================================

uploaded_file = st.sidebar.file_uploader(
    "Upload ERP CSV",
    type=["csv"],
)

if uploaded_file is not None:

    raw_data = pd.read_csv(
        uploaded_file
    )

    data_source = "Uploaded CSV"

else:

    raw_data = load_default_data()

    data_source = "Repository CSV"


df = normalize_data(raw_data)


# ============================================================
# EMPTY DATA CHECK
# ============================================================

if df.empty:

    html_block(
        """
        <div class="hero">
            <div class="hero-title">
                FinSight AI
            </div>

            <div class="hero-subtitle">
                No ERP dataset detected. Upload a CSV from the sidebar
                or add synthetic_erp_financials.csv to the repository.
            </div>
        </div>
        """
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## 📊 FinSight AI"
)

st.sidebar.caption(
    "Agentic FP&A & ERP Intelligence"
)

st.sidebar.markdown("---")


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


page = st.sidebar.radio(
    "Command Center",
    modules,
    index=0,
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.markdown("---")

business_units = sorted(
    df["Business Unit"]
    .dropna()
    .unique()
    .tolist()
)

departments = sorted(
    df["Department"]
    .dropna()
    .unique()
    .tolist()
)


selected_business_unit = st.sidebar.selectbox(
    "Business Unit",
    ["All"] + business_units,
)

selected_department = st.sidebar.selectbox(
    "Department",
    ["All"] + departments,
)


# ============================================================
# FILTERED VIEW
# ============================================================

view = df.copy()


if selected_business_unit != "All":

    view = view[
        view["Business Unit"]
        == selected_business_unit
    ]


if selected_department != "All":

    view = view[
        view["Department"]
        == selected_department
    ]


st.sidebar.markdown("---")

st.sidebar.caption(
    f"Records: {len(view):,}"
)

st.sidebar.caption(
    f"Data through: {view['Date'].max():%d %b %Y}"
)

st.sidebar.caption(
    f"Source: {data_source}"
)


# ============================================================
# GLOBAL FINANCIAL METRICS
# ============================================================

revenue = view["Revenue"].sum()

actual_cost = view["Actual"].sum()

budget = view["Budget"].sum()

variance = view["Variance"].sum()

profit = view["Profit"].sum()

margin = (
    profit /
    revenue *
    100
    if revenue != 0
    else 0
)


# ============================================================
# MONTHLY DATA
# ============================================================

monthly = (
    view
    .assign(
        Month=
        view["Date"]
        .dt
        .to_period("M")
        .dt
        .to_timestamp()
    )
    .groupby(
        "Month",
        as_index=False
    )
    .agg(
        Revenue=("Revenue", "sum"),
        Budget=("Budget", "sum"),
        Actual=("Actual", "sum"),
        Profit=("Profit", "sum"),
    )
)

monthly["Variance"] = (
    monthly["Actual"] -
    monthly["Budget"]
)

monthly["Margin"] = np.where(
    monthly["Revenue"] != 0,
    monthly["Profit"] /
    monthly["Revenue"] *
    100,
    0,
)


# ============================================================
# BUSINESS UNIT DATA
# ============================================================

business_unit_summary = (
    view
    .groupby(
        "Business Unit",
        as_index=False
    )
    .agg(
        Revenue=("Revenue", "sum"),
        Budget=("Budget", "sum"),
        Actual=("Actual", "sum"),
        Profit=("Profit", "sum"),
        Variance=("Variance", "sum"),
    )
)

business_unit_summary["Margin %"] = np.where(
    business_unit_summary["Revenue"] != 0,
    business_unit_summary["Profit"] /
    business_unit_summary["Revenue"] *
    100,
    0,
)

business_unit_summary = (
    business_unit_summary
    .sort_values(
        "Variance",
        ascending=False
    )
)


# ============================================================
# DEPARTMENT DATA
# ============================================================

department_summary = (
    view
    .groupby(
        "Department",
        as_index=False
    )
    .agg(
        Budget=("Budget", "sum"),
        Actual=("Actual", "sum"),
        Variance=("Variance", "sum"),
        Revenue=("Revenue", "sum"),
    )
    .sort_values(
        "Variance",
        ascending=False
    )
)


# ============================================================
# COST DATA
# ============================================================

cost_summary = (
    view
    .groupby(
        "Account / Cost Category",
        as_index=False
    )
    .agg(
        Budget=("Budget", "sum"),
        Actual=("Actual", "sum"),
        Variance=("Variance", "sum"),
    )
    .sort_values(
        "Variance",
        ascending=False
    )
)


# ============================================================
# GLOBAL HEADER
# ============================================================

html_block(
    """
    <div class="hero">

        <div class="hero-title">
            FinSight AI — CFO Command Center
        </div>

        <div class="hero-subtitle">
            Agentic FP&A • ERP Analytics • Forecasting • Risk
            • Cash & Liquidity • Management Actions
        </div>

    </div>
    """
)


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

if page == "Executive Dashboard":

    section(
        "Executive Dashboard",
        "Leadership view of profitability, cost performance, liquidity and risk."
    )

    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        kpi_card(
            "Revenue",
            money(revenue),
            "Current filtered view",
        )

    with k2:
        kpi_card(
            "Actual Cost",
            money(actual_cost),
            "Actual spend",
        )

    with k3:
        kpi_card(
            "Budget",
            money(budget),
            "Approved / modeled plan",
        )

    with k4:
        kpi_card(
            "Variance",
            money(variance),
            "Actual minus budget",
            negative=variance > 0,
        )

    with k5:
        kpi_card(
            "Profit Margin",
            percentage(margin),
            "Revenue less actual cost",
            negative=margin < 50,
        )

    st.markdown("")

    left, right = st.columns(
        [1.65, 1]
    )

    with left:

        section(
            "Revenue vs Budget vs Actual Cost"
        )

        if not monthly.empty:

            figure = go.Figure()

            figure.add_trace(
                go.Scatter(
                    x=monthly["Month"],
                    y=monthly["Revenue"],
                    mode="lines+markers",
                    name="Revenue",
                )
            )

            figure.add_trace(
                go.Scatter(
                    x=monthly["Month"],
                    y=monthly["Budget"],
                    mode="lines",
                    name="Budget",
                )
            )

            figure.add_trace(
                go.Scatter(
                    x=monthly["Month"],
                    y=monthly["Actual"],
                    mode="lines+markers",
                    name="Actual Cost",
                )
            )

            figure.update_layout(
                template="plotly_dark",
                height=350,
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10,
                ),
                hovermode="x unified",
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )

    with right:

        section(
            "P&L Summary"
        )

        pnl = pd.DataFrame(
            {
                "Metric": [
                    "Revenue",
                    "Actual Cost",
                    "Profit",
                ],
                "Value": [
                    revenue,
                    actual_cost,
                    profit,
                ],
            }
        )

        st.dataframe(
            pnl.style.format(
                {
                    "Value":
                    "₹{:,.0f}"
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        html_block(
            f"""
            <div class="insight-card">
                <b>Current Margin</b><br>
                {percentage(margin)}
                <br><br>
                <span class="muted">
                    Profit is calculated as Revenue minus Actual Cost.
                </span>
            </div>
            """
        )

    st.markdown("")

    c1, c2, c3 = st.columns(3)

    with c1:

        section(
            "Business Unit Variance"
        )

        st.dataframe(
            business_unit_summary[
                [
                    "Business Unit",
                    "Budget",
                    "Actual",
                    "Variance",
                ]
            ]
            .head(8)
            .style.format(
                {
                    "Budget": "₹{:,.0f}",
                    "Actual": "₹{:,.0f}",
                    "Variance": "₹{:,.0f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with c2:

        section(
            "Top Cost Variances"
        )

        st.dataframe(
            cost_summary
            .head(8)
            .style.format(
                {
                    "Budget": "₹{:,.0f}",
                    "Actual": "₹{:,.0f}",
                    "Variance": "₹{:,.0f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with c3:

        section(
            "AI CFO Snapshot"
        )

        if not cost_summary.empty:

            top_cost = cost_summary.iloc[0]

            html_block(
                f"""
                <div class="action-card">

                    <span class="pill">
                        COST DRIVER
                    </span>

                    <br><br>

                    <b>
                        {top_cost["Account / Cost Category"]}
                    </b>

                    <br>

                    <span class="muted">
                        Variance:
                        {money(top_cost["Variance"])}
                    </span>

                </div>
                """
            )

        if not business_unit_summary.empty:

            top_bu = (
                business_unit_summary
                .iloc[0]
            )

            html_block(
                f"""
                <div class="action-card">

                    <span class="pill">
                        BU HOTSPOT
                    </span>

                    <br><br>

                    <b>
                        {top_bu["Business Unit"]}
                    </b>

                    <br>

                    <span class="muted">
                        Variance:
                        {money(top_bu["Variance"])}
                    </span>

                </div>
                """
            )


# ============================================================
# AI CFO
# ============================================================

elif page == "AI CFO":

    section(
        "AI CFO Assistant",
        "Evidence-based management reasoning over the selected ERP dataset."
    )

    question = st.text_input(
        "Ask the AI CFO",
        placeholder=(
            "Why are costs over budget? "
            "Which business unit needs attention?"
        ),
    )

    if question:

        question_lower = (
            question.lower()
        )

        top_cost = (
            cost_summary.iloc[0]
            if not cost_summary.empty
            else None
        )

        top_bu = (
            business_unit_summary.iloc[0]
            if not business_unit_summary.empty
            else None
        )

        if any(
            word in question_lower
            for word in [
                "cost",
                "expense",
                "spend",
                "over budget",
            ]
        ):

            finding = (
                f"{top_cost['Account / Cost Category']} "
                "is the largest unfavorable cost driver."
            )

            evidence = (
                f"Budget {money(top_cost['Budget'])} "
                f"vs actual {money(top_cost['Actual'])}; "
                f"variance {money(top_cost['Variance'])}."
            )

            action = (
                "Review underlying transactions, challenge "
                "discretionary spend and reforecast the affected "
                "cost line."
            )

            priority = "P1"
            owner = "Finance Controller"

        elif any(
            word in question_lower
            for word in [
                "business unit",
                "bu",
                "region",
                "entity",
            ]
        ):

            finding = (
                f"{top_bu['Business Unit']} "
                "has the highest unfavorable variance."
            )

            evidence = (
                f"Budget {money(top_bu['Budget'])} "
                f"vs actual {money(top_bu['Actual'])}; "
                f"variance {money(top_bu['Variance'])}."
            )

            action = (
                "Run a BU-level variance bridge and assign "
                "corrective actions to the responsible "
                "business owner."
            )

            priority = "P1"
            owner = "BU Finance Lead"

        elif any(
            word in question_lower
            for word in [
                "margin",
                "profit",
                "profitability",
            ]
        ):

            finding = (
                f"Current filtered operating margin "
                f"is {percentage(margin)}."
            )

            evidence = (
                f"Revenue {money(revenue)} less actual cost "
                f"{money(actual_cost)} produces profit "
                f"of {money(profit)}."
            )

            action = (
                "Protect margin by controlling the largest "
                "unfavorable cost categories while validating "
                "revenue assumptions."
            )

            priority = "P1"
            owner = "CFO / FP&A"

        else:

            finding = (
                f"Current margin is {percentage(margin)} "
                f"with total variance of {money(variance)}."
            )

            evidence = (
                f"Revenue {money(revenue)}, actual cost "
                f"{money(actual_cost)}, budget "
                f"{money(budget)}."
            )

            action = (
                "Review the top cost driver, largest BU variance "
                "and flagged transactions before the next "
                "forecast cycle."
            )

            priority = "P2"
            owner = "FP&A"

        html_block(
            f"""
            <div class="action-card">

                <span class="pill">
                    FINDING
                </span>

                <br><br>

                {finding}

            </div>

            <div class="action-card">

                <span class="pill">
                    EVIDENCE
                </span>

                <br><br>

                {evidence}

            </div>

            <div class="action-card">

                <span class="pill">
                    BUSINESS IMPACT
                </span>

                <br><br>

                Total variance exposure is
                {money(variance)}.

            </div>

            <div class="action-card">

                <span class="pill">
                    MANAGEMENT ACTION
                </span>

                <br><br>

                {action}

            </div>
            """
        )

        c1, c2 = st.columns(2)

        with c1:
            kpi_card(
                "Priority",
                priority,
                "AI CFO recommendation",
            )

        with c2:
            kpi_card(
                "Owner",
                owner,
                "Suggested accountability",
            )


# ============================================================
# FINANCIAL PERFORMANCE
# ============================================================

elif page == "Financial Performance":

    section(
        "Financial Performance",
        "Performance intelligence across business units, departments and cost categories."
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Business Units",
            "Departments",
            "Cost Categories",
        ]
    )

    with tab1:

        st.dataframe(
            business_unit_summary.style.format(
                {
                    "Revenue": "₹{:,.0f}",
                    "Budget": "₹{:,.0f}",
                    "Actual": "₹{:,.0f}",
                    "Profit": "₹{:,.0f}",
                    "Variance": "₹{:,.0f}",
                    "Margin %": "{:.1f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        if not business_unit_summary.empty:

            figure = px.bar(
                business_unit_summary,
                x="Business Unit",
                y="Variance",
                title="Business Unit Variance",
                template="plotly_dark",
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )

    with tab2:

        st.dataframe(
            department_summary.style.format(
                {
                    "Budget": "₹{:,.0f}",
                    "Actual": "₹{:,.0f}",
                    "Variance": "₹{:,.0f}",
                    "Revenue": "₹{:,.0f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with tab3:

        st.dataframe(
            cost_summary.style.format(
                {
                    "Budget": "₹{:,.0f}",
                    "Actual": "₹{:,.0f}",
                    "Variance": "₹{:,.0f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# CASH FLOW CENTER
# ============================================================

elif page == "Cash Flow Center":

    section(
        "Cash Flow Center",
        "Modeled liquidity command center based on ERP revenue and cost activity."
    )

    st.warning(
        "Portfolio prototype: cash inflows, outflows and opening "
        "cash are modeled because the base ERP dataset does not "
        "contain live bank, AR or AP balances."
    )

    cashflow = monthly.copy()

    cashflow["Cash Inflow"] = (
        cashflow["Revenue"] *
        0.82
    )

    cashflow["Cash Outflow"] = (
        cashflow["Actual"] *
        0.72
    )

    cashflow["Net Cash Flow"] = (
        cashflow["Cash Inflow"] -
        cashflow["Cash Outflow"]
    )

    opening_cash = 10_000_000.0

    cashflow["Opening Cash"] = (
        opening_cash +
        cashflow["Net Cash Flow"]
        .cumsum()
        .shift(
            1,
            fill_value=0
        )
    )

    cashflow["Closing Cash"] = (
        cashflow["Opening Cash"] +
        cashflow["Net Cash Flow"]
    )

    closing_cash = (
        cashflow["Closing Cash"].iloc[-1]
        if not cashflow.empty
        else opening_cash
    )

    net_cash_flow = (
        cashflow["Net Cash Flow"].sum()
        if not cashflow.empty
        else 0
    )

    average_outflow = (
        cashflow["Cash Outflow"].mean()
        if not cashflow.empty
        else 0
    )

    runway = (
        closing_cash /
        average_outflow
        if average_outflow > 0
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card(
            "Opening Cash",
            money(opening_cash),
            "Modeled opening balance",
        )

    with c2:
        kpi_card(
            "Closing Cash",
            money(closing_cash),
            "Latest modeled balance",
        )

    with c3:
        kpi_card(
            "Net Cash Flow",
            money(net_cash_flow),
            "Cumulative modeled movement",
            negative=net_cash_flow < 0,
        )

    with c4:
        kpi_card(
            "Runway",
            f"{runway:.1f} mo",
            "Illustrative liquidity runway",
            negative=runway < 3,
        )

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=cashflow["Month"],
            y=cashflow["Cash Inflow"],
            name="Cash Inflow",
        )
    )

    figure.add_trace(
        go.Bar(
            x=cashflow["Month"],
            y=-cashflow["Cash Outflow"],
            name="Cash Outflow",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=cashflow["Month"],
            y=cashflow["Closing Cash"],
            mode="lines+markers",
            name="Closing Cash",
        )
    )

    figure.update_layout(
        template="plotly_dark",
        barmode="relative",
        height=380,
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10,
        ),
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    st.dataframe(
        cashflow.style.format(
            {
                "Revenue": "₹{:,.0f}",
                "Actual": "₹{:,.0f}",
                "Cash Inflow": "₹{:,.0f}",
                "Cash Outflow": "₹{:,.0f}",
                "Net Cash Flow": "₹{:,.0f}",
                "Opening Cash": "₹{:,.0f}",
                "Closing Cash": "₹{:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# BUDGET VS ACTUALS
# ============================================================

elif page == "Budget vs Actuals":

    section(
        "Budget vs Actuals",
        "Transaction-level variance management and budget discipline."
    )

    transaction_view = (
        view[
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
            ]
        ]
        .sort_values(
            "Variance",
            ascending=False
        )
        .head(300)
    )

    st.dataframe(
        transaction_view.style.format(
            {
                "Budget": "₹{:,.0f}",
                "Actual": "₹{:,.0f}",
                "Variance": "₹{:,.0f}",
                "Variance %": "{:.1f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    figure = px.bar(
        cost_summary.head(12),
        x="Account / Cost Category",
        y="Variance",
        title="Largest Cost Variances",
        template="plotly_dark",
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )


# ============================================================
# FORECASTING & PLANNING
# ============================================================

elif page == "Forecasting & Planning":

    section(
        "Forecasting & Planning",
        "Six-month outlook with Base, Optimistic and Downside scenarios."
    )

    if len(monthly) >= 3:

        x = np.arange(
            len(monthly)
        ).reshape(-1, 1)

        revenue_model = LinearRegression()

        cost_model = LinearRegression()

        revenue_model.fit(
            x,
            monthly["Revenue"].values,
        )

        cost_model.fit(
            x,
            monthly["Actual"].values,
        )

        future_periods = 6

        future_x = np.arange(
            len(monthly),
            len(monthly) +
            future_periods,
        ).reshape(-1, 1)

        future_dates = pd.date_range(
            monthly["Month"].max()
            + pd.offsets.MonthBegin(1),
            periods=future_periods,
            freq="MS",
        )

        base_revenue = np.maximum(
            revenue_model.predict(
                future_x
            ),
            0,
        )

        base_cost = np.maximum(
            cost_model.predict(
                future_x
            ),
            0,
        )

        forecast = pd.DataFrame(
            {
                "Month": future_dates,

                "Base Revenue":
                    base_revenue,

                "Base Cost":
                    base_cost,
            }
        )

        # IMPORTANT:
        # Scenarios are deliberately different.

        forecast["Optimistic Revenue"] = (
            forecast["Base Revenue"] *
            1.08
        )

        forecast["Optimistic Cost"] = (
            forecast["Base Cost"] *
            0.95
        )

        forecast["Downside Revenue"] = (
            forecast["Base Revenue"] *
            0.92
        )

        forecast["Downside Cost"] = (
            forecast["Base Cost"] *
            1.10
        )

        base_total = (
            forecast["Base Revenue"].sum()
        )

        optimistic_total = (
            forecast["Optimistic Revenue"].sum()
        )

        downside_total = (
            forecast["Downside Revenue"].sum()
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            kpi_card(
                "Base Revenue",
                money(base_total),
                "6-month forecast",
            )

        with c2:

            kpi_card(
                "Optimistic Revenue",
                money(optimistic_total),
                "+8% revenue / -5% cost",
            )

        with c3:

            kpi_card(
                "Downside Revenue",
                money(downside_total),
                "-8% revenue / +10% cost",
                negative=True,
            )

        st.markdown("")

        figure = go.Figure()

        figure.add_trace(
            go.Scatter(
                x=forecast["Month"],
                y=forecast["Base Revenue"],
                mode="lines+markers",
                name="Base",
            )
        )

        figure.add_trace(
            go.Scatter(
                x=forecast["Month"],
                y=forecast["Optimistic Revenue"],
                mode="lines+markers",
                name="Optimistic",
            )
        )

        figure.add_trace(
            go.Scatter(
                x=forecast["Month"],
                y=forecast["Downside Revenue"],
                mode="lines+markers",
                name="Downside",
            )
        )

        figure.update_layout(
            template="plotly_dark",
            height=370,
            hovermode="x unified",
            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10,
            ),
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

        base_margin = (
            (
                forecast["Base Revenue"].sum()
                -
                forecast["Base Cost"].sum()
            )
            /
            forecast["Base Revenue"].sum()
            *
            100
            if forecast["Base Revenue"].sum()
            else 0
        )

        optimistic_margin = (
            (
                forecast["Optimistic Revenue"].sum()
                -
                forecast["Optimistic Cost"].sum()
            )
            /
            forecast["Optimistic Revenue"].sum()
            *
            100
            if forecast["Optimistic Revenue"].sum()
            else 0
        )

        downside_margin = (
            (
                forecast["Downside Revenue"].sum()
                -
                forecast["Downside Cost"].sum()
            )
            /
            forecast["Downside Revenue"].sum()
            *
            100
            if forecast["Downside Revenue"].sum()
            else 0
        )

        scenario_table = pd.DataFrame(
            {
                "Scenario": [
                    "Base",
                    "Optimistic",
                    "Downside",
                ],

                "Revenue": [
                    forecast["Base Revenue"].sum(),
                    forecast["Optimistic Revenue"].sum(),
                    forecast["Downside Revenue"].sum(),
                ],

                "Cost": [
                    forecast["Base Cost"].sum(),
                    forecast["Optimistic Cost"].sum(),
                    forecast["Downside Cost"].sum(),
                ],

                "Margin %": [
                    base_margin,
                    optimistic_margin,
                    downside_margin,
                ],
            }
        )

        st.dataframe(
            scenario_table.style.format(
                {
                    "Revenue": "₹{:,.0f}",
                    "Cost": "₹{:,.0f}",
                    "Margin %": "{:.1f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "At least three monthly periods are required for forecasting."
        )


# ============================================================
# WHAT-IF SCENARIOS
# ============================================================

elif page == "What-if Scenarios":

    section(
        "What-if Scenario Planner",
        "Stress-test revenue, cost and profitability assumptions."
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        revenue_change = st.slider(
            "Revenue Change %",
            min_value=-30,
            max_value=30,
            value=5,
        )

    with c2:

        cost_change = st.slider(
            "Cost Change %",
            min_value=-30,
            max_value=30,
            value=-3,
        )

    with c3:

        contingency = st.slider(
            "Contingency %",
            min_value=0,
            max_value=15,
            value=3,
        )

    scenario_revenue = (
        revenue *
        (
            1 +
            revenue_change / 100
        )
    )

    scenario_cost = (
        actual_cost *
        (
            1 +
            cost_change / 100
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

    a, b, c, d = st.columns(4)

    with a:

        kpi_card(
            "Scenario Revenue",
            money(scenario_revenue),
            f"{revenue_change:+d}% assumption",
        )

    with b:

        kpi_card(
            "Scenario Cost",
            money(scenario_cost),
            f"{cost_change:+d}% assumption",
            negative=cost_change > 0,
        )

    with c:

        kpi_card(
            "Scenario Profit",
            money(scenario_profit),
            "Revenue minus cost",
            negative=scenario_profit < 0,
        )

    with d:

        kpi_card(
            "Scenario Margin",
            percentage(scenario_margin),
            f"Contingency {contingency}%",
            negative=scenario_margin < margin,
        )

    scenario_chart = pd.DataFrame(
        {
            "Metric": [
                "Current Revenue",
                "Scenario Revenue",
                "Current Cost",
                "Scenario Cost",
            ],

            "Value": [
                revenue,
                scenario_revenue,
                actual_cost,
                scenario_cost,
            ],
        }
    )

    figure = px.bar(
        scenario_chart,
        x="Metric",
        y="Value",
        title="Scenario Impact",
        template="plotly_dark",
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )


# ============================================================
# RISK & ANOMALY DETECTION
# ============================================================

elif page == "Risk & Anomaly Detection":

    section(
        "Risk & Anomaly Detection",
        "Transaction-level anomaly detection and financial risk hotspots."
    )

    risk_view = view.copy()

    if (
        "Anomaly Flag" in risk_view.columns
        and
        risk_view["Anomaly Flag"].sum() > 0
    ):

        risk_view[
            "Detected Anomaly"
        ] = (
            risk_view["Anomaly Flag"] > 0
        )

    else:

        features = risk_view[
            [
                "Budget",
                "Actual",
                "Revenue",
                "Variance",
            ]
        ].fillna(0)

        if len(features) >= 10:

            isolation_model = IsolationForest(
                contamination=0.05,
                random_state=42,
            )

            predictions = (
                isolation_model
                .fit_predict(features)
            )

            risk_view[
                "Detected Anomaly"
            ] = (
                predictions == -1
            )

        else:

            risk_view[
                "Detected Anomaly"
            ] = False

    flagged = risk_view[
        risk_view["Detected Anomaly"]
    ].copy()

    flagged_count = len(flagged)

    flagged_value = (
        flagged["Variance"].sum()
        if not flagged.empty
        else 0
    )

    risk_score = min(
        (
            flagged_count /
            max(len(risk_view), 1)
            *
            200
        ),
        100,
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        kpi_card(
            "Flagged Transactions",
            f"{flagged_count:,}",
            "Detected risk items",
        )

    with c2:

        kpi_card(
            "Flagged Exposure",
            money(flagged_value),
            "Variance exposure",
            negative=flagged_value > 0,
        )

    with c3:

        kpi_card(
            "Risk Score",
            percentage(risk_score),
            "Illustrative portfolio score",
            negative=risk_score > 50,
        )

    if not flagged.empty:

        anomaly_by_bu = (
            flagged
            .groupby(
                "Business Unit"
            )
            .size()
            .reset_index(
                name="Count"
            )
            .sort_values(
                "Count",
                ascending=False,
            )
        )

        figure = px.bar(
            anomaly_by_bu,
            x="Business Unit",
            y="Count",
            title="Anomalies by Business Unit",
            template="plotly_dark",
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

    columns_to_show = [
        "Transaction ID",
        "Date",
        "Business Unit",
        "Department",
        "Account / Cost Category",
        "Budget",
        "Actual",
        "Variance",
        "Variance %",
        "Anomaly",
    ]

    st.dataframe(
        flagged[
            columns_to_show
        ]
        .sort_values(
            "Variance",
            ascending=False
        )
        .head(250)
        .style.format(
            {
                "Budget": "₹{:,.0f}",
                "Actual": "₹{:,.0f}",
                "Variance": "₹{:,.0f}",
                "Variance %": "{:.1f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# COST INTELLIGENCE
# ============================================================

elif page == "Cost Intelligence":

    section(
        "Cost Intelligence",
        "Identify the categories creating the largest financial pressure."
    )

    left, right = st.columns(
        [1.2, 1]
    )

    with left:

        figure = px.bar(
            cost_summary.head(12),
            x="Account / Cost Category",
            y="Variance",
            title="Top Cost Variances",
            template="plotly_dark",
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

    with right:

        figure = px.pie(
            cost_summary.head(8),
            names="Account / Cost Category",
            values="Actual",
            hole=0.48,
            title="Actual Cost Mix",
            template="plotly_dark",
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

    st.dataframe(
        cost_summary.style.format(
            {
                "Budget": "₹{:,.0f}",
                "Actual": "₹{:,.0f}",
                "Variance": "₹{:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# MANAGEMENT ACTIONS
# ============================================================

elif page == "Management Actions":

    section(
        "Management Actions",
        "Prioritized management action backlog generated from unfavorable variances."
    )

    actions = []

    if not cost_summary.empty:

        threshold = (
            cost_summary["Variance"]
            .quantile(0.90)
        )

        for _, row in (
            cost_summary
            .head(15)
            .iterrows()
        ):

            if row["Variance"] > 0:

                priority = (
                    "P0"
                    if row["Variance"] >= threshold
                    else "P1"
                )

                actions.append(
                    [
                        priority,
                        "Cost Control",
                        row[
                            "Account / Cost Category"
                        ],
                        row["Variance"],
                        "Finance Controller",
                        "Investigate variance and reforecast",
                    ]
                )

    if not actions:

        st.success(
            "No unfavorable cost variances detected."
        )

    else:

        action_df = pd.DataFrame(
            actions,
            columns=[
                "Priority",
                "Type",
                "Area",
                "Exposure",
                "Owner",
                "Recommended Action",
            ],
        )

        st.dataframe(
            action_df.style.format(
                {
                    "Exposure":
                    "₹{:,.0f}"
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Action Backlog",
            action_df.to_csv(
                index=False
            ).encode("utf-8"),
            "finsight_management_actions.csv",
            "text/csv",
        )


# ============================================================
# REPORTS LIBRARY
# ============================================================

elif page == "Reports Library":

    section(
        "Reports Library",
        "Export management-ready datasets from the current filtered view."
    )

    report_bu = (
        business_unit_summary.copy()
    )

    report_department = (
        department_summary.copy()
    )

    report_cost = (
        cost_summary.copy()
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.download_button(
            "Download BU Report",
            report_bu
            .to_csv(index=False)
            .encode("utf-8"),
            "finsight_bu_performance.csv",
            "text/csv",
        )

    with c2:

        st.download_button(
            "Download Department Report",
            report_department
            .to_csv(index=False)
            .encode("utf-8"),
            "finsight_department_performance.csv",
            "text/csv",
        )

    with c3:

        st.download_button(
            "Download Cost Report",
            report_cost
            .to_csv(index=False)
            .encode("utf-8"),
            "finsight_cost_intelligence.csv",
            "text/csv",
        )

    st.markdown("")

    st.dataframe(
        report_bu.style.format(
            {
                "Revenue": "₹{:,.0f}",
                "Budget": "₹{:,.0f}",
                "Actual": "₹{:,.0f}",
                "Profit": "₹{:,.0f}",
                "Variance": "₹{:,.0f}",
                "Margin %": "{:.1f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# UPLOAD DATA
# ============================================================

elif page == "Upload Data":

    section(
        "Upload Data",
        "Load ERP-style CSV data into the FinSight semantic layer."
    )

    st.info(
        "Use the CSV uploader in the left sidebar. "
        "The dashboard recalculates automatically."
    )

    st.markdown(
        """
        **Recommended fields**

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
        view.head(100),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# ERP CONNECTIONS
# ============================================================

elif page == "ERP Connections":

    section(
        "ERP Connections",
        "Enterprise integration architecture and product roadmap."
    )

    integrations = [

        (
            "SAP / S4HANA",
            "Roadmap",
            "Planned connector for GL, cost centers, actuals and planning data.",
        ),

        (
            "Oracle Fusion",
            "Roadmap",
            "Planned finance data ingestion and semantic mapping.",
        ),

        (
            "Microsoft Dynamics",
            "Roadmap",
            "Planned ERP finance and operations integration.",
        ),

        (
            "CSV / Excel",
            "Available",
            "Current portfolio ingestion path.",
        ),
    ]

    for name, status, description in integrations:

        html_block(
            f"""
            <div class="action-card">

                <b>{name}</b>

                <span class="pill">
                    {status}
                </span>

                <br><br>

                <span class="muted">
                    {description}
                </span>

            </div>
            """
        )


# ============================================================
# DATA MAPPING
# ============================================================

elif page == "Data Mapping":

    section(
        "Data Mapping",
        "Standardized semantic layer used by the FP&A intelligence engine."
    )

    mapping = pd.DataFrame(
        {
            "Standard Field": [
                "Date",
                "Business Unit",
                "Department",
                "Account / Cost Category",
                "Budget",
                "Actual",
                "Revenue",
                "Variance",
                "Variance %",
                "Profit",
            ],

            "Purpose": [
                "Reporting period",
                "Organizational unit",
                "Management function",
                "Cost driver",
                "Plan / target",
                "Actual spend",
                "Top-line",
                "Actual minus Budget",
                "Variance / Budget",
                "Revenue minus Actual",
            ],
        }
    )

    st.dataframe(
        mapping,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# DATA QUALITY
# ============================================================

elif page == "Data Quality":

    section(
        "Data Quality",
        "Pre-reporting validation of critical finance dimensions."
    )

    required_columns = [
        "Date",
        "Business Unit",
        "Department",
        "Account / Cost Category",
        "Budget",
        "Actual",
        "Revenue",
    ]

    quality_rows = []

    for column in required_columns:

        if column in view.columns:

            missing = int(
                view[column]
                .isna()
                .sum()
            )

        else:

            missing = len(view)

        status = (
            "PASS"
            if missing == 0
            else "REVIEW"
        )

        quality_rows.append(
            [
                column,
                missing,
                status,
            ]
        )

    quality_df = pd.DataFrame(
        quality_rows,
        columns=[
            "Field",
            "Missing Values",
            "Status",
        ],
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        kpi_card(
            "Rows",
            f"{len(view):,}",
            "Current dataset",
        )

    with c2:

        kpi_card(
            "Columns",
            f"{len(view.columns):,}",
            "Normalized model",
        )

    with c3:

        quality_status = (
            "PASS"
            if (
                quality_df["Status"]
                == "REVIEW"
            ).sum()
            == 0
            else "REVIEW"
        )

        kpi_card(
            "Data Quality",
            quality_status,
            "Validation result",
            negative=quality_status == "REVIEW",
        )

    st.dataframe(
        quality_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# ALERTS
# ============================================================

elif page == "Alerts & Notifications":

    section(
        "Alerts & Notifications",
        "Rule-based financial alerts requiring management attention."
    )

    alerts = []

    if variance > 0:

        alerts.append(
            [
                "P1",
                "Cost Over Budget",
                (
                    "Total unfavorable variance is "
                    f"{money(variance)}."
                ),
            ]
        )

    if margin < 50:

        alerts.append(
            [
                "P1",
                "Margin Pressure",
                (
                    "Current margin is "
                    f"{percentage(margin)}."
                ),
            ]
        )

    if (
        not cost_summary.empty
        and
        cost_summary.iloc[0]["Variance"] > 0
    ):

        alerts.append(
            [
                "P1",
                "Top Cost Driver",
                (
                    f"{cost_summary.iloc[0]['Account / Cost Category']} "
                    f"has {money(cost_summary.iloc[0]['Variance'])} variance."
                ),
            ]
        )

    if not alerts:

        alerts.append(
            [
                "P3",
                "No Critical Alerts",
                "No major rule-based alert detected.",
            ]
        )

    alert_df = pd.DataFrame(
        alerts,
        columns=[
            "Priority",
            "Alert",
            "Details",
        ],
    )

    st.dataframe(
        alert_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# SETTINGS
# ============================================================

elif page == "Settings":

    section(
        "Settings",
        "Portfolio product configuration."
    )

    st.toggle(
        "Enable AI CFO Recommendations",
        value=True,
    )

    st.toggle(
        "Enable Anomaly Detection",
        value=True,
    )

    st.toggle(
        "Enable Management Actions",
        value=True,
    )

    st.selectbox(
        "Reporting Currency",
        [
            "INR (₹)",
            "USD ($)",
        ],
    )

    st.selectbox(
        "Fiscal Year",
        [
            "FY 2026",
            "FY 2027",
            "Calendar Year",
        ],
    )


# ============================================================
# AUDIT LOGS
# ============================================================

elif page == "Audit Logs":

    section(
        "Audit Logs",
        "Illustrative audit trail for analytics and data-processing events."
    )

    logs = pd.DataFrame(
        [
            [
                "Current Session",
                "Dataset Loaded",
                "Success",
            ],

            [
                "Current Session",
                "Semantic Normalization",
                "Success",
            ],

            [
                "Current Session",
                "Variance Calculation",
                "Success",
            ],

            [
                "Current Session",
                "Management Analytics",
                "Success",
            ],

            [
                "Current Session",
                "Dashboard Rendering",
                "Success",
            ],
        ],

        columns=[
            "Time",
            "Action",
            "Status",
        ],
    )

    st.dataframe(
        logs,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FOOTER
# ============================================================

html_block(
    """
    <div class="footer">
        FinSight AI • Agentic FP&A & ERP Intelligence •
        Portfolio Prototype • Synthetic ERP-style financial data
    </div>
    """
)
