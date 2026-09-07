import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from io import BytesIO

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FinSight AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.main {
    background-color: #f6f8fb;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 1.6rem 1.8rem;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #1e3a5f 52%,
        #0f766e 100%
    );
    color: white;
    margin-bottom: 1.4rem;
    box-shadow: 0 8px 25px rgba(15, 23, 42, 0.12);
}

.hero h1 {
    color: white;
    font-size: 2.35rem;
    margin: 0 0 0.3rem 0;
}

.hero p {
    color: #dbeafe;
    margin: 0;
    font-size: 1rem;
}

.insight-box {
    padding: 1rem 1.2rem;
    border-radius: 12px;
    background: #eff6ff;
    border-left: 5px solid #2563eb;
    margin-bottom: 0.9rem;
}

.action-box {
    padding: 1rem 1.2rem;
    border-radius: 12px;
    background: white;
    border: 1px solid #e2e8f0;
    margin-bottom: 0.9rem;
}

.section-box {
    padding: 1rem 1.2rem;
    border-radius: 12px;
    background: white;
    border: 1px solid #e2e8f0;
    margin-bottom: 1rem;
}

.status-good {
    padding: 0.8rem 1rem;
    border-radius: 10px;
    background: #ecfdf5;
    border-left: 5px solid #16a34a;
}

.status-watch {
    padding: 0.8rem 1rem;
    border-radius: 10px;
    background: #fffbeb;
    border-left: 5px solid #f59e0b;
}

.status-critical {
    padding: 0.8rem 1rem;
    border-radius: 10px;
    background: #fef2f2;
    border-left: 5px solid #dc2626;
}

.small-muted {
    color: #64748b;
    font-size: 0.85rem;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e2e8f0;
    padding: 0.8rem;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

REQUIRED_COLUMNS = [
    "Date",
    "Business Unit",
    "Account / Cost Category",
    "Budget",
    "Actual",
    "Revenue",
]


# ============================================================
# FORMATTING HELPERS
# ============================================================

def money(value):
    if value is None or pd.isna(value):
        return "₹0"

    value = float(value)
    absolute = abs(value)

    if absolute >= 1_00_00_000:
        return f"₹{value / 1_00_00_000:,.2f} Cr"

    if absolute >= 1_00_000:
        return f"₹{value / 1_00_000:,.2f} L"

    if absolute >= 1_000:
        return f"₹{value / 1_000:,.1f} K"

    return f"₹{value:,.0f}"


def money_full(value):
    if value is None or pd.isna(value):
        return "₹0"

    return f"₹{float(value):,.0f}"


def percentage(value):
    if value is None or pd.isna(value):
        return "0.00%"

    return f"{float(value):,.2f}%"


def safe_divide(a, b):
    try:
        if b == 0 or pd.isna(b):
            return 0

        return a / b

    except Exception:
        return 0


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def clean_column_name(column):
    text = str(column).strip().lower()

    replacements = {
        "_": " ",
        "-": " ",
        "/": " ",
        "&": " and ",
        ".": " ",
        "(": " ",
        ")": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return " ".join(text.split())


def normalize_columns(df):

    df = df.copy()

    aliases = {

        "Transaction ID": [
            "transaction id",
            "transaction_id",
            "transaction",
            "transaction number",
            "transaction no",
            "id",
        ],

        "Date": [
            "date",
            "transaction date",
            "transaction_date",
            "posting date",
            "posting_date",
            "document date",
            "document_date",
            "period",
            "month",
        ],

        "Business Unit": [
            "business unit",
            "business_unit",
            "businessunit",
            "bu",
            "entity",
            "company",
            "company code",
            "company_code",
        ],

        "Department": [
            "department",
            "dept",
            "department name",
            "department_name",
            "department code",
            "department_code",
        ],

        "Account / Cost Category": [
            "account",
            "account name",
            "account_name",
            "account cost category",
            "account / cost category",
            "cost category",
            "cost_category",
            "gl account",
            "gl_account",
            "gl",
            "expense category",
            "expense_category",
            "category",
        ],

        "Budget": [
            "budget",
            "budget amount",
            "budget_amount",
            "planned cost",
            "planned_cost",
            "plan",
        ],

        "Actual": [
            "actual",
            "actual amount",
            "actual_amount",
            "actual cost",
            "actual_cost",
        ],

        "Revenue": [
            "revenue",
            "revenue amount",
            "revenue_amount",
            "sales",
            "sales revenue",
            "income",
        ],

        "Variance": [
            "variance",
            "budget variance",
            "budget_variance",
            "cost variance",
            "cost_variance",
        ],

        "Variance %": [
            "variance %",
            "variance percent",
            "variance percentage",
            "variance_pct",
            "variance_percentage",
        ],

        "Profit": [
            "profit",
            "profit amount",
            "profit_amount",
            "operating profit",
            "operating_profit",
        ],

        "Anomaly Flag": [
            "anomaly flag",
            "anomaly_flag",
            "anomaly indicator",
            "anomaly_indicator",
        ],

        "Anomaly": [
            "anomaly",
            "anomaly type",
            "anomaly_type",
            "risk flag",
            "risk_flag",
        ],
    }

    lookup = {}

    for column in df.columns:
        key = clean_column_name(column)

        if key not in lookup:
            lookup[key] = column

    rename_map = {}

    for canonical, names in aliases.items():

        canonical_key = clean_column_name(canonical)

        if canonical_key in lookup:

            original = lookup[canonical_key]

            if original != canonical:
                rename_map[original] = canonical

            continue

        for name in names:

            key = clean_column_name(name)

            if key in lookup:

                original = lookup[key]

                if original != canonical:
                    rename_map[original] = canonical

                break

    return df.rename(columns=rename_map)


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(df):

    df = normalize_columns(df)

    # Department is optional in FinSight.
    if "Department" not in df.columns:
        df["Department"] = "Unmapped"
    else:
        df["Department"] = (
            df["Department"]
            .fillna("Unmapped")
            .astype(str)
            .str.strip()
        )

        df.loc[
            df["Department"].isin(
                ["", "nan", "None"]
            ),
            "Department"
        ] = "Unmapped"

    # Validate core columns.
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:

        detected = ", ".join(
            str(column)
            for column in df.columns
        )

        return None, (
            "Missing required columns: "
            + ", ".join(missing)
            + ". Detected columns: "
            + detected
        )

    # Dates.
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # Numeric fields.
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
            )

    # Core financial values.
    for column in [
        "Budget",
        "Actual",
        "Revenue",
    ]:

        df[column] = (
            df[column]
            .fillna(0)
        )

    # Variance.
    if "Variance" not in df.columns:

        df["Variance"] = (
            df["Actual"]
            - df["Budget"]
        )

    else:

        df["Variance"] = (
            df["Variance"]
            .fillna(
                df["Actual"]
                - df["Budget"]
            )
        )

    # Variance percentage.
    if "Variance %" not in df.columns:

        df["Variance %"] = np.where(
            df["Budget"].abs() > 0,
            (
                df["Actual"]
                - df["Budget"]
            )
            / df["Budget"].abs()
            * 100,
            0,
        )

    # Profit.
    if "Profit" not in df.columns:

        df["Profit"] = (
            df["Revenue"]
            - df["Actual"]
        )

    # Anomaly fields.
    if "Anomaly Flag" not in df.columns:
        df["Anomaly Flag"] = 0
    else:
        df["Anomaly Flag"] = (
            df["Anomaly Flag"]
            .fillna(0)
        )

    if "Anomaly" not in df.columns:
        df["Anomaly"] = "Normal"
    else:
        df["Anomaly"] = (
            df["Anomaly"]
            .fillna("Normal")
            .astype(str)
        )

    # Transaction ID.
    if "Transaction ID" not in df.columns:

        df["Transaction ID"] = [
            f"TXN-{i:05d}"
            for i in range(
                1,
                len(df) + 1
            )
        ]

    # Business Unit.
    df["Business Unit"] = (
        df["Business Unit"]
        .fillna("Unmapped")
        .astype(str)
        .str.strip()
    )

    # Cost category.
    df[
        "Account / Cost Category"
    ] = (
        df[
            "Account / Cost Category"
        ]
        .fillna("Unmapped")
        .astype(str)
        .str.strip()
    )

    # Remove invalid dates.
    df = df.dropna(
        subset=["Date"]
    ).copy()

    return (
        df.sort_values("Date")
        .reset_index(drop=True),
        None
    )


# ============================================================
# LOAD DEMO DATA
# ============================================================

@st.cache_data
def load_demo_data():

    possible_paths = [
        Path(
            "synthetic_erp_financials.csv"
        ),
        Path(
            "data/synthetic_erp_financials.csv"
        ),
    ]

    file_path = None

    for path in possible_paths:

        if path.exists():

            file_path = path
            break

    if file_path is None:

        return None, (
            "FinSight demo dataset was not found. "
            "Please make sure synthetic_erp_financials.csv "
            "is in the repository."
        )

    try:

        raw = pd.read_csv(
            file_path
        )

        return prepare_data(raw)

    except Exception as exc:

        return None, (
            "Unable to load demo data: "
            + str(exc)
        )


# ============================================================
# LOAD UPLOADED CSV
# ============================================================

@st.cache_data
def load_uploaded_data(file_bytes):

    try:

        raw = pd.read_csv(
            BytesIO(file_bytes)
        )

        return prepare_data(raw)

    except Exception as exc:

        return None, (
            "Unable to read uploaded CSV: "
            + str(exc)
        )


# ============================================================
# KPI ENGINE
# ============================================================

def calculate_kpis(df):

    revenue = df["Revenue"].sum()

    budget = df["Budget"].sum()

    actual = df["Actual"].sum()

    variance = actual - budget

    profit = df["Profit"].sum()

    margin = (
        safe_divide(
            profit,
            revenue
        )
        * 100
    )

    return {
        "revenue": revenue,
        "budget": budget,
        "actual": actual,
        "variance": variance,
        "profit": profit,
        "margin": margin,
    }


# ============================================================
# MONTHLY SUMMARY
# ============================================================

def monthly_summary(df):

    result = (
        df.assign(
            Month=(
                df["Date"]
                .dt
                .to_period("M")
                .astype(str)
            )
        )
        .groupby("Month")
        .agg(
            Revenue=("Revenue", "sum"),
            Budget=("Budget", "sum"),
            Actual=("Actual", "sum"),
            Profit=("Profit", "sum"),
        )
        .reset_index()
    )

    result["Variance"] = (
        result["Actual"]
        - result["Budget"]
    )

    result["Margin %"] = np.where(
        result["Revenue"] != 0,
        result["Profit"]
        / result["Revenue"]
        * 100,
        0,
    )

    return result


# ============================================================
# BUSINESS UNIT SUMMARY
# ============================================================

def business_unit_summary(df):

    result = (
        df.groupby("Business Unit")
        .agg(
            Revenue=("Revenue", "sum"),
            Budget=("Budget", "sum"),
            Actual=("Actual", "sum"),
            Profit=("Profit", "sum"),
            Transactions=(
                "Transaction ID",
                "count"
            ),
        )
        .reset_index()
    )

    result["Variance"] = (
        result["Actual"]
        - result["Budget"]
    )

    result["Variance %"] = np.where(
        result["Budget"].abs() != 0,
        result["Variance"]
        / result["Budget"].abs()
        * 100,
        0,
    )

    result["Margin %"] = np.where(
        result["Revenue"] != 0,
        result["Profit"]
        / result["Revenue"]
        * 100,
        0,
    )

    return result.sort_values(
        "Variance",
        ascending=False
    )


# ============================================================
# DEPARTMENT SUMMARY
# ============================================================

def department_summary(df):

    result = (
        df.groupby("Department")
        .agg(
            Revenue=("Revenue", "sum"),
            Budget=("Budget", "sum"),
            Actual=("Actual", "sum"),
            Profit=("Profit", "sum"),
            Transactions=(
                "Transaction ID",
                "count"
            ),
        )
        .reset_index()
    )

    result["Variance"] = (
        result["Actual"]
        - result["Budget"]
    )

    result["Variance %"] = np.where(
        result["Budget"].abs() != 0,
        result["Variance"]
        / result["Budget"].abs()
        * 100,
        0,
    )

    result["Margin %"] = np.where(
        result["Revenue"] != 0,
        result["Profit"]
        / result["Revenue"]
        * 100,
        0,
    )

    return result.sort_values(
        "Variance",
        ascending=False
    )


# ============================================================
# COST CATEGORY SUMMARY
# ============================================================

def category_summary(df):

    result = (
        df.groupby(
            "Account / Cost Category"
        )
        .agg(
            Revenue=("Revenue", "sum"),
            Budget=("Budget", "sum"),
            Actual=("Actual", "sum"),
            Transactions=(
                "Transaction ID",
                "count"
            ),
        )
        .reset_index()
    )

    result["Variance"] = (
        result["Actual"]
        - result["Budget"]
    )

    result["Variance %"] = np.where(
        result["Budget"].abs() != 0,
        result["Variance"]
        / result["Budget"].abs()
        * 100,
        0,
    )

    return result.sort_values(
        "Variance",
        ascending=False
    )


# ============================================================
# ANOMALY DETECTION
# ============================================================

def detect_anomalies(df):

    result = df.copy()

    if len(result) < 10:

        result["Anomaly Flag"] = 0
        result["Anomaly"] = "Normal"

        return result

    features = result[
        [
            "Budget",
            "Actual",
            "Revenue",
            "Variance",
        ]
    ].fillna(0)

    try:

        model = IsolationForest(
            contamination=0.05,
            random_state=42,
        )

        predictions = (
            model.fit_predict(features)
        )

        result["Anomaly Flag"] = np.where(
            predictions == -1,
            1,
            0,
        )

        result["Anomaly"] = np.where(
            result["Anomaly Flag"] == 1,
            "Potential Anomaly",
            "Normal",
        )

    except Exception:

        result["Anomaly Flag"] = 0
        result["Anomaly"] = "Normal"

    return result


# ============================================================
# AI CFO DECISION ENGINE
# ============================================================

def generate_ai_cfo_insights(df):

    insights = []

    kpis = calculate_kpis(df)

    categories = category_summary(df)

    business_units = (
        business_unit_summary(df)
    )

    # --------------------------------------------------------
    # COST DRIVER
    # --------------------------------------------------------

    unfavorable_categories = (
        categories[
            categories["Variance"] > 0
        ]
    )

    if not unfavorable_categories.empty:

        top = (
            unfavorable_categories
            .iloc[0]
        )

        insights.append({

            "Finding": (
                f"{top['Account / Cost Category']} "
                "is the largest unfavorable cost driver."
            ),

            "Evidence": (
                f"Unfavorable variance of "
                f"{money_full(top['Variance'])} "
                f"({percentage(top['Variance %'])})."
            ),

            "Business Impact": (
                "Continued overspend may put pressure "
                "on operating margin and forecast accuracy."
            ),

            "Management Action": (
                "Review vendor spend, utilization, "
                "headcount, contracts and discretionary "
                "expenses within this category."
            ),

            "Priority": "P0",

            "Owner": "FP&A / Cost Owner",
        })

    # --------------------------------------------------------
    # BUSINESS UNIT
    # --------------------------------------------------------

    unfavorable_bu = (
        business_units[
            business_units["Variance"] > 0
        ]
    )

    if not unfavorable_bu.empty:

        top_bu = (
            unfavorable_bu
            .iloc[0]
        )

        insights.append({

            "Finding": (
                f"{top_bu['Business Unit']} "
                "has the highest unfavorable "
                "budget variance."
            ),

            "Evidence": (
                f"Actual cost exceeds budget by "
                f"{money_full(top_bu['Variance'])}."
            ),

            "Business Impact": (
                "The business unit is creating "
                "disproportionate cost pressure "
                "versus plan."
            ),

            "Management Action": (
                "Perform a BU-level review of "
                "controllable versus committed costs "
                "and refresh the forecast."
            ),

            "Priority": "P1",

            "Owner": "BU Finance Partner",
        })

    # --------------------------------------------------------
    # MARGIN
    # --------------------------------------------------------

    if kpis["margin"] < 50:

        insights.append({

            "Finding": (
                "Operating margin is below "
                "the management threshold."
            ),

            "Evidence": (
                f"Current modeled margin is "
                f"{percentage(kpis['margin'])}."
            ),

            "Business Impact": (
                "Lower margin can reduce "
                "profitability and investment capacity."
            ),

            "Management Action": (
                "Review pricing, revenue mix, "
                "major cost drivers and resource allocation."
            ),

            "Priority": "P1",

            "Owner": "FP&A / Leadership",
        })

    # --------------------------------------------------------
    # ANOMALIES
    # --------------------------------------------------------

    anomaly_count = int(
        df["Anomaly Flag"].sum()
    )

    if anomaly_count > 0:

        insights.append({

            "Finding": (
                f"{anomaly_count:,} transaction(s) "
                "require anomaly review."
            ),

            "Evidence": (
                "Unusual financial behavior was "
                "identified by the anomaly detection layer."
            ),

            "Business Impact": (
                "Unusual transactions may distort "
                "forecast, budget and management reporting."
            ),

            "Management Action": (
                "Investigate flagged transactions "
                "and validate business justification."
            ),

            "Priority": "P1",

            "Owner": "Finance Controller",
        })

    return pd.DataFrame(insights)


# ============================================================
# CASH FLOW MODEL
# ============================================================

def build_cash_flow(df):

    monthly = monthly_summary(df).copy()

    if monthly.empty:
        return monthly

    # Synthetic prototype assumptions.
    monthly["Cash Inflow"] = (
        monthly["Revenue"] * 0.85
    )

    monthly["Cash Outflow"] = (
        monthly["Actual"] * 0.80
    )

    opening_cash = 1_00_00_000

    current_cash = opening_cash

    opening_values = []
    closing_values = []

    for _, row in monthly.iterrows():

        opening_values.append(
            current_cash
        )

        current_cash = (
            current_cash
            + row["Cash Inflow"]
            - row["Cash Outflow"]
        )

        closing_values.append(
            current_cash
        )

    monthly["Opening Cash"] = opening_values

    monthly["Closing Cash"] = closing_values

    monthly["Net Cash Flow"] = (
        monthly["Cash Inflow"]
        - monthly["Cash Outflow"]
    )

    monthly["Liquidity Status"] = np.where(

        monthly["Closing Cash"] < 0,

        "Critical",

        np.where(

            monthly["Closing Cash"]
            < opening_cash * 0.50,

            "Watch",

            "Healthy",
        ),
    )

    return monthly


# ============================================================
# FORECAST ENGINE
# ============================================================

def build_forecast(df, periods=6):

    monthly = monthly_summary(
        df
    ).copy()

    if len(monthly) < 3:
        return None

    monthly["Period Number"] = np.arange(
        len(monthly)
    )

    future_periods = np.arange(
        len(monthly),
        len(monthly) + periods
    )

    future_frame = pd.DataFrame({
        "Period Number": future_periods
    })

    # --------------------------------------------------------
    # BASE REVENUE MODEL
    # --------------------------------------------------------

    revenue_model = LinearRegression()

    revenue_model.fit(
        monthly[["Period Number"]],
        monthly["Revenue"]
    )

    base_revenue = (
        revenue_model
        .predict(future_frame)
    )

    # --------------------------------------------------------
    # BASE COST MODEL
    # --------------------------------------------------------

    cost_model = LinearRegression()

    cost_model.fit(
        monthly[["Period Number"]],
        monthly["Actual"]
    )

    base_cost = (
        cost_model
        .predict(future_frame)
    )

    base_revenue = np.maximum(
        base_revenue,
        0
    )

    base_cost = np.maximum(
        base_cost,
        0
    )

    # --------------------------------------------------------
    # FUTURE MONTHS
    # --------------------------------------------------------

    last_month = pd.to_datetime(
        monthly["Month"].iloc[-1]
    )

    future_dates = pd.date_range(
        start=(
            last_month
            + pd.offsets.MonthBegin(1)
        ),
        periods=periods,
        freq="MS"
    )

    forecast = pd.DataFrame({

        "Month":
            future_dates.strftime("%Y-%m"),

        "Base Revenue":
            base_revenue,

        "Base Cost":
            base_cost,
    })

    forecast["Base Profit"] = (
        forecast["Base Revenue"]
        - forecast["Base Cost"]
    )

    # ========================================================
    # OPTIMISTIC CASE
    #
    # Revenue +12%
    # Cost -5%
    # ========================================================

    forecast["Optimistic Revenue"] = (
        forecast["Base Revenue"] * 1.12
    )

    forecast["Optimistic Cost"] = (
        forecast["Base Cost"] * 0.95
    )

    forecast["Optimistic Profit"] = (
        forecast["Optimistic Revenue"]
        - forecast["Optimistic Cost"]
    )

    # ========================================================
    # DOWNSIDE CASE
    #
    # Revenue -12%
    # Cost +10%
    # ========================================================

    forecast["Downside Revenue"] = (
        forecast["Base Revenue"] * 0.88
    )

    forecast["Downside Cost"] = (
        forecast["Base Cost"] * 1.10
    )

    forecast["Downside Profit"] = (
        forecast["Downside Revenue"]
        - forecast["Downside Cost"]
    )

    return forecast


# ============================================================
# ACTION CENTER ENGINE
# ============================================================

def build_action_center(df):

    actions = []

    categories = category_summary(df)

    for _, row in (
        categories.head(10)
        .iterrows()
    ):

        variance = row["Variance"]

        if variance <= 0:
            continue

        variance_pct = row["Variance %"]

        if variance_pct >= 15:
            priority = "P0"

        elif variance_pct >= 8:
            priority = "P1"

        else:
            priority = "P2"

        actions.append({

            "Priority": priority,

            "Area": (
                row[
                    "Account / Cost Category"
                ]
            ),

            "Issue": (
                "Unfavorable cost variance"
            ),

            "Financial Impact": (
                money_full(variance)
            ),

            "Recommended Action": (
                "Review spend against budget, "
                "identify controllable costs "
                "and refresh the forecast."
            ),

            "Owner": (
                "FP&A / Cost Owner"
            ),

            "Status": "Open",
        })

    return pd.DataFrame(actions)


# ============================================================
# SIDEBAR — DATA SOURCE
# ============================================================

with st.sidebar:

    st.markdown(
        """
<div>
<h2>📊 FinSight AI</h2>
<div class="small-muted">
Agentic FP&A & ERP Intelligence
</div>
</div>
""",
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        "### Data Source"
    )

    uploaded_file = st.file_uploader(
        "Upload ERP CSV",
        type=["csv"],
        help=(
            "Upload an ERP-style financial CSV. "
            "Department is optional."
        ),
    )

    if uploaded_file is not None:

        df, data_error = (
            load_uploaded_data(
                uploaded_file.getvalue()
            )
        )

        data_source = (
            "Uploaded ERP CSV"
        )

    else:

        df, data_error = (
            load_demo_data()
        )

        data_source = (
            "FinSight Demo ERP Data"
        )

    st.caption(
        f"Source: {data_source}"
    )


# ============================================================
# DATA ERROR HANDLING
# ============================================================

if df is None:

    st.error(data_error)

    st.markdown(
        """
### Required ERP fields

FinSight requires:

- Date
- Business Unit
- Account / Cost Category
- Budget
- Actual
- Revenue

**Department is optional.**

Common ERP column variations are automatically recognized.
"""
    )

    st.stop()


# ============================================================
# NAVIGATION
# ============================================================

with st.sidebar:

    st.divider()

    st.markdown(
        "### Navigation"
    )

    pages = [
        "🏠 Executive Dashboard",
        "📊 FP&A Performance",
        "⚠️ Risk & Anomalies",
        "💵 Cash & Liquidity",
        "📈 Forecast & Scenarios",
        "🤖 AI CFO",
        "🎯 Action Center",
        "🔎 Data Explorer",
    ]

    selected_page = st.radio(
        "Navigation",
        pages,
        label_visibility="collapsed",
    )


# ============================================================
# GLOBAL FILTERS
# ============================================================

with st.sidebar:

    st.divider()

    st.markdown(
        "### Global Filters"
    )

    business_units = [
        "All"
    ] + sorted(
        df["Business Unit"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_bu = st.selectbox(
        "Business Unit",
        business_units,
    )

    departments = [
        "All"
    ] + sorted(
        df["Department"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_department = st.selectbox(
        "Department",
        departments,
    )


# ============================================================
# APPLY FILTERS
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
# HERO
# ============================================================

st.markdown(
    """<div class="hero">
<h1>📊 FinSight AI</h1>
<p>Agentic FP&A & ERP Intelligence Command Center — turning financial data into management decisions.</p>
</div>""",
    unsafe_allow_html=True,
)


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

if selected_page == "🏠 Executive Dashboard":

    st.title(
        "Executive Dashboard"
    )

    st.caption(
        "Executive view of revenue, cost, profitability, "
        "budget performance and financial risk."
    )

    kpis = calculate_kpis(
        filtered_df
    )

    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )

    col1.metric(
        "Revenue",
        money(kpis["revenue"])
    )

    col2.metric(
        "Actual Cost",
        money(kpis["actual"])
    )

    col3.metric(
        "Budget",
        money(kpis["budget"])
    )

    col4.metric(
        "Variance",
        money(kpis["variance"])
    )

    col5.metric(
        "Profit Margin",
        percentage(kpis["margin"])
    )

    st.divider()

    monthly = monthly_summary(
        filtered_df
    )

    left, right = st.columns(2)

    with left:

        st.subheader(
            "Revenue vs Actual Cost"
        )

        if not monthly.empty:

            chart_data = monthly.melt(
                id_vars=["Month"],
                value_vars=[
                    "Revenue",
                    "Actual"
                ],
                var_name="Metric",
                value_name="Amount"
            )

            fig = px.line(
                chart_data,
                x="Month",
                y="Amount",
                color="Metric",
                markers=True
            )

            fig.update_layout(
                height=380,
                xaxis_title="Month",
                yaxis_title="Amount",
                legend_title=""
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with right:

        st.subheader(
            "Business Unit Variance"
        )

        bu = business_unit_summary(
            filtered_df
        )

        if not bu.empty:

            fig = px.bar(
                bu,
                x="Business Unit",
                y="Variance"
            )

            fig.add_hline(
                y=0,
                line_dash="dash"
            )

            fig.update_layout(
                height=380
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.subheader(
        "🤖 AI CFO Executive Snapshot"
    )

    insights = (
        generate_ai_cfo_insights(
            filtered_df
        )
    )

    if insights.empty:

        st.success(
            "No material management risks were identified."
        )

    else:

        for _, row in (
            insights.head(3)
            .iterrows()
        ):

            st.markdown(
                f"""<div class="insight-box">
<b>{row["Priority"]} — {row["Finding"]}</b>
<br><br>
<b>Evidence:</b> {row["Evidence"]}
<br><br>
<b>Management Action:</b> {row["Management Action"]}
</div>""",
                unsafe_allow_html=True
            )


# ============================================================
# FP&A PERFORMANCE
# ============================================================

elif selected_page == "📊 FP&A Performance":

    st.title(
        "FP&A Performance"
    )

    st.caption(
        "Budget vs actual, variance, profitability "
        "and cost-driver analysis."
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Business Units",
            "Departments",
            "Cost Categories"
        ]
    )

    # --------------------------------------------------------
    # BUSINESS UNITS
    # --------------------------------------------------------

    with tab1:

        result = business_unit_summary(
            filtered_df
        )

        display = result.copy()

        for column in [
            "Revenue",
            "Budget",
            "Actual",
            "Profit",
            "Variance"
        ]:

            display[column] = (
                display[column]
                .map(money_full)
            )

        display["Variance %"] = (
            result["Variance %"]
            .map(percentage)
        )

        display["Margin %"] = (
            result["Margin %"]
            .map(percentage)
        )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # DEPARTMENTS
    # --------------------------------------------------------

    with tab2:

        result = department_summary(
            filtered_df
        )

        display = result.copy()

        for column in [
            "Revenue",
            "Budget",
            "Actual",
            "Profit",
            "Variance"
        ]:

            display[column] = (
                display[column]
                .map(money_full)
            )

        display["Variance %"] = (
            result["Variance %"]
            .map(percentage)
        )

        display["Margin %"] = (
            result["Margin %"]
            .map(percentage)
        )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # COST CATEGORIES
    # --------------------------------------------------------

    with tab3:

        result = category_summary(
            filtered_df
        )

        display = result.copy()

        for column in [
            "Revenue",
            "Budget",
            "Actual",
            "Variance"
        ]:

            display[column] = (
                display[column]
                .map(money_full)
            )

        display["Variance %"] = (
            result["Variance %"]
            .map(percentage)
        )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.subheader(
        "Monthly Financial Trend"
    )

    monthly = monthly_summary(
        filtered_df
    )

    display = monthly.copy()

    for column in [
        "Revenue",
        "Budget",
        "Actual",
        "Profit",
        "Variance"
    ]:

        display[column] = (
            display[column]
            .map(money_full)
        )

    display["Margin %"] = (
        monthly["Margin %"]
        .map(percentage)
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# RISK & ANOMALIES
# ============================================================

elif selected_page == "⚠️ Risk & Anomalies":

    st.title(
        "Risk & Anomalies"
    )

    st.caption(
        "Machine-learning anomaly detection "
        "and financial risk monitoring."
    )

    anomaly_df = detect_anomalies(
        filtered_df
    )

    anomaly_count = int(
        anomaly_df[
            "Anomaly Flag"
        ].sum()
    )

    total_transactions = len(
        anomaly_df
    )

    anomaly_rate = (
        safe_divide(
            anomaly_count,
            total_transactions
        )
        * 100
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Transactions",
        f"{total_transactions:,}"
    )

    col2.metric(
        "Potential Anomalies",
        f"{anomaly_count:,}"
    )

    col3.metric(
        "Anomaly Rate",
        percentage(anomaly_rate)
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader(
            "Anomalies by Business Unit"
        )

        anomaly_bu = (
            anomaly_df[
                anomaly_df[
                    "Anomaly Flag"
                ] == 1
            ]
            .groupby(
                "Business Unit"
            )
            .size()
            .reset_index(
                name="Anomalies"
            )
        )

        if anomaly_bu.empty:

            st.success(
                "No anomalies detected."
            )

        else:

            fig = px.bar(
                anomaly_bu,
                x="Business Unit",
                y="Anomalies"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with right:

        st.subheader(
            "Anomalies by Cost Category"
        )

        anomaly_category = (
            anomaly_df[
                anomaly_df[
                    "Anomaly Flag"
                ] == 1
            ]
            .groupby(
                "Account / Cost Category"
            )
            .size()
            .reset_index(
                name="Anomalies"
            )
        )

        if anomaly_category.empty:

            st.success(
                "No anomalies detected."
            )

        else:

            fig = px.bar(
                anomaly_category,
                x="Account / Cost Category",
                y="Anomalies"
            )

            fig.update_layout(
                xaxis_tickangle=-35
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.subheader(
        "Flagged Transactions"
    )

    flagged = anomaly_df[
        anomaly_df[
            "Anomaly Flag"
        ] == 1
    ].copy()

    if flagged.empty:

        st.success(
            "No unusual transactions identified."
        )

    else:

        columns = [
            "Transaction ID",
            "Date",
            "Business Unit",
            "Department",
            "Account / Cost Category",
            "Budget",
            "Actual",
            "Variance",
            "Variance %",
            "Anomaly"
        ]

        available = [
            column
            for column in columns
            if column in flagged.columns
        ]

        display = flagged[
            available
        ].copy()

        for column in [
            "Budget",
            "Actual",
            "Variance"
        ]:

            if column in display.columns:

                display[column] = (
                    display[column]
                    .map(money_full)
                )

        if "Variance %" in display.columns:

            display["Variance %"] = (
                flagged[
                    "Variance %"
                ]
                .map(percentage)
            )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# CASH & LIQUIDITY
# ============================================================

elif selected_page == "💵 Cash & Liquidity":

    st.title(
        "Cash & Liquidity Command Center"
    )

    st.caption(
        "Modeled cash-flow intelligence for "
        "working-capital and liquidity planning."
    )

    cash = build_cash_flow(
        filtered_df
    )

    if cash.empty:

        st.warning(
            "Insufficient financial history "
            "to model cash flow."
        )

    else:

        latest = cash.iloc[-1]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Cash Inflow",
            money(latest["Cash Inflow"])
        )

        col2.metric(
            "Cash Outflow",
            money(latest["Cash Outflow"])
        )

        col3.metric(
            "Net Cash Flow",
            money(latest["Net Cash Flow"])
        )

        col4.metric(
            "Closing Cash",
            money(latest["Closing Cash"])
        )

        st.divider()

        left, right = st.columns(2)

        with left:

            st.subheader(
                "Cash Inflow vs Outflow"
            )

            chart_data = cash.melt(
                id_vars=["Month"],
                value_vars=[
                    "Cash Inflow",
                    "Cash Outflow"
                ],
                var_name="Metric",
                value_name="Amount"
            )

            fig = px.bar(
                chart_data,
                x="Month",
                y="Amount",
                color="Metric",
                barmode="group"
            )

            fig.update_layout(
                height=380
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with right:

            st.subheader(
                "Closing Cash Trend"
            )

            fig = px.line(
                cash,
                x="Month",
                y="Closing Cash",
                markers=True
            )

            fig.update_layout(
                height=380
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader(
            "Liquidity Status"
        )

        status = latest[
            "Liquidity Status"
        ]

        if status == "Healthy":

            st.markdown(
                """<div class="status-good">
<b>Liquidity Status: Healthy</b><br>
Modeled closing cash remains above the monitoring threshold.
</div>""",
                unsafe_allow_html=True
            )

        elif status == "Watch":

            st.markdown(
                """<div class="status-watch">
<b>Liquidity Status: Watch</b><br>
Modeled cash position is approaching the monitoring threshold.
</div>""",
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """<div class="status-critical">
<b>Liquidity Status: Critical</b><br>
Modeled closing cash has fallen below zero.
</div>""",
                unsafe_allow_html=True
            )

        st.subheader(
            "Monthly Liquidity View"
        )

        display = cash.copy()

        for column in [
            "Cash Inflow",
            "Cash Outflow",
            "Net Cash Flow",
            "Opening Cash",
            "Closing Cash"
        ]:

            display[column] = (
                display[column]
                .map(money_full)
            )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "Prototype note: cash flow is modeled "
            "from synthetic ERP-style financial data. "
            "It is not a live bank or treasury integration."
        )


# ============================================================
# FORECAST & SCENARIOS
# ============================================================

elif selected_page == "📈 Forecast & Scenarios":

    st.title(
        "Forecast & Scenario Planner"
    )

    st.caption(
        "Forward-looking revenue, cost and profitability "
        "scenarios for FP&A planning."
    )

    forecast = build_forecast(
        filtered_df,
        periods=6
    )

    if forecast is None:

        st.warning(
            "At least three months of financial history "
            "are required for forecasting."
        )

    else:

        historical = monthly_summary(
            filtered_df
        )

        # ----------------------------------------------------
        # HISTORICAL GROWTH
        # ----------------------------------------------------

        if len(historical) >= 2:

            first_revenue = historical[
                "Revenue"
            ].iloc[0]

            last_revenue = historical[
                "Revenue"
            ].iloc[-1]

            period_count = max(
                len(historical) - 1,
                1
            )

            historical_growth = (
                (
                    safe_divide(
                        last_revenue,
                        first_revenue
                    )
                ** (1 / period_count)
                - 1
                )
                * 100
            )

        else:

            historical_growth = 0

        historical_margin = (
            safe_divide(
                historical["Profit"].sum(),
                historical["Revenue"].sum()
            )
            * 100
        )

        # ----------------------------------------------------
        # FORECAST KPIs
        # ----------------------------------------------------

        base_profit_total = forecast[
            "Base Profit"
        ].sum()

        optimistic_profit_total = forecast[
            "Optimistic Profit"
        ].sum()

        downside_profit_total = forecast[
            "Downside Profit"
        ].sum()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Historical Growth",
            percentage(
                historical_growth
            )
        )

        col2.metric(
            "Historical Margin",
            percentage(
                historical_margin
            )
        )

        col3.metric(
            "Forecast Horizon",
            "6 Months"
        )

        col4.metric(
            "Base Forecast Profit",
            money(
                base_profit_total
            )
        )

        st.divider()

        # ----------------------------------------------------
        # SCENARIO ASSUMPTIONS
        # ----------------------------------------------------

        st.subheader(
            "Scenario Assumptions"
        )

        a1, a2, a3 = st.columns(3)

        with a1:

            st.markdown(
                """<div class="section-box">
<b>📊 BASE CASE</b>
<br><br>
Revenue: Trend-based
<br>
Cost: Trend-based
<br>
Assumption: Current trajectory continues
</div>""",
                unsafe_allow_html=True
            )

        with a2:

            st.markdown(
                """<div class="section-box">
<b>🚀 OPTIMISTIC CASE</b>
<br><br>
Revenue: <b>+12%</b> vs Base
<br>
Cost: <b>-5%</b> vs Base
<br>
Assumption: Stronger growth + cost discipline
</div>""",
                unsafe_allow_html=True
            )

        with a3:

            st.markdown(
                """<div class="section-box">
<b>⚠️ DOWNSIDE CASE</b>
<br><br>
Revenue: <b>-12%</b> vs Base
<br>
Cost: <b>+10%</b> vs Base
<br>
Assumption: Slower growth + cost pressure
</div>""",
                unsafe_allow_html=True
            )

        st.divider()

        # ----------------------------------------------------
        # REVENUE SCENARIOS
        # ----------------------------------------------------

        st.subheader(
            "Revenue Scenario Comparison"
        )

        revenue_chart = forecast[
            [
                "Month",
                "Base Revenue",
                "Optimistic Revenue",
                "Downside Revenue"
            ]
        ].melt(
            id_vars=["Month"],
            var_name="Scenario",
            value_name="Revenue"
        )

        revenue_chart["Scenario"] = (
            revenue_chart["Scenario"]
            .replace({
                "Base Revenue": "Base",
                "Optimistic Revenue": "Optimistic",
                "Downside Revenue": "Downside"
            })
        )

        fig = px.line(
            revenue_chart,
            x="Month",
            y="Revenue",
            color="Scenario",
            markers=True
        )

        fig.update_layout(
            height=430,
            xaxis_title="Forecast Month",
            yaxis_title="Revenue",
            legend_title="Scenario"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ----------------------------------------------------
        # PROFIT SCENARIOS
        # ----------------------------------------------------

        st.subheader(
            "Profit Scenario Comparison"
        )

        profit_chart = forecast[
            [
                "Month",
                "Base Profit",
                "Optimistic Profit",
                "Downside Profit"
            ]
        ].melt(
            id_vars=["Month"],
            var_name="Scenario",
            value_name="Profit"
        )

        profit_chart["Scenario"] = (
            profit_chart["Scenario"]
            .replace({
                "Base Profit": "Base",
                "Optimistic Profit": "Optimistic",
                "Downside Profit": "Downside"
            })
        )

        fig = px.line(
            profit_chart,
            x="Month",
            y="Profit",
            color="Scenario",
            markers=True
        )

        fig.update_layout(
            height=430,
            xaxis_title="Forecast Month",
            yaxis_title="Profit",
            legend_title="Scenario"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()

        # ----------------------------------------------------
        # SIX-MONTH SCENARIO TOTALS
        # ----------------------------------------------------

        st.subheader(
            "6-Month Scenario Summary"
        )

        base_revenue_total = forecast[
            "Base Revenue"
        ].sum()

        optimistic_revenue_total = forecast[
            "Optimistic Revenue"
        ].sum()

        downside_revenue_total = forecast[
            "Downside Revenue"
        ].sum()

        base_cost_total = forecast[
            "Base Cost"
        ].sum()

        optimistic_cost_total = forecast[
            "Optimistic Cost"
        ].sum()

        downside_cost_total = forecast[
            "Downside Cost"
        ].sum()

        base_margin = (
            safe_divide(
                base_profit_total,
                base_revenue_total
            )
            * 100
        )

        optimistic_margin = (
            safe_divide(
                optimistic_profit_total,
                optimistic_revenue_total
            )
            * 100
        )

        downside_margin = (
            safe_divide(
                downside_profit_total,
                downside_revenue_total
            )
            * 100
        )

        summary = pd.DataFrame({

            "Scenario": [
                "📊 Base",
                "🚀 Optimistic",
                "⚠️ Downside"
            ],

            "6M Revenue": [
                money_full(
                    base_revenue_total
                ),
                money_full(
                    optimistic_revenue_total
                ),
                money_full(
                    downside_revenue_total
                )
            ],

            "6M Cost": [
                money_full(
                    base_cost_total
                ),
                money_full(
                    optimistic_cost_total
                ),
                money_full(
                    downside_cost_total
                )
            ],

            "6M Profit": [
                money_full(
                    base_profit_total
                ),
                money_full(
                    optimistic_profit_total
                ),
                money_full(
                    downside_profit_total
                )
            ],

            "Profit Margin": [
                percentage(
                    base_margin
                ),
                percentage(
                    optimistic_margin
                ),
                percentage(
                    downside_margin
                )
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # MONTHLY FORECAST DETAIL
        # ----------------------------------------------------

        st.subheader(
            "Monthly Forecast Detail"
        )

        detail = forecast[
            [
                "Month",
                "Base Revenue",
                "Base Cost",
                "Base Profit",
                "Optimistic Revenue",
                "Optimistic Cost",
                "Optimistic Profit",
                "Downside Revenue",
                "Downside Cost",
                "Downside Profit"
            ]
        ].copy()

        currency_columns = [
            "Base Revenue",
            "Base Cost",
            "Base Profit",
            "Optimistic Revenue",
            "Optimistic Cost",
            "Optimistic Profit",
            "Downside Revenue",
            "Downside Cost",
            "Downside Profit"
        ]

        for column in currency_columns:

            detail[column] = (
                detail[column]
                .map(money_full)
            )

        st.dataframe(
            detail,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # MANAGEMENT INTERPRETATION
        # ----------------------------------------------------

        st.subheader(
            "🤖 AI CFO Scenario Interpretation"
        )

        profit_range = (
            optimistic_profit_total
            - downside_profit_total
        )

        st.markdown(
            f"""<div class="insight-box">
<b>Management Finding</b>
<br><br>
The six-month modeled profit range between the
Optimistic and Downside scenarios is
<b>{money_full(profit_range)}</b>.
<br><br>

<b>🚀 Optimistic:</b>
{money_full(optimistic_profit_total)}
profit at {percentage(optimistic_margin)} margin.
<br><br>

<b>📊 Base:</b>
{money_full(base_profit_total)}
profit at {percentage(base_margin)} margin.
<br><br>

<b>⚠️ Downside:</b>
{money_full(downside_profit_total)}
profit at {percentage(downside_margin)} margin.
<br><br>

<b>Recommended Action:</b>
Use the Base case for operating planning,
the Optimistic case for upside capacity planning,
and the Downside case for liquidity and cost-control
contingency planning.
</div>""",
            unsafe_allow_html=True
        )

        st.info(
            "Forecast note: this is a portfolio prototype "
            "using a transparent statistical trend model. "
            "Scenario assumptions are illustrative and are "
            "not a production financial forecast."
        )


# ============================================================
# AI CFO
# ============================================================

elif selected_page == "🤖 AI CFO":

    st.title(
        "🤖 AI CFO"
    )

    st.caption(
        "Evidence-based financial decision support "
        "for management questions."
    )

    st.markdown(
        """<div class="insight-box">
<b>Ask FinSight AI a finance question.</b>
<br><br>
Examples:
<br>
• What is driving the cost overrun?
<br>
• Which business unit needs attention?
<br>
• What should management review first?
<br>
• Where is the largest unfavorable variance?
</div>""",
        unsafe_allow_html=True
    )

    question = st.text_input(
        "Management Question",
        placeholder=(
            "e.g. What is driving the cost overrun?"
        )
    )

    if st.button(
        "Run AI CFO Analysis",
        type="primary"
    ):

        insights = (
            generate_ai_cfo_insights(
                filtered_df
            )
        )

        if insights.empty:

            st.success(
                "No material financial risk "
                "was identified."
            )

        else:

            question_lower = (
                question.lower()
            )

            selected = pd.DataFrame()

            # Cost questions.
            if any(
                word in question_lower
                for word in [
                    "cost",
                    "overrun",
                    "expense",
                    "spend",
                    "variance"
                ]
            ):

                selected = insights[
                    insights[
                        "Finding"
                    ].str.contains(
                        "cost|variance|overrun",
                        case=False,
                        regex=True
                    )
                ]

            # BU questions.
            if selected.empty:

                if any(
                    word in question_lower
                    for word in [
                        "business unit",
                        "bu",
                        "region"
                    ]
                ):

                    selected = insights[
                        insights[
                            "Finding"
                        ].str.contains(
                            "business unit",
                            case=False
                        )
                    ]

            # Default.
            if selected.empty:

                selected = insights

            row = selected.iloc[0]

            st.subheader(
                "AI CFO Recommendation"
            )

            st.markdown(
                f"""<div class="insight-box">

<h4>Finding</h4>
{row["Finding"]}

<h4>Evidence</h4>
{row["Evidence"]}

<h4>Business Impact</h4>
{row["Business Impact"]}

<h4>Management Action</h4>
{row["Management Action"]}

<h4>Priority</h4>
{row["Priority"]}

<h4>Owner</h4>
{row["Owner"]}

</div>""",
                unsafe_allow_html=True
            )

            st.subheader(
                "Decision Queue"
            )

            st.dataframe(
                insights,
                use_container_width=True,
                hide_index=True
            )

    else:

        st.info(
            "Enter a management question and "
            "run the AI CFO analysis."
        )


# ============================================================
# ACTION CENTER
# ============================================================

elif selected_page == "🎯 Action Center":

    st.title(
        "🎯 Management Action Center"
    )

    st.caption(
        "Prioritized financial actions generated "
        "from variance and cost-driver analysis."
    )

    actions = build_action_center(
        filtered_df
    )

    if actions.empty:

        st.success(
            "No open financial actions were generated."
        )

    else:

        p0 = len(
            actions[
                actions["Priority"] == "P0"
            ]
        )

        p1 = len(
            actions[
                actions["Priority"] == "P1"
            ]
        )

        p2 = len(
            actions[
                actions["Priority"] == "P2"
            ]
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "P0 Critical",
            p0
        )

        col2.metric(
            "P1 High",
            p1
        )

        col3.metric(
            "P2 Monitor",
            p2
        )

        st.divider()

        for _, row in actions.iterrows():

            st.markdown(
                f"""<div class="action-box">

<b>{row["Priority"]} — {row["Area"]}</b>

<br><br>

<b>Issue:</b>
{row["Issue"]}

<br>

<b>Financial Impact:</b>
{row["Financial Impact"]}

<br>

<b>Recommended Action:</b>
{row["Recommended Action"]}

<br>

<b>Owner:</b>
{row["Owner"]}

<br>

<b>Status:</b>
{row["Status"]}

</div>""",
                unsafe_allow_html=True
            )

        st.subheader(
            "Action Queue"
        )

        st.dataframe(
            actions,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# DATA EXPLORER
# ============================================================

elif selected_page == "🔎 Data Explorer":

    st.title(
        "🔎 Data Explorer"
    )

    st.caption(
        "Explore the underlying ERP-style "
        "financial dataset."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Records",
        f"{len(filtered_df):,}"
    )

    col2.metric(
        "Columns",
        f"{len(filtered_df.columns):,}"
    )

    col3.metric(
        "Business Units",
        f"{filtered_df['Business Unit'].nunique():,}"
    )

    st.divider()

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = (
        filtered_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "⬇️ Download Filtered CSV",
        data=csv_data,
        file_name=(
            "finsight_filtered_financials.csv"
        ),
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """<div style="text-align:center;color:#64748b;padding:1rem;">
<b>FinSight AI</b> · Agentic FP&A & ERP Intelligence
<br>
<span style="font-size:0.8rem;">
Portfolio prototype using synthetic ERP-style financial data
</span>
</div>""",
    unsafe_allow_html=True
)
