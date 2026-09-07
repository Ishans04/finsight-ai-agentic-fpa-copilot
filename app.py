import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
        padding: 1.5rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #1e3a5f 52%,
            #0f766e 100%
        );
        color: white;
        margin-bottom: 1.3rem;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.12);
    }

    .hero h1 {
        color: white;
        font-size: 2.3rem;
        margin-bottom: 0.2rem;
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
        margin-bottom: 0.8rem;
    }

    .action-box {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        background: white;
        border: 1px solid #e2e8f0;
        margin-bottom: 0.8rem;
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

# IMPORTANT:
# Department is intentionally NOT mandatory.
# Some ERP extracts do not contain department-level data.

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
    """Compact INR formatting."""

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
    """Full INR formatting."""

    if value is None or pd.isna(value):
        return "₹0"

    return f"₹{float(value):,.0f}"


def percentage(value):
    """Percentage formatting."""

    if value is None or pd.isna(value):
        return "0.00%"

    return f"{float(value):,.2f}%"


def safe_divide(a, b):
    """Safe division."""

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
    """
    Convert a raw column name into a normalized comparison key.

    Examples:

    business_unit
    Business Unit
    BUSINESS UNIT

    all become:

    business unit
    """

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

    text = " ".join(text.split())

    return text


def normalize_columns(df):
    """
    Map common ERP / finance column names to FinSight's
    canonical schema.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Canonical name -> possible aliases
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Build normalized source lookup
    # --------------------------------------------------------

    source_lookup = {}

    for column in df.columns:

        key = clean_column_name(column)

        if key not in source_lookup:
            source_lookup[key] = column

    rename_map = {}

    # --------------------------------------------------------
    # Match aliases
    # --------------------------------------------------------

    for canonical, possible_names in aliases.items():

        # First try canonical itself
        canonical_key = clean_column_name(canonical)

        if canonical_key in source_lookup:

            original = source_lookup[canonical_key]

            if original != canonical:
                rename_map[original] = canonical

            continue

        # Then aliases
        for alias in possible_names:

            alias_key = clean_column_name(alias)

            if alias_key in source_lookup:

                original = source_lookup[alias_key]

                if original != canonical:
                    rename_map[original] = canonical

                break

    df = df.rename(columns=rename_map)

    return df


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(df):
    """
    Prepare an ERP dataset for FinSight.

    Department is optional.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Normalize columns
    # --------------------------------------------------------

    df = normalize_columns(df)

    # --------------------------------------------------------
    # Department is OPTIONAL
    # --------------------------------------------------------

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
            df["Department"].isin(["", "nan", "None"]),
            "Department"
        ] = "Unmapped"

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:

        detected = ", ".join(
            [str(column) for column in df.columns]
        )

        return None, (
            "Missing required columns: "
            + ", ".join(missing)
            + ". Detected columns: "
            + detected
        )

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Required numeric fields
    # --------------------------------------------------------

    for column in [
        "Budget",
        "Actual",
        "Revenue",
    ]:

        df[column] = (
            df[column]
            .fillna(0)
        )

    # --------------------------------------------------------
    # Derived Variance
    # --------------------------------------------------------

    if "Variance" not in df.columns:

        df["Variance"] = (
            df["Actual"]
            - df["Budget"]
        )

    else:

        df["Variance"] = (
            df["Variance"]
            .fillna(
                df["Actual"] - df["Budget"]
            )
        )

    # --------------------------------------------------------
    # Derived Variance %
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Derived Profit
    # --------------------------------------------------------

    if "Profit" not in df.columns:

        df["Profit"] = (
            df["Revenue"]
            - df["Actual"]
        )

    # --------------------------------------------------------
    # Anomaly fields
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Transaction ID
    # --------------------------------------------------------

    if "Transaction ID" not in df.columns:

        df["Transaction ID"] = [
            f"TXN-{number:05d}"
            for number in range(
                1,
                len(df) + 1
            )
        ]

    # --------------------------------------------------------
    # Business Unit
    # --------------------------------------------------------

    df["Business Unit"] = (
        df["Business Unit"]
        .fillna("Unmapped")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Account / Cost Category
    # --------------------------------------------------------

    df["Account / Cost Category"] = (
        df["Account / Cost Category"]
        .fillna("Unmapped")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Remove invalid dates
    # --------------------------------------------------------

    df = df.dropna(
        subset=["Date"]
    ).copy()

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    return df, None


# ============================================================
# LOAD DEMO DATA
# ============================================================

@st.cache_data
def load_demo_data():

    paths = [
        Path("synthetic_erp_financials.csv"),
        Path("data/synthetic_erp_financials.csv"),
    ]

    selected_path = None

    for path in paths:

        if path.exists():

            selected_path = path
            break

    if selected_path is None:

        return None, (
            "FinSight demo dataset was not found. "
            "Expected synthetic_erp_financials.csv "
            "in the repository."
        )

    try:

        raw_df = pd.read_csv(
            selected_path
        )

        return prepare_data(
            raw_df
        )

    except Exception as exc:

        return None, (
            "Unable to load demo data: "
            + str(exc)
        )


# ============================================================
# LOAD UPLOADED DATA
# ============================================================

@st.cache_data
def load_uploaded_data(file_bytes):

    try:

        raw_df = pd.read_csv(
            BytesIO(file_bytes)
        )

        return prepare_data(
            raw_df
        )

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

    variance = (
        actual
        - budget
    )

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
# MONTHLY ANALYSIS
# ============================================================

def monthly_summary(df):

    result = (
        df.assign(
            Month=df["Date"]
            .dt
            .to_period("M")
            .astype(str)
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
# BUSINESS UNIT ANALYSIS
# ============================================================

def business_unit_summary(df):

    result = (
        df.groupby("Business Unit")
        .agg(
            Revenue=("Revenue", "sum"),
            Budget=("Budget", "sum"),
            Actual=("Actual", "sum"),
            Profit=("Profit", "sum"),
            Transactions=("Transaction ID", "count"),
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
# DEPARTMENT ANALYSIS
# ============================================================

def department_summary(df):

    result = (
        df.groupby("Department")
        .agg(
            Revenue=("Revenue", "sum"),
            Budget=("Budget", "sum"),
            Actual=("Actual", "sum"),
            Profit=("Profit", "sum"),
            Transactions=("Transaction ID", "count"),
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
# COST CATEGORY ANALYSIS
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
            Transactions=("Transaction ID", "count"),
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

        prediction = model.fit_predict(
            features
        )

        result["Anomaly Flag"] = np.where(
            prediction == -1,
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

    category = category_summary(df)

    business_units = business_unit_summary(df)

    # --------------------------------------------------------
    # TOP COST DRIVER
    # --------------------------------------------------------

    unfavorable_categories = category[
        category["Variance"] > 0
    ]

    if not unfavorable_categories.empty:

        top = unfavorable_categories.iloc[0]

        insights.append({

            "Finding": (
                f"{top['Account / Cost Category']} "
                f"is the largest unfavorable cost driver."
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
    # BUSINESS UNIT RISK
    # --------------------------------------------------------

    unfavorable_bu = business_units[
        business_units["Variance"] > 0
    ]

    if not unfavorable_bu.empty:

        top_bu = unfavorable_bu.iloc[0]

        insights.append({

            "Finding": (
                f"{top_bu['Business Unit']} "
                f"has the highest unfavorable "
                f"budget variance."
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
    # MARGIN RISK
    # --------------------------------------------------------

    if kpis["margin"] < 50:

        insights.append({

            "Finding": (
                "Operating margin is below the "
                "management threshold."
            ),

            "Evidence": (
                f"Current modeled margin is "
                f"{percentage(kpis['margin'])}."
            ),

            "Business Impact": (
                "Lower margin can reduce profitability "
                "and investment capacity."
            ),

            "Management Action": (
                "Review pricing, revenue mix, "
                "major cost drivers and resource allocation."
            ),

            "Priority": "P1",

            "Owner": "FP&A / Leadership",
        })

    # --------------------------------------------------------
    # ANOMALY RISK
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

    return pd.DataFrame(
        insights
    )


# ============================================================
# CASH FLOW MODEL
# ============================================================

def build_cash_flow(df):

    monthly = monthly_summary(
        df
    ).copy()

    if monthly.empty:

        return monthly

    # --------------------------------------------------------
    # Modeled cash assumptions
    # --------------------------------------------------------

    monthly["Cash Inflow"] = (
        monthly["Revenue"]
        * 0.85
    )

    monthly["Cash Outflow"] = (
        monthly["Actual"]
        * 0.80
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

    monthly["Opening Cash"] = (
        opening_values
    )

    monthly["Closing Cash"] = (
        closing_values
    )

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
        len(monthly) + periods,
    )

    # --------------------------------------------------------
    # Revenue model
    # --------------------------------------------------------

    revenue_model = LinearRegression()

    revenue_model.fit(
        monthly[["Period Number"]],
        monthly["Revenue"],
    )

    future_frame = pd.DataFrame({

        "Period Number":
        future_periods

    })

    revenue_forecast = (
        revenue_model.predict(
            future_frame
        )
    )

    # --------------------------------------------------------
    # Cost model
    # --------------------------------------------------------

    cost_model = LinearRegression()

    cost_model.fit(
        monthly[["Period Number"]],
        monthly["Actual"],
    )

    cost_forecast = (
        cost_model.predict(
            future_frame
        )
    )

    # Prevent negative forecasts
    revenue_forecast = np.maximum(
        revenue_forecast,
        0
    )

    cost_forecast = np.maximum(
        cost_forecast,
        0
    )

    # --------------------------------------------------------
    # Future dates
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
        freq="MS",
    )

    forecast = pd.DataFrame({

        "Month":
        future_dates.strftime("%Y-%m"),

        "Base Revenue":
        revenue_forecast,

        "Base Cost":
        cost_forecast,
    })

    forecast["Base Profit"] = (
        forecast["Base Revenue"]
        - forecast["Base Cost"]
    )

    # --------------------------------------------------------
    # Optimistic scenario
    # --------------------------------------------------------

    forecast["Optimistic Revenue"] = (
        forecast["Base Revenue"]
        * 1.08
    )

    forecast["Optimistic Cost"] = (
        forecast["Base Cost"]
        * 0.97
    )

    forecast["Optimistic Profit"] = (
        forecast["Optimistic Revenue"]
        - forecast["Optimistic Cost"]
    )

    # --------------------------------------------------------
    # Downside scenario
    # --------------------------------------------------------

    forecast["Downside Revenue"] = (
        forecast["Base Revenue"]
        * 0.92
    )

    forecast["Downside Cost"] = (
        forecast["Base Cost"]
        * 1.08
    )

    forecast["Downside Profit"] = (
        forecast["Downside Revenue"]
        - forecast["Downside Cost"]
    )

    return forecast


# ============================================================
# ACTION CENTER
# ============================================================

def build_action_center(df):

    actions = []

    categories = category_summary(
        df
    )

    for _, row in categories.head(10).iterrows():

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

            "Area":
            row["Account / Cost Category"],

            "Issue":
            "Unfavorable cost variance",

            "Financial Impact":
            money_full(variance),

            "Recommended Action":
            (
                "Review spend against budget, "
                "identify controllable costs and "
                "refresh the forecast."
            ),

            "Owner":
            "FP&A / Cost Owner",

            "Status":
            "Open",
        })

    return pd.DataFrame(
        actions
    )


# ============================================================
# LOAD DATA
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <h2>📊 FinSight AI</h2>
        <div class="small-muted">
        Agentic FP&A & ERP Intelligence
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Data Source")

    uploaded_file = st.file_uploader(
        "Upload ERP CSV",
        type=["csv"],
        help="Upload an ERP-style financial CSV.",
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
# STOP ONLY IF DATA REALLY CANNOT LOAD
# ============================================================

if df is None:

    st.error(
        data_error
    )

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

        FinSight recognizes common ERP variations such as:

        `business_unit`

        `department`

        `account`

        `cost_category`

        `gl_account`

        `budget`

        `actual`

        `revenue`

        `variance`

        `variance_pct`
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
    """
    <div class="hero">

        <h1>📊 FinSight AI</h1>

        <p>
            Agentic FP&A & ERP Intelligence Command Center
            — turning financial data into management decisions.
        </p>

    </div>
    """,
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
        money(kpis["revenue"]),
    )

    col2.metric(
        "Actual Cost",
        money(kpis["actual"]),
    )

    col3.metric(
        "Budget",
        money(kpis["budget"]),
    )

    col4.metric(
        "Variance",
        money(kpis["variance"]),
    )

    col5.metric(
        "Profit Margin",
        percentage(kpis["margin"]),
    )

    st.divider()

    left, right = st.columns(2)

    monthly = monthly_summary(
        filtered_df
    )

    # --------------------------------------------------------
    # Revenue vs Cost
    # --------------------------------------------------------

    with left:

        st.subheader(
            "Revenue vs Actual Cost"
        )

        if not monthly.empty:

            chart_data = monthly.melt(
                id_vars=["Month"],
                value_vars=[
                    "Revenue",
                    "Actual",
                ],
                var_name="Metric",
                value_name="Amount",
            )

            fig = px.line(
                chart_data,
                x="Month",
                y="Amount",
                color="Metric",
                markers=True,
            )

            fig.update_layout(
                height=380,
                xaxis_title="Month",
                yaxis_title="Amount",
                legend_title="",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    # --------------------------------------------------------
    # BU Variance
    # --------------------------------------------------------

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
                y="Variance",
                title="Actual Cost vs Budget",
            )

            fig.add_hline(
                y=0,
                line_dash="dash",
            )

            fig.update_layout(
                height=380
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    # --------------------------------------------------------
    # AI CFO snapshot
    # --------------------------------------------------------

    st.subheader(
        "🤖 AI CFO Executive Snapshot"
    )

    insights = generate_ai_cfo_insights(
        filtered_df
    )

    if insights.empty:

        st.success(
            "No material management risks were "
            "identified in the current dataset."
        )

    else:

        for _, row in insights.head(3).iterrows():

            st.markdown(
                f"""
                <div class="insight-box">

                    <b>
                    {row["Priority"]} —
                    {row["Finding"]}
                    </b>

                    <br><br>

                    <b>Evidence:</b>
                    {row["Evidence"]}

                    <br><br>

                    <b>Management Action:</b>
                    {row["Management Action"]}

                </div>
                """,
                unsafe_allow_html=True,
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
            "Cost Categories",
        ]
    )

    # --------------------------------------------------------
    # BU
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
            "Variance",
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
            hide_index=True,
        )

    # --------------------------------------------------------
    # Departments
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
            "Variance",
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
            hide_index=True,
        )

    # --------------------------------------------------------
    # Cost categories
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
            "Variance",
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
            hide_index=True,
        )

    # --------------------------------------------------------
    # Monthly
    # --------------------------------------------------------

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
        "Variance",
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
        hide_index=True,
    )


# ============================================================
# RISK & ANOMALIES
# ============================================================

elif selected_page == "⚠️ Risk & Anomalies":

    st.title(
        "Risk & Anomalies"
    )

    st.caption(
        "Machine-learning anomaly detection and "
        "financial risk monitoring."
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

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Transactions",
        f"{total_transactions:,}",
    )

    c2.metric(
        "Potential Anomalies",
        f"{anomaly_count:,}",
    )

    c3.metric(
        "Anomaly Rate",
        percentage(anomaly_rate),
    )

    st.divider()

    left, right = st.columns(2)

    # --------------------------------------------------------
    # Anomaly by BU
    # --------------------------------------------------------

    with left:

        st.subheader(
            "Anomalies by Business Unit"
        )

        anomaly_bu = (
            anomaly_df[
                anomaly_df["Anomaly Flag"] == 1
            ]
            .groupby("Business Unit")
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
                y="Anomalies",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    # --------------------------------------------------------
    # Anomaly by category
    # --------------------------------------------------------

    with right:

        st.subheader(
            "Anomalies by Cost Category"
        )

        anomaly_category = (
            anomaly_df[
                anomaly_df["Anomaly Flag"] == 1
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
                y="Anomalies",
            )

            fig.update_layout(
                xaxis_tickangle=-35
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    # --------------------------------------------------------
    # Flagged transactions
    # --------------------------------------------------------

    st.subheader(
        "Flagged Transactions"
    )

    flagged = anomaly_df[
        anomaly_df["Anomaly Flag"] == 1
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
            "Anomaly",
        ]

        display_columns = [
            column
            for column in columns
            if column in flagged.columns
        ]

        display = flagged[
            display_columns
        ].copy()

        for column in [
            "Budget",
            "Actual",
            "Variance",
        ]:

            if column in display.columns:

                display[column] = (
                    display[column]
                    .map(money_full)
                )

        if "Variance %" in display.columns:

            display["Variance %"] = (
                flagged["Variance %"]
                .map(percentage)
            )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
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
            "Insufficient financial history."
        )

    else:

        latest = cash.iloc[-1]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Cash Inflow",
            money(latest["Cash Inflow"]),
        )

        c2.metric(
            "Cash Outflow",
            money(latest["Cash Outflow"]),
        )

        c3.metric(
            "Net Cash Flow",
            money(latest["Net Cash Flow"]),
        )

        c4.metric(
            "Closing Cash",
            money(latest["Closing Cash"]),
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
                    "Cash Outflow",
                ],
                var_name="Metric",
                value_name="Amount",
            )

            fig = px.bar(
                chart_data,
                x="Month",
                y="Amount",
                color="Metric",
                barmode="group",
            )

            fig.update_layout(
                height=380
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with right:

            st.subheader(
                "Closing Cash Trend"
            )

            fig = px.line(
                cash,
                x="Month",
                y="Closing Cash",
                markers=True,
            )

            fig.update_layout(
                height=380
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.subheader(
            "Liquidity Status"
        )

        status = latest[
            "Liquidity Status"
        ]

        if status == "Healthy":

            st.success(
                "Liquidity status: Healthy"
            )

        elif status == "Watch":

            st.warning(
                "Liquidity status: Watch"
            )

        else:

            st.error(
                "Liquidity status: Critical"
            )

        display = cash.copy()

        for column in [
            "Cash Inflow",
            "Cash Outflow",
            "Net Cash Flow",
            "Opening Cash",
            "Closing Cash",
        ]:

            display[column] = (
                display[column]
                .map(money_full)
            )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "Prototype note: cash flow is modeled from "
            "the synthetic ERP dataset. It is not a live "
            "bank or treasury integration."
        )


# ============================================================
# FORECAST & SCENARIOS
# ============================================================

elif selected_page == "📈 Forecast & Scenarios":

    st.title(
        "Forecast & Scenario Planner"
    )

    st.caption(
        "Forward-looking revenue and cost scenarios "
        "for FP&A planning."
    )

    forecast = build_forecast(
        filtered_df,
        periods=6,
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
        # Historical growth
        # ----------------------------------------------------

        if len(historical) >= 2:

            first_revenue = (
                historical["Revenue"]
                .iloc[0]
            )

            last_revenue = (
                historical["Revenue"]
                .iloc[-1]
            )

            periods = max(
                len(historical) - 1,
                1,
            )

            historical_growth = (

                safe_divide(
                    last_revenue,
                    first_revenue
                )
                ** (1 / periods)
                - 1

            ) * 100

        else:

            historical_growth = 0

        historical_margin = (
            safe_divide(
                historical["Profit"].sum(),
                historical["Revenue"].sum(),
            )
            * 100
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Historical Avg Growth",
            percentage(
                historical_growth
            ),
        )

        c2.metric(
            "Historical Margin",
            percentage(
                historical_margin
            ),
        )

        c3.metric(
            "Forecast Horizon",
            "6 Months",
        )

        st.divider()

        scenario = st.selectbox(
            "Scenario",
            [
                "Base",
                "Optimistic",
                "Downside",
            ],
        )

        if scenario == "Base":

            revenue_column = (
                "Base Revenue"
            )

            cost_column = (
                "Base Cost"
            )

            profit_column = (
                "Base Profit"
            )

        elif scenario == "Optimistic":

            revenue_column = (
                "Optimistic Revenue"
            )

            cost_column = (
                "Optimistic Cost"
            )

            profit_column = (
                "Optimistic Profit"
            )

        else:

            revenue_column = (
                "Downside Revenue"
            )

            cost_column = (
                "Downside Cost"
            )

            profit_column = (
                "Downside Profit"
            )

        chart_data = forecast[
            [
                "Month",
                revenue_column,
                cost_column,
            ]
        ].melt(
            id_vars=["Month"],
            var_name="Metric",
            value_name="Amount",
        )

        fig = px.line(
            chart_data,
            x="Month",
            y="Amount",
            color="Metric",
            markers=True,
        )

        fig.update_layout(
            height=420
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.subheader(
            f"{scenario} Scenario Forecast"
        )

        display = forecast[
            [
                "Month",
                revenue_column,
                cost_column,
                profit_column,
            ]
        ].copy()

        display.columns = [
            "Month",
            "Revenue",
            "Cost",
            "Profit",
        ]

        for column in [
            "Revenue",
            "Cost",
            "Profit",
        ]:

            display[column] = (
                display[column]
                .map(money_full)
            )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "Forecast uses a transparent statistical "
            "trend model with scenario assumptions. "
            "It is intended for portfolio demonstration "
            "and decision support."
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
        """
        <div class="insight-box">

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

        </div>
        """,
        unsafe_allow_html=True,
    )

    question = st.text_input(
        "Management Question",
        placeholder=(
            "e.g. What is driving the cost overrun?"
        ),
    )

    if st.button(
        "Run AI CFO Analysis",
        type="primary",
    ):

        insights = generate_ai_cfo_insights(
            filtered_df
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

            selected = None

            # ------------------------------------------------
            # Cost questions
            # ------------------------------------------------

            if any(
                word in question_lower
                for word in [
                    "cost",
                    "overrun",
                    "expense",
                    "spend",
                    "variance",
                ]
            ):

                selected = insights[
                    insights["Finding"]
                    .str.contains(
                        "cost|variance|overrun",
                        case=False,
                        regex=True,
                    )
                ]

            # ------------------------------------------------
            # BU questions
            # ------------------------------------------------

            if (
                selected is None
                or selected.empty
            ):

                if any(
                    word in question_lower
                    for word in [
                        "business unit",
                        "bu",
                        "region",
                    ]
                ):

                    selected = insights[
                        insights["Finding"]
                        .str.contains(
                            "business unit",
                            case=False,
                        )
                    ]

            # ------------------------------------------------
            # Default
            # ------------------------------------------------

            if (
                selected is None
                or selected.empty
            ):

                selected = insights

            row = selected.iloc[0]

            st.subheader(
                "AI CFO Recommendation"
            )

            st.markdown(
                f"""
                <div class="insight-box">

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

                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader(
                "Decision Queue"
            )

            st.dataframe(
                insights,
                use_container_width=True,
                hide_index=True,
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

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "P0 Critical",
            p0,
        )

        c2.metric(
            "P1 High",
            p1,
        )

        c3.metric(
            "P2 Monitor",
            p2,
        )

        st.divider()

        for _, row in actions.iterrows():

            st.markdown(
                f"""
                <div class="action-box">

                <b>
                {row["Priority"]} —
                {row["Area"]}
                </b>

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

                </div>
                """,
                unsafe_allow_html=True,
            )

        st.subheader(
            "Action Queue"
        )

        st.dataframe(
            actions,
            use_container_width=True,
            hide_index=True,
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

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Records",
        f"{len(filtered_df):,}",
    )

    c2.metric(
        "Columns",
        f"{len(filtered_df.columns):,}",
    )

    c3.metric(
        "Business Units",
        f"{filtered_df['Business Unit'].nunique():,}",
    )

    st.divider()

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
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
        mime="text/csv",
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div
        style="
        text-align:center;
        color:#64748b;
        padding:1rem;
        "
    >

        <b>FinSight AI</b>
        · Agentic FP&A & ERP Intelligence

        <br>

        <span style="font-size:0.8rem;">
        Portfolio prototype using synthetic ERP-style
        financial data
        </span>

    </div>
    """,
    unsafe_allow_html=True,
)
