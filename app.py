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
            background-color: #f7f9fc;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        .hero {
            padding: 1.4rem 1.6rem;
            border-radius: 16px;
            background: linear-gradient(
                135deg,
                #0f172a 0%,
                #1e3a5f 50%,
                #0f766e 100%
            );
            color: white;
            margin-bottom: 1.2rem;
        }

        .hero h1 {
            color: white;
            margin-bottom: 0.2rem;
            font-size: 2.2rem;
        }

        .hero p {
            color: #dbeafe;
            margin-bottom: 0;
            font-size: 1rem;
        }

        .section-card {
            padding: 1rem 1.2rem;
            border-radius: 12px;
            background: white;
            border: 1px solid #e5e7eb;
            margin-bottom: 1rem;
        }

        .insight-box {
            padding: 1rem;
            border-radius: 12px;
            background: #eff6ff;
            border-left: 5px solid #2563eb;
            margin-bottom: 0.8rem;
        }

        .action-box {
            padding: 1rem;
            border-radius: 12px;
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            margin-bottom: 0.8rem;
        }

        .priority-p0 {
            border-left: 5px solid #dc2626;
        }

        .priority-p1 {
            border-left: 5px solid #f59e0b;
        }

        .priority-p2 {
            border-left: 5px solid #16a34a;
        }

        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #e5e7eb;
            padding: 0.8rem;
            border-radius: 12px;
        }

        .small-muted {
            color: #64748b;
            font-size: 0.85rem;
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
    "Department",
    "Account / Cost Category",
    "Budget",
    "Actual",
    "Revenue",
]

DISPLAY_COLUMNS = [
    "Transaction ID",
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
    "Anomaly Flag",
    "Anomaly",
]


# ============================================================
# FORMATTING FUNCTIONS
# ============================================================

def money(value):
    """Compact Indian currency formatting."""
    if pd.isna(value):
        return "₹0"

    value = float(value)
    abs_value = abs(value)

    if abs_value >= 1e7:
        return f"₹{value / 1e7:,.2f} Cr"

    if abs_value >= 1e5:
        return f"₹{value / 1e5:,.2f} L"

    if abs_value >= 1e3:
        return f"₹{value / 1e3:,.1f} K"

    return f"₹{value:,.0f}"


def money_full(value):
    """Full Indian currency formatting."""
    if pd.isna(value):
        return "₹0"

    return f"₹{float(value):,.0f}"


def pct(value):
    """Percentage formatting."""
    if pd.isna(value):
        return "0.00%"

    return f"{float(value):,.2f}%"


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0

    return numerator / denominator


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def clean_column_name(column):
    """
    Normalize a column name so variations such as:
    account
    Account
    ACCOUNT
    Account / Cost Category
    GL Account
    Cost Category
    can be detected reliably.
    """

    text = str(column).strip().lower()

    replacements = {
        "&": "and",
        "/": " ",
        "-": " ",
        "_": " ",
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
    Convert common ERP / finance column names into FinSight's
    internal canonical schema.
    """

    alias_map = {
        "Transaction ID": [
            "transaction id",
            "transaction",
            "transaction number",
            "transaction no",
            "transaction_id",
            "id",
        ],

        "Date": [
            "date",
            "transaction date",
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
            "bu",
            "businessunit",
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
        ],

        "Account / Cost Category": [
            "account",
            "account name",
            "account_name",
            "account cost category",
            "account / cost category",
            "cost category",
            "cost_category",
            "cost center category",
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
            "sales",
            "sales revenue",
            "income",
            "revenue amount",
            "revenue_amount",
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
            "variance percentage %",
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

    normalized_lookup = {}

    for original in df.columns:
        normalized_lookup[clean_column_name(original)] = original

    rename_dict = {}

    for canonical, aliases in alias_map.items():
        canonical_key = clean_column_name(canonical)

        if canonical_key in normalized_lookup:
            rename_dict[normalized_lookup[canonical_key]] = canonical
            continue

        for alias in aliases:
            alias_key = clean_column_name(alias)

            if alias_key in normalized_lookup:
                rename_dict[normalized_lookup[alias_key]] = canonical
                break

    df = df.rename(columns=rename_dict)

    return df


# ============================================================
# DATA ENGINE
# ============================================================

@st.cache_data
def load_demo_data():
    """
    Load FinSight's bundled synthetic ERP dataset.
    """

    possible_paths = [
        Path("synthetic_erp_financials.csv"),
        Path("data/synthetic_erp_financials.csv"),
    ]

    file_path = None

    for path in possible_paths:
        if path.exists():
            file_path = path
            break

    if file_path is None:
        return None, "Demo dataset not found."

    try:
        df = pd.read_csv(file_path)

        df = normalize_columns(df)

        return prepare_data(df)

    except Exception as exc:
        return None, f"Unable to load demo dataset: {exc}"


@st.cache_data
def load_uploaded_data(file_bytes):
    """
    Load uploaded CSV.
    """

    try:
        from io import BytesIO

        df = pd.read_csv(BytesIO(file_bytes))

        df = normalize_columns(df)

        return prepare_data(df)

    except Exception as exc:
        return None, f"Unable to read uploaded CSV: {exc}"


def prepare_data(df):
    """
    Validate and prepare the dataset for all FinSight modules.
    """

    df = df.copy()

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        return None, (
            "Missing required columns: "
            + ", ".join(missing)
            + ". "
            + "Detected columns: "
            + ", ".join(map(str, df.columns))
        )

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Numeric fields
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
    # Derived financial metrics
    # --------------------------------------------------------

    if "Variance" not in df.columns:
        df["Variance"] = df["Actual"] - df["Budget"]

    if "Variance %" not in df.columns:
        df["Variance %"] = np.where(
            df["Budget"].abs() > 0,
            (df["Actual"] - df["Budget"]) / df["Budget"].abs() * 100,
            0,
        )

    if "Profit" not in df.columns:
        df["Profit"] = df["Revenue"] - df["Actual"]

    if "Anomaly Flag" not in df.columns:
        df["Anomaly Flag"] = 0

    if "Anomaly" not in df.columns:
        df["Anomaly"] = "Normal"

    # --------------------------------------------------------
    # Text fields
    # --------------------------------------------------------

    text_columns = [
        "Business Unit",
        "Department",
        "Account / Cost Category",
    ]

    for column in text_columns:
        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
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
    # Remove unusable dates
    # --------------------------------------------------------

    df = df.dropna(subset=["Date"]).copy()

    df = df.sort_values("Date").reset_index(drop=True)

    return df, None


# ============================================================
# ANALYTICS
# ============================================================

def calculate_kpis(df):

    revenue = df["Revenue"].sum()
    actual = df["Actual"].sum()
    budget = df["Budget"].sum()
    variance = actual - budget
    profit = df["Profit"].sum()

    margin = safe_divide(
        profit,
        revenue
    ) * 100

    return {
        "revenue": revenue,
        "actual": actual,
        "budget": budget,
        "variance": variance,
        "profit": profit,
        "margin": margin,
    }


def monthly_summary(df):

    monthly = (
        df.assign(
            Month=df["Date"].dt.to_period("M").astype(str)
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

    monthly["Variance"] = (
        monthly["Actual"] - monthly["Budget"]
    )

    monthly["Margin %"] = np.where(
        monthly["Revenue"] != 0,
        monthly["Profit"] / monthly["Revenue"] * 100,
        0,
    )

    return monthly


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
        result["Actual"] - result["Budget"]
    )

    result["Variance %"] = np.where(
        result["Budget"] != 0,
        result["Variance"] / result["Budget"].abs() * 100,
        0,
    )

    result["Margin %"] = np.where(
        result["Revenue"] != 0,
        result["Profit"] / result["Revenue"] * 100,
        0,
    )

    return result.sort_values(
        "Variance",
        ascending=False
    )


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
        result["Actual"] - result["Budget"]
    )

    result["Variance %"] = np.where(
        result["Budget"] != 0,
        result["Variance"] / result["Budget"].abs() * 100,
        0,
    )

    return result.sort_values(
        "Variance",
        ascending=False
    )


def category_summary(df):

    result = (
        df.groupby("Account / Cost Category")
        .agg(
            Budget=("Budget", "sum"),
            Actual=("Actual", "sum"),
            Revenue=("Revenue", "sum"),
            Transactions=("Transaction ID", "count"),
        )
        .reset_index()
    )

    result["Variance"] = (
        result["Actual"] - result["Budget"]
    )

    result["Variance %"] = np.where(
        result["Budget"] != 0,
        result["Variance"] / result["Budget"].abs() * 100,
        0,
    )

    return result.sort_values(
        "Variance",
        ascending=False
    )


def detect_anomalies(df):

    result = df.copy()

    if len(result) < 10:
        result["Anomaly Flag"] = 0
        result["Anomaly"] = "Normal"
        return result

    features = result[
        ["Budget", "Actual", "Revenue", "Variance"]
    ].fillna(0)

    try:
        model = IsolationForest(
            contamination=0.05,
            random_state=42
        )

        predictions = model.fit_predict(features)

        result["Anomaly Flag"] = np.where(
            predictions == -1,
            1,
            0
        )

        result["Anomaly"] = np.where(
            result["Anomaly Flag"] == 1,
            "Potential Anomaly",
            "Normal"
        )

    except Exception:
        result["Anomaly Flag"] = 0
        result["Anomaly"] = "Normal"

    return result


# ============================================================
# AI CFO ENGINE
# ============================================================

def generate_ai_cfo_insights(df):

    insights = []

    kpis = calculate_kpis(df)

    bu = business_unit_summary(df)
    cat = category_summary(df)

    # --------------------------------------------------------
    # Cost variance
    # --------------------------------------------------------

    unfavorable = cat[cat["Variance"] > 0]

    if not unfavorable.empty:

        top_category = unfavorable.iloc[0]

        insights.append({
            "Finding": (
                f"{top_category['Account / Cost Category']} "
                f"is the largest unfavorable cost driver."
            ),
            "Evidence": (
                f"Variance of {money_full(top_category['Variance'])} "
                f"({pct(top_category['Variance %'])})."
            ),
            "Business Impact": (
                "Continued overspend may reduce operating margin "
                "and increase forecast pressure."
            ),
            "Management Action": (
                "Review vendor, utilization, staffing and discretionary "
                "spend within this cost category."
            ),
            "Priority": "P0",
            "Owner": "Finance / Business Owner",
        })

    # --------------------------------------------------------
    # Business unit
    # --------------------------------------------------------

    unfavorable_bu = bu[bu["Variance"] > 0]

    if not unfavorable_bu.empty:

        top_bu = unfavorable_bu.iloc[0]

        insights.append({
            "Finding": (
                f"{top_bu['Business Unit']} has the highest "
                "unfavorable budget variance."
            ),
            "Evidence": (
                f"Actual cost exceeds budget by "
                f"{money_full(top_bu['Variance'])}."
            ),
            "Business Impact": (
                "The business unit is creating disproportionate "
                "cost pressure versus plan."
            ),
            "Management Action": (
                "Perform a BU-level budget review and identify "
                "controllable versus committed costs."
            ),
            "Priority": "P1",
            "Owner": "BU Finance Partner",
        })

    # --------------------------------------------------------
    # Margin
    # --------------------------------------------------------

    if kpis["margin"] < 50:

        insights.append({
            "Finding": "Operating margin is below the 50% management threshold.",
            "Evidence": f"Current modeled margin is {pct(kpis['margin'])}.",
            "Business Impact": (
                "Lower margin can reduce profitability and "
                "limit investment capacity."
            ),
            "Management Action": (
                "Review pricing, revenue mix and major cost drivers."
            ),
            "Priority": "P1",
            "Owner": "FP&A / Business Leadership",
        })

    # --------------------------------------------------------
    # Anomalies
    # --------------------------------------------------------

    anomaly_count = int(
        df["Anomaly Flag"].sum()
    )

    if anomaly_count > 0:

        insights.append({
            "Finding": (
                f"{anomaly_count} transaction(s) require anomaly review."
            ),
            "Evidence": (
                "Isolation Forest / source anomaly indicators "
                "identified unusual financial behavior."
            ),
            "Business Impact": (
                "Unusual transactions can distort forecasts, "
                "budgets and management reporting."
            ),
            "Management Action": (
                "Investigate the flagged transactions and validate "
                "business justification."
            ),
            "Priority": "P1",
            "Owner": "Finance Controller",
        })

    return pd.DataFrame(insights)


# ============================================================
# CASH FLOW ENGINE
# ============================================================

def build_cash_flow(df):

    monthly = monthly_summary(df).copy()

    if monthly.empty:
        return monthly

    monthly["Cash Inflow"] = (
        monthly["Revenue"] * 0.85
    )

    monthly["Cash Outflow"] = (
        monthly["Actual"] * 0.80
    )

    opening_cash = 1_00_00_000

    closing_values = []
    opening_values = []

    current_cash = opening_cash

    for _, row in monthly.iterrows():

        opening_values.append(current_cash)

        current_cash = (
            current_cash
            + row["Cash Inflow"]
            - row["Cash Outflow"]
        )

        closing_values.append(current_cash)

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
            monthly["Closing Cash"] < opening_cash * 0.50,
            "Watch",
            "Healthy"
        )
    )

    return monthly


# ============================================================
# FORECAST ENGINE
# ============================================================

def build_forecast(df, periods=6):

    monthly = monthly_summary(df)

    if len(monthly) < 3:
        return None

    monthly = monthly.copy()

    monthly["Period Number"] = np.arange(
        len(monthly)
    )

    future_periods = np.arange(
        len(monthly),
        len(monthly) + periods
    )

    # Revenue model
    revenue_model = LinearRegression()

    revenue_model.fit(
        monthly[["Period Number"]],
        monthly["Revenue"]
    )

    revenue_forecast = revenue_model.predict(
        pd.DataFrame(
            {"Period Number": future_periods}
        )
    )

    # Cost model
    cost_model = LinearRegression()

    cost_model.fit(
        monthly[["Period Number"]],
        monthly["Actual"]
    )

    cost_forecast = cost_model.predict(
        pd.DataFrame(
            {"Period Number": future_periods}
        )
    )

    last_date = pd.to_datetime(
        monthly["Month"].iloc[-1]
    )

    future_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=periods,
        freq="MS"
    )

    forecast = pd.DataFrame({
        "Month": future_dates.strftime("%Y-%m"),
        "Base Revenue": revenue_forecast,
        "Base Cost": cost_forecast,
    })

    forecast["Base Profit"] = (
        forecast["Base Revenue"]
        - forecast["Base Cost"]
    )

    forecast["Optimistic Revenue"] = (
        forecast["Base Revenue"] * 1.08
    )

    forecast["Optimistic Cost"] = (
        forecast["Base Cost"] * 0.97
    )

    forecast["Optimistic Profit"] = (
        forecast["Optimistic Revenue"]
        - forecast["Optimistic Cost"]
    )

    forecast["Downside Revenue"] = (
        forecast["Base Revenue"] * 0.92
    )

    forecast["Downside Cost"] = (
        forecast["Base Cost"] * 1.08
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

    category = category_summary(df)

    for _, row in category.head(8).iterrows():

        variance = row["Variance"]

        if variance <= 0:
            continue

        if row["Variance %"] >= 15:
            priority = "P0"
        elif row["Variance %"] >= 8:
            priority = "P1"
        else:
            priority = "P2"

        actions.append({
            "Priority": priority,
            "Area": row["Account / Cost Category"],
            "Issue": "Unfavorable cost variance",
            "Financial Impact": money_full(variance),
            "Recommended Action": (
                "Review spend against budget and identify "
                "controllable cost opportunities."
            ),
            "Owner": "FP&A / Cost Owner",
            "Status": "Open",
        })

    return pd.DataFrame(actions)


# ============================================================
# SIDEBAR
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
        help="Upload a CSV containing financial ERP-style data.",
    )

    if uploaded_file is not None:

        df, error = load_uploaded_data(
            uploaded_file.getvalue()
        )

        data_source = "Uploaded ERP CSV"

    else:

        df, error = load_demo_data()

        data_source = "FinSight Demo ERP Data"

    st.caption(data_source)

    st.divider()

    st.markdown("### Navigation")

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
        "Go to",
        pages,
        label_visibility="collapsed",
    )


# ============================================================
# DATA VALIDATION
# ============================================================

if df is None:

    st.error(error)

    st.markdown(
        """
        ### Required ERP fields

        Your CSV should contain at least:

        - Date
        - Business Unit
        - Department
        - Account / Cost Category
        - Budget
        - Actual
        - Revenue

        FinSight automatically recognizes common variations such as
        `business_unit`, `department`, `account`, `budget`, `actual`,
        `revenue`, `cost_category`, `GL Account`, etc.
        """
    )

    st.stop()


# ============================================================
# GLOBAL FILTERS
# ============================================================

with st.sidebar:

    st.divider()

    st.markdown("### Global Filters")

    bu_options = ["All"] + sorted(
        df["Business Unit"].dropna().unique().tolist()
    )

    selected_bu = st.selectbox(
        "Business Unit",
        bu_options,
    )

    dept_options = ["All"] + sorted(
        df["Department"].dropna().unique().tolist()
    )

    selected_department = st.selectbox(
        "Department",
        dept_options,
    )


filtered_df = df.copy()

if selected_bu != "All":

    filtered_df = filtered_df[
        filtered_df["Business Unit"] == selected_bu
    ]

if selected_department != "All":

    filtered_df = filtered_df[
        filtered_df["Department"] == selected_department
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

    st.title("Executive Dashboard")

    st.caption(
        "Management view of revenue, cost, profitability, "
        "budget performance and financial risk."
    )

    kpis = calculate_kpis(filtered_df)

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Revenue",
        money(kpis["revenue"])
    )

    c2.metric(
        "Actual Cost",
        money(kpis["actual"])
    )

    c3.metric(
        "Budget",
        money(kpis["budget"])
    )

    c4.metric(
        "Variance",
        money(kpis["variance"]),
        delta=(
            f"{pct(safe_divide(kpis['variance'], kpis['budget']) * 100)}"
            if kpis["budget"] != 0
            else "0%"
        )
    )

    c5.metric(
        "Profit Margin",
        pct(kpis["margin"])
    )

    st.divider()

    left, right = st.columns(2)

    monthly = monthly_summary(filtered_df)

    with left:

        st.subheader("Revenue vs Actual Cost")

        if not monthly.empty:

            chart_df = monthly.melt(
                id_vars=["Month"],
                value_vars=["Revenue", "Actual"],
                var_name="Metric",
                value_name="Amount",
            )

            fig = px.line(
                chart_df,
                x="Month",
                y="Amount",
                color="Metric",
                markers=True,
                title="Monthly Financial Trend",
            )

            fig.update_layout(
                height=380,
                yaxis_title="Amount",
                xaxis_title="Month",
                legend_title="",
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with right:

        st.subheader("Business Unit Variance")

        bu = business_unit_summary(filtered_df)

        if not bu.empty:

            fig = px.bar(
                bu,
                x="Business Unit",
                y="Variance",
                title="Budget vs Actual Variance",
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
                use_container_width=True
            )

    st.subheader("AI CFO Executive Snapshot")

    insights = generate_ai_cfo_insights(
        filtered_df
    )

    if insights.empty:

        st.success(
            "No material management risks were identified "
            "from the current dataset."
        )

    else:

        for _, row in insights.head(3).iterrows():

            st.markdown(
                f"""
                <div class="insight-box">
                    <b>{row['Priority']} — {row['Finding']}</b><br>
                    <span>{row['Evidence']}</span><br><br>
                    <b>Management Action:</b>
                    {row['Management Action']}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# FP&A PERFORMANCE
# ============================================================

elif selected_page == "📊 FP&A Performance":

    st.title("FP&A Performance")

    st.caption(
        "Budget, actuals, variance and profitability analysis "
        "across the organization."
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Business Units",
            "Departments",
            "Cost Categories",
        ]
    )

    with tab1:

        result = business_unit_summary(
            filtered_df
        )

        display = result.copy()

        for col in [
            "Revenue",
            "Budget",
            "Actual",
            "Profit",
            "Variance",
        ]:
            display[col] = display[col].map(
                money_full
            )

        display["Variance %"] = result[
            "Variance %"
        ].map(pct)

        display["Margin %"] = result[
            "Margin %"
        ].map(pct)

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )

    with tab2:

        result = department_summary(
            filtered_df
        )

        display = result.copy()

        for col in [
            "Revenue",
            "Budget",
            "Actual",
            "Profit",
            "Variance",
        ]:
            display[col] = display[col].map(
                money_full
            )

        display["Variance %"] = result[
            "Variance %"
        ].map(pct)

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )

    with tab3:

        result = category_summary(
            filtered_df
        )

        display = result.copy()

        for col in [
            "Budget",
            "Actual",
            "Revenue",
            "Variance",
        ]:
            display[col] = display[col].map(
                money_full
            )

        display["Variance %"] = result[
            "Variance %"
        ].map(pct)

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    st.subheader("Monthly Financial Trend")

    monthly = monthly_summary(
        filtered_df
    )

    if not monthly.empty:

        display = monthly.copy()

        for col in [
            "Revenue",
            "Budget",
            "Actual",
            "Profit",
            "Variance",
        ]:
            display[col] = display[col].map(
                money_full
            )

        display["Margin %"] = monthly[
            "Margin %"
        ].map(pct)

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# RISK & ANOMALIES
# ============================================================

elif selected_page == "⚠️ Risk & Anomalies":

    st.title("Risk & Anomalies")

    st.caption(
        "Financial risk detection using budget variance "
        "and machine-learning anomaly detection."
    )

    anomaly_df = detect_anomalies(
        filtered_df
    )

    anomaly_count = int(
        anomaly_df["Anomaly Flag"].sum()
    )

    total_transactions = len(
        anomaly_df
    )

    risk_rate = safe_divide(
        anomaly_count,
        total_transactions
    ) * 100

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Transactions",
        f"{total_transactions:,}"
    )

    c2.metric(
        "Potential Anomalies",
        f"{anomaly_count:,}"
    )

    c3.metric(
        "Anomaly Rate",
        pct(risk_rate)
    )

    st.divider()

    left, right = st.columns(2)

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
            .reset_index(name="Anomalies")
        )

        if not anomaly_bu.empty:

            fig = px.bar(
                anomaly_bu,
                x="Business Unit",
                y="Anomalies",
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "No anomalies detected."
            )

    with right:

        st.subheader(
            "Anomalies by Cost Category"
        )

        anomaly_cat = (
            anomaly_df[
                anomaly_df["Anomaly Flag"] == 1
            ]
            .groupby("Account / Cost Category")
            .size()
            .reset_index(name="Anomalies")
        )

        if not anomaly_cat.empty:

            fig = px.bar(
                anomaly_cat,
                x="Account / Cost Category",
                y="Anomalies",
            )

            fig.update_layout(
                xaxis_tickangle=-35
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "No anomalies detected."
            )

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
            "Anomaly",
        ]

        display = flagged[
            [
                c
                for c in display_cols
                if c in flagged.columns
            ]
        ].copy()

        for col in [
            "Budget",
            "Actual",
            "Variance",
        ]:
            if col in display.columns:

                display[col] = display[
                    col
                ].map(money_full)

        if "Variance %" in display.columns:

            display["Variance %"] = flagged[
                "Variance %"
            ].map(pct)

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# CASH & LIQUIDITY
# ============================================================

elif selected_page == "💵 Cash & Liquidity":

    st.title("Cash & Liquidity Command Center")

    st.caption(
        "Modeled cash-flow intelligence for working-capital "
        "and liquidity planning."
    )

    cash = build_cash_flow(
        filtered_df
    )

    if cash.empty:

        st.warning(
            "Not enough financial history to model cash flow."
        )

    else:

        latest = cash.iloc[-1]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Cash Inflow",
            money(latest["Cash Inflow"])
        )

        c2.metric(
            "Cash Outflow",
            money(latest["Cash Outflow"])
        )

        c3.metric(
            "Net Cash Flow",
            money(latest["Net Cash Flow"])
        )

        c4.metric(
            "Closing Cash",
            money(latest["Closing Cash"])
        )

        st.divider()

        left, right = st.columns(2)

        with left:

            st.subheader(
                "Cash Inflow vs Outflow"
            )

            chart_df = cash.melt(
                id_vars=["Month"],
                value_vars=[
                    "Cash Inflow",
                    "Cash Outflow",
                ],
                var_name="Metric",
                value_name="Amount",
            )

            fig = px.bar(
                chart_df,
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
                markers=True,
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

        for col in [
            "Cash Inflow",
            "Cash Outflow",
            "Net Cash Flow",
            "Opening Cash",
            "Closing Cash",
        ]:
            display[col] = display[
                col
            ].map(money_full)

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "Note: cash flow is a modeled portfolio-prototype "
            "layer derived from the synthetic ERP dataset. "
            "It is not a live treasury or bank integration."
        )


# ============================================================
# FORECAST & SCENARIOS
# ============================================================

elif selected_page == "📈 Forecast & Scenarios":

    st.title("Forecast & Scenario Planner")

    st.caption(
        "Forward-looking revenue and cost scenarios "
        "for FP&A planning."
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

        if len(historical) >= 2:

            first_revenue = historical[
                "Revenue"
            ].iloc[0]

            last_revenue = historical[
                "Revenue"
            ].iloc[-1]

            historical_growth = (
                safe_divide(
                    last_revenue,
                    first_revenue
                ) ** (
                    1 / max(len(historical) - 1, 1)
                ) - 1
            ) * 100

        else:

            historical_growth = 0

        historical_margin = safe_divide(
            historical["Profit"].sum(),
            historical["Revenue"].sum()
        ) * 100

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Historical Avg Growth",
            pct(historical_growth)
        )

        c2.metric(
            "Historical Margin",
            pct(historical_margin)
        )

        c3.metric(
            "Forecast Horizon",
            "6 Months"
        )

        st.divider()

        scenario = st.selectbox(
            "Scenario",
            [
                "Base",
                "Optimistic",
                "Downside",
            ]
        )

        if scenario == "Base":

            revenue_col = "Base Revenue"
            cost_col = "Base Cost"
            profit_col = "Base Profit"

        elif scenario == "Optimistic":

            revenue_col = "Optimistic Revenue"
            cost_col = "Optimistic Cost"
            profit_col = "Optimistic Profit"

        else:

            revenue_col = "Downside Revenue"
            cost_col = "Downside Cost"
            profit_col = "Downside Profit"

        chart_df = forecast[
            ["Month", revenue_col, cost_col]
        ].melt(
            id_vars=["Month"],
            var_name="Metric",
            value_name="Amount",
        )

        fig = px.line(
            chart_df,
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
            use_container_width=True
        )

        st.subheader(
            f"{scenario} Scenario Forecast"
        )

        display = forecast[
            [
                "Month",
                revenue_col,
                cost_col,
                profit_col,
            ]
        ].copy()

        display.columns = [
            "Month",
            "Revenue",
            "Cost",
            "Profit",
        ]

        for col in [
            "Revenue",
            "Cost",
            "Profit",
        ]:
            display[col] = display[
                col
            ].map(money_full)

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "Forecast uses a transparent statistical trend model "
            "with scenario assumptions. It is intended for "
            "portfolio demonstration and decision support."
        )


# ============================================================
# AI CFO
# ============================================================

elif selected_page == "🤖 AI CFO":

    st.title("🤖 AI CFO")

    st.caption(
        "Evidence-based financial decision support "
        "for management questions."
    )

    st.markdown(
        """
        <div class="section-card">
            <b>Ask FinSight AI a finance question.</b><br>
            <span class="small-muted">
            Examples: What is driving the cost overrun?
            Which business unit needs attention?
            What should management review first?
            </span>
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
        type="primary"
    ):

        insights = generate_ai_cfo_insights(
            filtered_df
        )

        if insights.empty:

            st.success(
                "FinSight AI did not identify a material "
                "financial risk from the current dataset."
            )

        else:

            # Choose insight based on keywords
            question_lower = question.lower()

            selected_insight = None

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

                selected_insight = insights[
                    insights["Finding"].str.contains(
                        "cost|variance|overrun",
                        case=False,
                        regex=True
                    )
                ]

            if (
                selected_insight is None
                or selected_insight.empty
            ):

                selected_insight = insights

            row = selected_insight.iloc[0]

            st.subheader(
                "AI CFO Recommendation"
            )

            st.markdown(
                f"""
                <div class="insight-box">
                    <h4>Finding</h4>
                    {row['Finding']}

                    <h4>Evidence</h4>
                    {row['Evidence']}

                    <h4>Business Impact</h4>
                    {row['Business Impact']}

                    <h4>Management Action</h4>
                    {row['Management Action']}

                    <h4>Priority</h4>
                    {row['Priority']}

                    <h4>Owner</h4>
                    {row['Owner']}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader(
                "Full Decision Queue"
            )

            st.dataframe(
                insights,
                use_container_width=True,
                hide_index=True,
            )

    else:

        st.info(
            "Enter a management question and run the "
            "AI CFO analysis."
        )


# ============================================================
# ACTION CENTER
# ============================================================

elif selected_page == "🎯 Action Center":

    st.title("🎯 Management Action Center")

    st.caption(
        "Prioritized financial actions generated from "
        "budget variance and cost-driver analysis."
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
            p0
        )

        c2.metric(
            "P1 High",
            p1
        )

        c3.metric(
            "P2 Monitor",
            p2
        )

        st.divider()

        for _, row in actions.iterrows():

            priority_class = (
                "priority-p0"
                if row["Priority"] == "P0"
                else (
                    "priority-p1"
                    if row["Priority"] == "P1"
                    else "priority-p2"
                )
            )

            st.markdown(
                f"""
                <div class="action-box {priority_class}">
                    <b>{row['Priority']} — {row['Area']}</b><br><br>

                    <b>Issue:</b>
                    {row['Issue']}<br>

                    <b>Financial Impact:</b>
                    {row['Financial Impact']}<br>

                    <b>Recommended Action:</b>
                    {row['Recommended Action']}<br>

                    <b>Owner:</b>
                    {row['Owner']}<br>

                    <b>Status:</b>
                    {row['Status']}
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

    st.title("🔎 Data Explorer")

    st.caption(
        "Explore the underlying ERP-style financial dataset."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Records",
        f"{len(filtered_df):,}"
    )

    c2.metric(
        "Columns",
        f"{len(filtered_df.columns):,}"
    )

    c3.metric(
        "Business Units",
        f"{filtered_df['Business Unit'].nunique():,}"
    )

    st.divider()

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Filtered CSV",
        data=csv_data,
        file_name="finsight_filtered_financials.csv",
        mime="text/csv",
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#64748b; padding:1rem;">
        <b>FinSight AI</b> · Agentic FP&A & ERP Intelligence<br>
        <span style="font-size:0.8rem;">
        Portfolio prototype using synthetic ERP-style financial data
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
