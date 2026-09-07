import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression


# ============================================================
# FINSIGHT AI
# Agentic FP&A & ERP Intelligence
# ============================================================

st.set_page_config(
    page_title="FinSight AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,0.2);
}

[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 12px;
    padding: 14px;
    background: rgba(128,128,128,0.04);
}

.hero {
    padding: 22px 26px;
    border-radius: 16px;
    margin-bottom: 20px;
    border: 1px solid rgba(128,128,128,0.2);
    background: linear-gradient(
        135deg,
        rgba(80,80,180,0.12),
        rgba(0,160,160,0.08)
    );
}

.hero h1 {
    margin: 0;
    font-size: 2.3rem;
}

.hero p {
    margin-top: 8px;
    opacity: 0.75;
    font-size: 1rem;
}

.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 10px;
}

.insight-box {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.22);
    background: rgba(128,128,128,0.05);
    margin-bottom: 12px;
}

.small-muted {
    opacity: 0.65;
    font-size: 0.85rem;
}

.action-box {
    padding: 15px;
    border-radius: 10px;
    border: 1px solid rgba(128,128,128,0.2);
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def money(value):
    if pd.isna(value):
        return "₹0"

    value = float(value)

    if abs(value) >= 1e7:
        return f"₹{value / 1e7:.2f} Cr"

    if abs(value) >= 1e5:
        return f"₹{value / 1e5:.2f} L"

    return f"₹{value:,.0f}"


def money_full(value):
    if pd.isna(value):
        return "₹0"

    return f"₹{float(value):,.0f}"


def pct(value):
    if pd.isna(value):
        return "0.00%"

    return f"{float(value):.2f}%"


def format_table(df):
    return df.style.format({
        col: "₹{:,.0f}"
        for col in df.columns
        if col in [
            "Revenue",
            "Budget",
            "Actual",
            "Profit",
            "Variance",
            "Cash Inflow",
            "Cash Outflow",
            "Closing Cash"
        ]
    })


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    possible_files = [
        Path("synthetic_erp_financials.csv"),
        Path("data/synthetic_erp_financials.csv")
    ]

    file_path = None

    for path in possible_files:
        if path.exists():
            file_path = path
            break

    if file_path is None:
        return pd.DataFrame()

    data = pd.read_csv(file_path)

    rename_map = {
        "transaction_id": "Transaction ID",
        "date": "Date",
        "business_unit": "Business Unit",
        "department": "Department",
        "account": "Account / Cost Category",
        "budget": "Budget",
        "actual": "Actual",
        "revenue": "Revenue",
        "variance": "Variance",
        "variance_pct": "Variance %",
        "profit": "Profit",
        "anomaly_flag": "Anomaly Flag",
        "anomaly": "Anomaly"
    }

    data = data.rename(columns=rename_map)

    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(
            data["Date"],
            errors="coerce"
        )

    numeric_columns = [
        "Budget",
        "Actual",
        "Revenue",
        "Variance",
        "Variance %",
        "Profit"
    ]

    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(
                data[col],
                errors="coerce"
            )

    if "Variance" not in data.columns:
        data["Variance"] = (
            data["Actual"] -
            data["Budget"]
        )

    if "Variance %" not in data.columns:
        data["Variance %"] = np.where(
            data["Budget"] != 0,
            data["Variance"] /
            data["Budget"] * 100,
            0
        )

    if "Profit" not in data.columns:
        data["Profit"] = (
            data["Revenue"] -
            data["Actual"]
        )

    return data


# ============================================================
# FILE UPLOAD
# ============================================================

default_df = load_data()

with st.sidebar:

    st.markdown("## 📊 FinSight AI")

    st.caption(
        "Agentic FP&A & ERP Intelligence"
    )

    st.divider()

    st.markdown("### Data Source")

    uploaded_file = st.file_uploader(
        "Upload ERP CSV",
        type=["csv"],
        help=(
            "Upload a CSV containing financial "
            "transactions."
        )
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        rename_map = {
            "transaction_id": "Transaction ID",
            "date": "Date",
            "business_unit": "Business Unit",
            "department": "Department",
            "account": "Account / Cost Category",
            "budget": "Budget",
            "actual": "Actual",
            "revenue": "Revenue",
            "variance": "Variance",
            "variance_pct": "Variance %",
            "profit": "Profit",
            "anomaly_flag": "Anomaly Flag",
            "anomaly": "Anomaly"
        }

        df = df.rename(columns=rename_map)

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(
                df["Date"],
                errors="coerce"
            )

        if "Variance" not in df.columns:
            df["Variance"] = (
                df["Actual"] -
                df["Budget"]
            )

        if "Variance %" not in df.columns:
            df["Variance %"] = np.where(
                df["Budget"] != 0,
                df["Variance"] /
                df["Budget"] * 100,
                0
            )

        if "Profit" not in df.columns:
            df["Profit"] = (
                df["Revenue"] -
                df["Actual"]
            )

        data_source = "Uploaded ERP Data"

    else:

        df = default_df.copy()
        data_source = "FinSight Demo ERP Data"

    st.success(data_source)

    st.divider()

    st.markdown("### Navigation")

    page = st.radio(
        "Go to",
        [
            "🏠 Executive Dashboard",
            "📊 FP&A Performance",
            "⚠️ Risk & Anomalies",
            "💰 Cash & Liquidity",
            "🔮 Forecast & Scenarios",
            "🤖 AI CFO",
            "🎯 Action Center",
            "📁 Data Explorer"
        ],
        label_visibility="collapsed"
    )


# ============================================================
# DATA VALIDATION
# ============================================================

if df.empty:

    st.error(
        "No financial dataset found. "
        "Upload an ERP CSV to continue."
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
    "Profit"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )

    st.stop()


# ============================================================
# GLOBAL FILTERS
# ============================================================

with st.sidebar:

    st.divider()

    st.markdown("### Global Filters")

    selected_bu = st.multiselect(
        "Business Unit",
        sorted(
            df["Business Unit"]
            .dropna()
            .unique()
        ),
        default=[]
    )

    selected_dept = st.multiselect(
        "Department",
        sorted(
            df["Department"]
            .dropna()
            .unique()
        ),
        default=[]
    )


filtered_df = df.copy()

if selected_bu:
    filtered_df = filtered_df[
        filtered_df["Business Unit"].isin(
            selected_bu
        )
    ]

if selected_dept:
    filtered_df = filtered_df[
        filtered_df["Department"].isin(
            selected_dept
        )
    ]


# ============================================================
# COMMON METRICS
# ============================================================

revenue = filtered_df["Revenue"].sum()
budget = filtered_df["Budget"].sum()
actual = filtered_df["Actual"].sum()
profit = filtered_df["Profit"].sum()

variance = actual - budget

variance_percent = (
    variance / budget * 100
    if budget != 0 else 0
)

margin = (
    profit / revenue * 100
    if revenue != 0 else 0
)

if "Anomaly Flag" in filtered_df.columns:
    anomaly_count = (
        filtered_df["Anomaly Flag"] == -1
    ).sum()
else:
    anomaly_count = 0


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <h1>FinSight AI</h1>
    <p>
        Agentic FP&A & ERP Intelligence Command Center
        · From financial data to management action
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# PAGE 1 — EXECUTIVE DASHBOARD
# ============================================================

if page == "🏠 Executive Dashboard":

    st.markdown(
        '<div class="section-title">'
        'Executive Command Center'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Revenue",
        money(revenue)
    )

    c2.metric(
        "Actual Cost",
        money(actual)
    )

    c3.metric(
        "Budget",
        money(budget)
    )

    c4.metric(
        "Variance",
        money(variance),
        f"{variance_percent:.2f}%"
    )

    c5.metric(
        "Profit Margin",
        f"{margin:.2f}%"
    )

    st.divider()

    # Monthly trend

    monthly = (
        filtered_df
        .assign(
            Month=filtered_df["Date"]
            .dt.to_period("M")
            .astype(str)
        )
        .groupby("Month")
        .agg(
            Revenue=("Revenue", "sum"),
            Budget=("Budget", "sum"),
            Actual=("Actual", "sum"),
            Profit=("Profit", "sum")
        )
        .reset_index()
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=monthly["Month"],
                y=monthly["Revenue"],
                mode="lines+markers",
                name="Revenue"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=monthly["Month"],
                y=monthly["Actual"],
                mode="lines+markers",
                name="Actual Cost"
            )
        )

        fig.update_layout(
            title="Revenue vs Actual Cost",
            xaxis_title="Month",
            yaxis_title="₹",
            height=430,
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        bu = (
            filtered_df
            .groupby("Business Unit")
            .agg(
                Budget=("Budget", "sum"),
                Actual=("Actual", "sum")
            )
            .reset_index()
        )

        bu["Variance"] = (
            bu["Actual"] -
            bu["Budget"]
        )

        fig = px.bar(
            bu.sort_values(
                "Variance",
                ascending=False
            ),
            x="Business Unit",
            y="Variance",
            title="Budget vs Actual Variance"
        )

        fig.update_layout(
            height=430,
            yaxis_title="Variance (₹)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # AI insight

    st.markdown(
        '<div class="section-title">'
        '🤖 AI CFO Snapshot'
        '</div>',
        unsafe_allow_html=True
    )

    cost = (
        filtered_df
        .groupby("Account / Cost Category")
        .agg(
            Budget=("Budget", "sum"),
            Actual=("Actual", "sum")
        )
        .reset_index()
    )

    cost["Variance"] = (
        cost["Actual"] -
        cost["Budget"]
    )

    if len(cost) > 0:

        top_cost = cost.loc[
            cost["Variance"].idxmax()
        ]

        st.markdown(
            f"""
            <div class="insight-box">
            <b>Key financial signal</b><br><br>
            <b>{top_cost['Account / Cost Category']}</b>
            is the largest unfavorable cost driver,
            with a variance of
            <b>{money_full(top_cost['Variance'])}</b>.
            <br><br>
            Overall actual cost is
            <b>{money_full(actual)}</b>
            against a budget of
            <b>{money_full(budget)}</b>.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.caption(
        f"Data source: {data_source} · "
        f"{len(filtered_df):,} transactions"
    )


# ============================================================
# PAGE 2 — FP&A PERFORMANCE
# ============================================================

elif page == "📊 FP&A Performance":

    st.header("📊 FP&A Performance")

    st.write(
        "Drill from business-unit performance "
        "into departments and cost categories."
    )

    bu = (
        filtered_df
        .groupby("Business Unit")
        .agg(
            Revenue=("Revenue", "sum"),
            Budget=("Budget", "sum"),
            Actual=("Actual", "sum"),
            Profit=("Profit", "sum"),
            Transactions=("Transaction ID", "count")
        )
        .reset_index()
    )

    bu["Variance"] = (
        bu["Actual"] -
        bu["Budget"]
    )

    bu["Variance %"] = np.where(
        bu["Budget"] != 0,
        bu["Variance"] /
        bu["Budget"] * 100,
        0
    )

    bu["Margin %"] = np.where(
        bu["Revenue"] != 0,
        bu["Profit"] /
        bu["Revenue"] * 100,
        0
    )

    st.subheader("Business Unit Performance")

    display_bu = bu.copy()

    st.dataframe(
        display_bu.style.format({
            "Revenue": "₹{:,.0f}",
            "Budget": "₹{:,.0f}",
            "Actual": "₹{:,.0f}",
            "Profit": "₹{:,.0f}",
            "Variance": "₹{:,.0f}",
            "Variance %": "{:.2f}%",
            "Margin %": "{:.2f}%",
            "Transactions": "{:,.0f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    dept = (
        filtered_df
        .groupby("Department")
        .agg(
            Revenue=("Revenue", "sum"),
            Budget=("Budget", "sum"),
            Actual=("Actual", "sum"),
            Profit=("Profit", "sum")
        )
        .reset_index()
    )

    dept["Variance"] = (
        dept["Actual"] -
        dept["Budget"]
    )

    dept["Variance %"] = np.where(
        dept["Budget"] != 0,
        dept["Variance"] /
        dept["Budget"] * 100,
        0
    )

    st.subheader("Department Performance")

    st.dataframe(
        dept.style.format({
            "Revenue": "₹{:,.0f}",
            "Budget": "₹{:,.0f}",
            "Actual": "₹{:,.0f}",
            "Profit": "₹{:,.0f}",
            "Variance": "₹{:,.0f}",
            "Variance %": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    category = (
        filtered_df
        .groupby("Account / Cost Category")
        .agg(
            Budget=("Budget", "sum"),
            Actual=("Actual", "sum")
        )
        .reset_index()
    )

    category["Variance"] = (
        category["Actual"] -
        category["Budget"]
    )

    category["Variance %"] = np.where(
        category["Budget"] != 0,
        category["Variance"] /
        category["Budget"] * 100,
        0
    )

    st.subheader("Cost Driver Analysis")

    st.dataframe(
        category
        .sort_values(
            "Variance",
            ascending=False
        )
        .style.format({
            "Budget": "₹{:,.0f}",
            "Actual": "₹{:,.0f}",
            "Variance": "₹{:,.0f}",
            "Variance %": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 3 — RISK & ANOMALIES
# ============================================================

elif page == "⚠️ Risk & Anomalies":

    st.header("⚠️ Risk & Anomaly Intelligence")

    st.write(
        "Identify unusual financial transactions "
        "and potential control risks."
    )

    if "Anomaly Flag" not in filtered_df.columns:

        st.info(
            "The uploaded dataset does not contain "
            "anomaly flags. Running the ML anomaly engine..."
        )

        numeric = filtered_df[
            ["Budget", "Actual", "Revenue"]
        ].fillna(0)

        detector = IsolationForest(
            contamination=0.05,
            random_state=42
        )

        filtered_df["Anomaly Flag"] = (
            detector.fit_predict(numeric)
        )

    anomalies = filtered_df[
        filtered_df["Anomaly Flag"] == -1
    ].copy()

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "Transactions",
        f"{len(filtered_df):,}"
    )

    r2.metric(
        "Anomalies",
        f"{len(anomalies):,}"
    )

    r3.metric(
        "Anomaly Rate",
        pct(
            len(anomalies) /
            len(filtered_df) * 100
            if len(filtered_df)
            else 0
        )
    )

    st.divider()

    if len(anomalies) > 0:

        anomaly_bu = (
            anomalies
            .groupby("Business Unit")
            .size()
            .reset_index(
                name="Anomalies"
            )
            .sort_values(
                "Anomalies",
                ascending=False
            )
        )

        col1, col2 = st.columns(2)

        with col1:

            fig = px.bar(
                anomaly_bu,
                x="Business Unit",
                y="Anomalies",
                title="Anomalies by Business Unit"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            anomaly_cost = (
                anomalies
                .groupby(
                    "Account / Cost Category"
                )
                .size()
                .reset_index(
                    name="Anomalies"
                )
                .sort_values(
                    "Anomalies",
                    ascending=False
                )
            )

            fig = px.bar(
                anomaly_cost,
                x="Account / Cost Category",
                y="Anomalies",
                title="Anomalies by Cost Category"
            )

            fig.update_layout(
                xaxis_tickangle=-35
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader(
            "Transactions Requiring Investigation"
        )

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
            c for c in columns
            if c in anomalies.columns
        ]

        st.dataframe(
            anomalies[
                available
            ]
            .sort_values(
                "Actual",
                ascending=False
            )
            .head(50)
            .style.format({
                "Budget": "₹{:,.0f}",
                "Actual": "₹{:,.0f}",
                "Variance": "₹{:,.0f}",
                "Variance %": "{:.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            "No anomalies detected in the current dataset."
        )


# ============================================================
# PAGE 4 — CASH & LIQUIDITY
# ============================================================

elif page == "💰 Cash & Liquidity":

    st.header("💰 Cash Flow & Liquidity Command Center")

    st.info(
        "Cash-flow values shown here are a modeled "
        "synthetic ERP cash-flow simulation."
    )

    cash = filtered_df.copy()

    np.random.seed(42)

    cash["Receivable"] = (
        cash["Revenue"] *
        np.random.uniform(
            0.20,
            0.45,
            len(cash)
        )
    )

    cash["Payable"] = (
        cash["Actual"] *
        np.random.uniform(
            0.15,
            0.40,
            len(cash)
        )
    )

    cash["Month"] = (
        cash["Date"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly_cash = (
        cash
        .groupby("Month")
        .agg(
            Cash_Inflow=("Receivable", "sum"),
            Cash_Outflow=("Payable", "sum")
        )
        .reset_index()
    )

    monthly_cash["Net Cash Flow"] = (
        monthly_cash["Cash_Inflow"] -
        monthly_cash["Cash_Outflow"]
    )

    opening_cash = 10_000_000

    running_cash = opening_cash
    opening_values = []
    closing_values = []

    for value in monthly_cash["Net Cash Flow"]:

        opening_values.append(
            running_cash
        )

        running_cash += value

        closing_values.append(
            running_cash
        )

    monthly_cash["Opening Cash"] = (
        opening_values
    )

    monthly_cash["Closing Cash"] = (
        closing_values
    )

    ending_cash = (
        monthly_cash["Closing Cash"].iloc[-1]
    )

    minimum_cash = (
        monthly_cash["Closing Cash"].min()
    )

    avg_outflow = (
        monthly_cash["Cash_Outflow"].mean()
    )

    runway = (
        ending_cash / avg_outflow
        if avg_outflow > 0
        else np.inf
    )

    if minimum_cash < 0:
        status = "CRITICAL"
    elif minimum_cash < opening_cash * 0.25:
        status = "HIGH RISK"
    elif minimum_cash < opening_cash * 0.50:
        status = "WATCH"
    else:
        status = "HEALTHY"

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Opening Cash",
        money(opening_cash)
    )

    c2.metric(
        "Ending Cash",
        money(ending_cash)
    )

    c3.metric(
        "Avg Monthly Outflow",
        money(avg_outflow)
    )

    c4.metric(
        "Liquidity Status",
        status
    )

    st.divider()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=monthly_cash["Month"],
            y=monthly_cash["Cash_Inflow"],
            mode="lines+markers",
            name="Cash Inflow"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=monthly_cash["Month"],
            y=monthly_cash["Cash_Outflow"],
            mode="lines+markers",
            name="Cash Outflow"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=monthly_cash["Month"],
            y=monthly_cash["Closing Cash"],
            mode="lines+markers",
            name="Closing Cash"
        )
    )

    fig.update_layout(
        title="Cash Flow & Liquidity Trend",
        xaxis_title="Month",
        yaxis_title="₹",
        height=500,
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Monthly Cash Position")

    st.dataframe(
        monthly_cash.style.format({
            "Cash_Inflow": "₹{:,.0f}",
            "Cash_Outflow": "₹{:,.0f}",
            "Net Cash Flow": "₹{:,.0f}",
            "Opening Cash": "₹{:,.0f}",
            "Closing Cash": "₹{:,.0f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("AI CFO Liquidity Recommendation")

    if status == "CRITICAL":

        st.error(
            "Immediate liquidity intervention recommended. "
            "Accelerate collections and defer non-critical spending."
        )

    elif status == "HIGH RISK":

        st.warning(
            "Liquidity risk is elevated. "
            "Prioritize receivable collections and review upcoming liabilities."
        )

    elif status == "WATCH":

        st.warning(
            "Cash position requires monitoring. "
            "Review payment timing and discretionary expenditure."
        )

    else:

        st.success(
            "Liquidity position is currently healthy "
            "under the modeled assumptions."
        )


# ============================================================
# PAGE 5 — FORECAST & SCENARIOS
# ============================================================

elif page == "🔮 Forecast & Scenarios":

    st.header("🔮 Revenue Forecast & Scenario Planning")

    monthly = (
        filtered_df
        .assign(
            Month=filtered_df["Date"]
            .dt.to_period("M")
            .astype(str)
        )
        .groupby("Month")
        .agg(
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum")
        )
        .reset_index()
    )

    monthly["Month_Date"] = pd.to_datetime(
        monthly["Month"]
    )

    monthly = monthly.sort_values(
        "Month_Date"
    ).reset_index(drop=True)

    monthly["Period"] = np.arange(
        len(monthly)
    )

    model = LinearRegression()

    model.fit(
        monthly[["Period"]],
        monthly["Revenue"]
    )

    periods = np.arange(
        len(monthly),
        len(monthly) + 6
    )

    future_dates = pd.date_range(
        monthly["Month_Date"].max()
        + pd.offsets.MonthBegin(1),
        periods=6,
        freq="MS"
    )

    future_X = pd.DataFrame({
        "Period": periods
    })

    base = model.predict(
        future_X
    )

    base = np.maximum(
        base,
        0
    )

    optimistic = (
        base * 1.08
    )

    downside = (
        base * 0.92
    )

    forecast = pd.DataFrame({
        "Month": future_dates,
        "Base Case": base,
        "Optimistic": optimistic,
        "Downside": downside
    })

    margin_hist = (
        monthly["Profit"].sum() /
        monthly["Revenue"].sum()
        if monthly["Revenue"].sum()
        else 0
    )

    forecast["Base Profit"] = (
        forecast["Base Case"] *
        margin_hist
    )

    forecast["Optimistic Profit"] = (
        forecast["Optimistic"] *
        margin_hist
    )

    forecast["Downside Profit"] = (
        forecast["Downside"] *
        margin_hist
    )

    st.subheader("Next 6 Months")

    display_forecast = forecast.copy()

    display_forecast["Month"] = (
        display_forecast["Month"]
        .dt.strftime("%b %Y")
    )

    st.dataframe(
        display_forecast.style.format({
            "Base Case": "₹{:,.0f}",
            "Optimistic": "₹{:,.0f}",
            "Downside": "₹{:,.0f}",
            "Base Profit": "₹{:,.0f}",
            "Optimistic Profit": "₹{:,.0f}",
            "Downside Profit": "₹{:,.0f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=monthly["Month_Date"],
            y=monthly["Revenue"],
            mode="lines+markers",
            name="Historical"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast["Month"],
            y=forecast["Base Case"],
            mode="lines+markers",
            name="Base Case"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast["Month"],
            y=forecast["Optimistic"],
            mode="lines+markers",
            name="Optimistic"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast["Month"],
            y=forecast["Downside"],
            mode="lines+markers",
            name="Downside"
        )
    )

    fig.update_layout(
        title="Revenue Forecast & Scenarios",
        xaxis_title="Month",
        yaxis_title="Revenue (₹)",
        height=550,
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("AI CFO Forecast Interpretation")

    base_total = forecast["Base Case"].sum()
    upside_total = forecast["Optimistic"].sum()
    downside_total = forecast["Downside"].sum()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Base Revenue",
        money(base_total)
    )

    c2.metric(
        "Optimistic",
        money(upside_total)
    )

    c3.metric(
        "Downside",
        money(downside_total)
    )

    st.info(
        "Base case represents the model trend. "
        "Optimistic and downside cases are explicit ±8% "
        "management stress scenarios, not guaranteed predictions."
    )


# ============================================================
# PAGE 6 — AI CFO
# ============================================================

elif page == "🤖 AI CFO":

    st.header("🤖 AI CFO")

    st.write(
        "Ask a natural-language finance question "
        "and FinSight will convert financial evidence "
        "into a management-oriented response."
    )

    question = st.text_input(
        "Ask your CFO question",
        placeholder=(
            "Why did costs increase?"
        )
    )

    examples = [
        "Why did costs increase?",
        "Which business unit needs attention?",
        "What are the biggest cost drivers?",
        "Where are the anomalies?",
        "What should management focus on?"
    ]

    st.caption(
        "Examples: " +
        " · ".join(examples)
    )

    if st.button(
        "Run AI CFO Analysis",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "Enter a finance question first."
            )

        else:

            cost = (
                filtered_df
                .groupby(
                    "Account / Cost Category"
                )
                .agg(
                    Budget=("Budget", "sum"),
                    Actual=("Actual", "sum")
                )
                .reset_index()
            )

            cost["Variance"] = (
                cost["Actual"] -
                cost["Budget"]
            )

            bu = (
                filtered_df
                .groupby("Business Unit")
                .agg(
                    Budget=("Budget", "sum"),
                    Actual=("Actual", "sum"),
                    Revenue=("Revenue", "sum"),
                    Profit=("Profit", "sum")
                )
                .reset_index()
            )

            bu["Variance"] = (
                bu["Actual"] -
                bu["Budget"]
            )

            top_cost = cost.loc[
                cost["Variance"].idxmax()
            ]

            top_bu = bu.loc[
                bu["Variance"].idxmax()
            ]

            st.subheader(
                "Executive Finding"
            )

            st.markdown(
                f"""
                <div class="insight-box">

                <b>Finding</b><br>
                {top_cost['Account / Cost Category']}
                is the largest unfavorable cost driver.

                <br><br>

                <b>Evidence</b><br>
                Actual spend:
                {money_full(top_cost['Actual'])}

                <br>
                Budget:
                {money_full(top_cost['Budget'])}

                <br>
                Variance:
                {money_full(top_cost['Variance'])}

                <br><br>

                <b>Business Impact</b><br>
                Overall actual cost is
                {money_full(actual)}
                versus budget of
                {money_full(budget)}.

                <br><br>

                <b>Management Action</b><br>
                Investigate the root cause of the
                {top_cost['Account / Cost Category']}
                variance, validate the responsible cost center
                and establish corrective controls.

                <br><br>

                <b>Priority</b><br>
                P1 — High

                <br><br>

                <b>Owner</b><br>
                Finance / Cost Center Owner

                </div>
                """,
                unsafe_allow_html=True
            )

            st.subheader(
                "Business Unit Requiring Attention"
            )

            st.warning(
                f"{top_bu['Business Unit']} has the "
                f"largest unfavorable variance of "
                f"{money_full(top_bu['Variance'])}."
            )

            st.subheader(
                "Decision Path"
            )

            st.write(
                "Financial Data → Variance Analysis → "
                "Cost Driver → Business Impact → "
                "Risk → Management Action"
            )


# ============================================================
# PAGE 7 — ACTION CENTER
# ============================================================

elif page == "🎯 Action Center":

    st.header("🎯 Management Action Center")

    st.write(
        "AI-generated actions derived from financial signals."
    )

    actions = []

    cost = (
        filtered_df
        .groupby(
            "Account / Cost Category"
        )
        .agg(
            Budget=("Budget", "sum"),
            Actual=("Actual", "sum")
        )
        .reset_index()
    )

    cost["Variance"] = (
        cost["Actual"] -
        cost["Budget"]
    )

    cost["Variance %"] = np.where(
        cost["Budget"] != 0,
        cost["Variance"] /
        cost["Budget"] * 100,
        0
    )

    for _, row in cost.iterrows():

        if row["Variance %"] >= 20:

            priority = "P0 — Critical"

        elif row["Variance %"] >= 10:

            priority = "P1 — High"

        elif row["Variance %"] >= 5:

            priority = "P2 — Medium"

        else:

            continue

        actions.append({
            "Priority": priority,
            "Action":
                f"Investigate {row['Account / Cost Category']} "
                f"overspend",
            "Financial Impact":
                row["Variance"],
            "Owner":
                "Finance / Cost Center Owner",
            "Status":
                "Open"
        })

    if actions:

        action_df = pd.DataFrame(
            actions
        )

        priority_order = {
            "P0 — Critical": 0,
            "P1 — High": 1,
            "P2 — Medium": 2
        }

        action_df["_order"] = (
            action_df["Priority"]
            .map(priority_order)
        )

        action_df = (
            action_df
            .sort_values("_order")
            .drop(columns="_order")
        )

        st.dataframe(
            action_df.style.format({
                "Financial Impact":
                    "₹{:,.0f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "Recommended Execution Sequence"
        )

        st.write(
            "1. Resolve P0 critical items."
        )

        st.write(
            "2. Investigate P1 high-priority variances."
        )

        st.write(
            "3. Assign accountable cost-center owners."
        )

        st.write(
            "4. Track corrective action to closure."
        )

    else:

        st.success(
            "No material cost actions triggered "
            "under the current thresholds."
        )


# ============================================================
# PAGE 8 — DATA EXPLORER
# ============================================================

elif page == "📁 Data Explorer":

    st.header("📁 ERP Data Explorer")

    st.write(
        "Explore the underlying financial transactions "
        "used by FinSight AI."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Records",
        f"{len(filtered_df):,}"
    )

    c2.metric(
        "Fields",
        f"{len(filtered_df.columns):,}"
    )

    c3.metric(
        "Business Units",
        f"{filtered_df['Business Unit'].nunique():,}"
    )

    st.divider()

    st.subheader(
        "Transaction Data"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

    csv = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Filtered Data",
        csv,
        "finsight_filtered_financials.csv",
        "text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "FinSight AI · Agentic FP&A & ERP Intelligence · "
    "Portfolio prototype using synthetic ERP-style financial data"
)
