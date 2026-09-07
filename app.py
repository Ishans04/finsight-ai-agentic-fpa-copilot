elif selected_page == "📈 Forecast & Scenarios":

    st.title("Forecast & Scenario Planner")

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

        # ====================================================
        # HISTORICAL METRICS
        # ====================================================

        if len(historical) >= 2:

            first_revenue = historical[
                "Revenue"
            ].iloc[0]

            last_revenue = historical[
                "Revenue"
            ].iloc[-1]

            periods_count = max(
                len(historical) - 1,
                1
            )

            historical_growth = (
                (
                    safe_divide(
                        last_revenue,
                        first_revenue
                    )
                ** (1 / periods_count)
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

        # ====================================================
        # KPI ROW
        # ====================================================

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Historical Growth",
            percentage(historical_growth)
        )

        col2.metric(
            "Historical Margin",
            percentage(historical_margin)
        )

        col3.metric(
            "Forecast Horizon",
            "6 Months"
        )

        base_total_profit = forecast[
            "Base Profit"
        ].sum()

        col4.metric(
            "Base Forecast Profit",
            money(base_total_profit)
        )

        st.divider()

        # ====================================================
        # SCENARIO ASSUMPTIONS
        # ====================================================

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
Revenue: +12% vs Base
<br>
Cost: -5% vs Base
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
Revenue: -12% vs Base
<br>
Cost: +10% vs Base
<br>
Assumption: Slower growth + cost pressure
</div>""",
                unsafe_allow_html=True
            )

        st.divider()

        # ====================================================
        # ALL THREE SCENARIOS — REVENUE
        # ====================================================

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

        # ====================================================
        # ALL THREE SCENARIOS — PROFIT
        # ====================================================

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

        # ====================================================
        # SCENARIO SUMMARY
        # ====================================================

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

        base_profit_total = forecast[
            "Base Profit"
        ].sum()

        optimistic_profit_total = forecast[
            "Optimistic Profit"
        ].sum()

        downside_profit_total = forecast[
            "Downside Profit"
        ].sum()

        base_margin = (
            safe_divide(
                base_profit_total,
                base_revenue_total
            ) * 100
        )

        optimistic_margin = (
            safe_divide(
                optimistic_profit_total,
                optimistic_revenue_total
            ) * 100
        )

        downside_margin = (
            safe_divide(
                downside_profit_total,
                downside_revenue_total
            ) * 100
        )

        summary = pd.DataFrame({

            "Scenario": [
                "Base",
                "Optimistic",
                "Downside"
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

        # ====================================================
        # MONTHLY FORECAST DETAIL
        # ====================================================

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

        # ====================================================
        # MANAGEMENT INTERPRETATION
        # ====================================================

        st.subheader(
            "AI CFO Scenario Interpretation"
        )

        profit_gap = (
            optimistic_profit_total
            - downside_profit_total
        )

        st.markdown(
            f"""<div class="insight-box">
<b>Management Finding</b>
<br><br>
The six-month modeled profit range between the
Optimistic and Downside scenarios is
<b>{money_full(profit_gap)}</b>.
<br><br>
<b>Optimistic:</b>
{money_full(optimistic_profit_total)}
profit at {percentage(optimistic_margin)} margin.
<br><br>
<b>Base:</b>
{money_full(base_profit_total)}
profit at {percentage(base_margin)} margin.
<br><br>
<b>Downside:</b>
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
