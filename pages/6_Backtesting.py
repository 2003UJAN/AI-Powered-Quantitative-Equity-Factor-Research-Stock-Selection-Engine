# pages/6_Backtesting.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from modules.backtesting_engine import run_backtest


st.title("Portfolio Backtesting")
st.caption(
    "Historical simulation of factor-based stock selection "
    "and portfolio performance."
)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

prices = st.session_state.get("market_prices")
volumes = st.session_state.get("market_volumes")
benchmark_prices = st.session_state.get("benchmark_prices")

if prices is None or prices.empty:
    st.warning(
        "Market data is not available. Please visit the "
        "Market Universe page and download the stock data first."
    )
    st.stop()

st.subheader("Backtesting Configuration")

# ---------------------------------------------------------
# 2. BACKTEST CONTROLS
# ---------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    top_n = st.number_input(
        "Number of stocks",
        min_value=2,
        max_value=max(2, prices.shape[1]),
        value=min(5, max(2, prices.shape[1])),
        step=1
    )

with col2:
    rebalance_frequency = st.selectbox(
        "Rebalancing frequency",
        ["Monthly", "Quarterly"]
    )

with col3:
    weighting_method = st.selectbox(
        "Portfolio weighting",
        ["Equal Weight", "Score Weighted"]
    )

col4, col5, col6 = st.columns(3)

with col4:
    strategy = st.selectbox(
        "Stock selection strategy",
        [
            "Momentum",
            "Momentum + Low Volatility",
            "Momentum + Low Volatility + Liquidity"
        ]
    )

with col5:
    transaction_cost_bps = st.number_input(
        "Transaction cost (basis points)",
        min_value=0.0,
        max_value=200.0,
        value=10.0,
        step=1.0,
        help="10 basis points = 0.10% of traded notional."
    )

with col6:
    momentum_lookback = st.selectbox(
        "Momentum lookback",
        [63, 126, 252],
        index=1,
        format_func=lambda x: {
            63: "3 Months (63 days)",
            126: "6 Months (126 days)",
            252: "12 Months (252 days)"
        }[x]
    )

volatility_lookback = st.selectbox(
    "Volatility lookback",
    [21, 63, 126],
    index=1,
    format_func=lambda x: {
        21: "1 Month (21 days)",
        63: "3 Months (63 days)",
        126: "6 Months (126 days)"
    }[x]
)

st.divider()

# ---------------------------------------------------------
# 3. DATA INFORMATION
# ---------------------------------------------------------

data_col1, data_col2, data_col3 = st.columns(3)

with data_col1:
    st.metric("Stocks available", prices.shape[1])

with data_col2:
    st.metric("Trading observations", prices.shape[0])

with data_col3:
    st.metric(
        "Data period",
        f"{prices.index.min().strftime('%d %b %Y')} – "
        f"{prices.index.max().strftime('%d %b %Y')}"
    )

st.info(
    "This is a historical simulation using the currently loaded "
    "stock universe. It is not a point-in-time reconstruction of "
    "historical NIFTY 50 constituents."
)

# ---------------------------------------------------------
# 4. RUN BACKTEST
# ---------------------------------------------------------

if st.button(
    "Run Portfolio Backtest",
    type="primary",
    use_container_width=True
):

    try:
        with st.spinner("Running portfolio backtest..."):

            backtest = run_backtest(
                prices=prices,
                volumes=volumes,
                benchmark_prices=benchmark_prices,
                top_n=int(top_n),
                rebalance_frequency=rebalance_frequency,
                weighting_method=weighting_method,
                strategy=strategy,
                transaction_cost_bps=float(
                    transaction_cost_bps
                ),
                momentum_lookback=int(momentum_lookback),
                volatility_lookback=int(volatility_lookback)
            )

            st.session_state["backtest_results"] = backtest

            st.session_state["backtest_config"] = {
                "Top N": int(top_n),
                "Rebalancing": rebalance_frequency,
                "Weighting": weighting_method,
                "Strategy": strategy,
                "Transaction Cost (bps)": float(
                    transaction_cost_bps
                ),
                "Momentum Lookback": int(momentum_lookback),
                "Volatility Lookback": int(
                    volatility_lookback
                )
            }

        st.success("Backtest completed successfully!")

    except Exception as e:
        st.error(f"Backtesting failed: {e}")
        st.exception(e)

# ---------------------------------------------------------
# 5. DISPLAY RESULTS
# ---------------------------------------------------------

backtest = st.session_state.get("backtest_results")

if backtest is None:
    st.info(
        "Configure your strategy above and click "
        "'Run Portfolio Backtest' to view results."
    )
    st.stop()

results = backtest["results"]
metrics = backtest["metrics"]
target_weights = backtest["target_weights"]

st.divider()
st.header("Backtest Performance")

# ---------------------------------------------------------
# 6. PERFORMANCE METRICS
# ---------------------------------------------------------

portfolio_metrics = metrics["Portfolio"]

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "Portfolio Total Return",
        f"{portfolio_metrics['Total Return']:.2%}"
    )

with metric2:
    st.metric(
        "Portfolio CAGR",
        f"{portfolio_metrics['CAGR']:.2%}"
    )

with metric3:
    st.metric(
        "Sharpe Ratio",
        f"{portfolio_metrics['Sharpe Ratio']:.2f}"
        if pd.notna(portfolio_metrics["Sharpe Ratio"])
        else "N/A"
    )

metric4, metric5, metric6 = st.columns(3)

with metric4:
    st.metric(
        "Annualised Volatility",
        f"{portfolio_metrics['Annualised Volatility']:.2%}"
    )

with metric5:
    st.metric(
        "Maximum Drawdown",
        f"{portfolio_metrics['Maximum Drawdown']:.2%}"
    )

with metric6:
    st.metric(
        "Positive Trading Days",
        f"{portfolio_metrics['Positive Days (%)']:.1f}%"
    )

# ---------------------------------------------------------
# 7. BENCHMARK COMPARISON
# ---------------------------------------------------------

st.subheader("Portfolio vs NIFTY 50")

if "Benchmark Equity" in results.columns:

    portfolio_final = results["Portfolio Equity"].iloc[-1]
    benchmark_final = results["Benchmark Equity"].iloc[-1]

    comparison_col1, comparison_col2 = st.columns(2)

    with comparison_col1:
        st.metric(
            "Portfolio Growth",
            f"{(portfolio_final - 1) * 100:.2f}%"
        )

    with comparison_col2:
        st.metric(
            "NIFTY 50 Growth",
            f"{(benchmark_final - 1) * 100:.2f}%"
        )

    equity_data = results[
        ["Portfolio Equity", "Benchmark Equity"]
    ].copy()

    equity_data = equity_data.rename(
        columns={
            "Portfolio Equity": "Strategy Portfolio",
            "Benchmark Equity": "NIFTY 50"
        }
    )

    fig_equity = px.line(
        equity_data,
        x=equity_data.index,
        y=equity_data.columns,
        title="Growth of ₹1 Invested",
        labels={
            "value": "Portfolio Value",
            "index": "Date",
            "variable": "Investment"
        }
    )

    fig_equity.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        legend_title_text=""
    )

    st.plotly_chart(
        fig_equity,
        use_container_width=True
    )

    # Benchmark metrics table
    benchmark_metrics = metrics["NIFTY 50"]

    comparison_table = pd.DataFrame({
        "Metric": [
            "Total Return",
            "CAGR",
            "Annualised Volatility",
            "Sharpe Ratio",
            "Maximum Drawdown",
            "Positive Days (%)"
        ],
        "Strategy Portfolio": [
            portfolio_metrics["Total Return"],
            portfolio_metrics["CAGR"],
            portfolio_metrics["Annualised Volatility"],
            portfolio_metrics["Sharpe Ratio"],
            portfolio_metrics["Maximum Drawdown"],
            portfolio_metrics["Positive Days (%)"]
        ],
        "NIFTY 50": [
            benchmark_metrics["Total Return"],
            benchmark_metrics["CAGR"],
            benchmark_metrics["Annualised Volatility"],
            benchmark_metrics["Sharpe Ratio"],
            benchmark_metrics["Maximum Drawdown"],
            benchmark_metrics["Positive Days (%)"]
        ]
    })

    st.dataframe(
        comparison_table.style.format({
            "Strategy Portfolio": "{:.2%}",
            "NIFTY 50": "{:.2%}"
        }, subset=[
            "Strategy Portfolio",
            "NIFTY 50"
        ]),
        use_container_width=True,
        hide_index=True
    )

else:
    st.warning(
        "Benchmark data is unavailable. Portfolio performance "
        "is displayed without a NIFTY 50 comparison."
    )

# ---------------------------------------------------------
# 8. DRAWDOWN ANALYSIS
# ---------------------------------------------------------

st.subheader("Portfolio Drawdown")

drawdown_columns = ["Portfolio Drawdown"]

if "Benchmark Drawdown" in results.columns:
    drawdown_columns.append("Benchmark Drawdown")

drawdown_data = results[drawdown_columns].copy()

drawdown_data = drawdown_data.rename(
    columns={
        "Portfolio Drawdown": "Strategy Portfolio",
        "Benchmark Drawdown": "NIFTY 50"
    }
)

fig_drawdown = px.line(
    drawdown_data,
    x=drawdown_data.index,
    y=drawdown_data.columns,
    title="Historical Drawdown",
    labels={
        "value": "Drawdown",
        "index": "Date",
        "variable": "Portfolio"
    }
)

fig_drawdown.update_layout(
    template="plotly_dark",
    hovermode="x unified",
    yaxis_tickformat=".0%"
)

st.plotly_chart(
    fig_drawdown,
    use_container_width=True
)

# ---------------------------------------------------------
# 9. TURNOVER AND TRANSACTION COSTS
# ---------------------------------------------------------

st.subheader("Portfolio Turnover")

turnover_col1, turnover_col2 = st.columns(2)

with turnover_col1:
    st.metric(
        "Total Rebalances",
        metrics["Total Rebalances"]
    )

with turnover_col2:
    st.metric(
        "Average Daily Turnover",
        f"{metrics['Average Daily Turnover']:.2%}"
    )

rebalance_data = results[
    results["Turnover"] > 0
][["Turnover", "Transaction Cost"]].copy()

if not rebalance_data.empty:

    fig_turnover = px.bar(
        rebalance_data,
        x=rebalance_data.index,
        y="Turnover",
        title="Portfolio Turnover at Rebalancing Dates",
        labels={
            "Turnover": "Traded Notional / Portfolio Value",
            "index": "Rebalance Date"
        }
    )

    fig_turnover.update_layout(
        template="plotly_dark"
    )

    st.plotly_chart(
        fig_turnover,
        use_container_width=True
    )

# ---------------------------------------------------------
# 10. PORTFOLIO HOLDINGS
# ---------------------------------------------------------

st.subheader("Latest Portfolio Holdings")

latest_weights = target_weights.loc[
    target_weights.sum(axis=1) > 0
]

if not latest_weights.empty:

    holdings = latest_weights.iloc[-1]
    holdings = holdings[holdings > 0].sort_values(
        ascending=False
    )

    holdings_df = pd.DataFrame({
        "Ticker": holdings.index,
        "Portfolio Weight": holdings.values
    })

    holdings_df["Portfolio Weight"] = (
        holdings_df["Portfolio Weight"].map(
            lambda x: f"{x:.2%}"
        )
    )

    st.dataframe(
        holdings_df,
        use_container_width=True,
        hide_index=True
    )

    fig_holdings = px.pie(
        names=holdings.index,
        values=holdings.values,
        title="Latest Portfolio Allocation",
        hole=0.45
    )

    fig_holdings.update_layout(
        template="plotly_dark"
    )

    st.plotly_chart(
        fig_holdings,
        use_container_width=True
    )

else:
    st.warning(
        "No eligible portfolio holdings were generated. "
        "Try a longer historical data period."
    )

# ---------------------------------------------------------
# 11. DOWNLOAD RESULTS
# ---------------------------------------------------------

st.subheader("Export Backtest Results")

csv_data = results.to_csv().encode("utf-8")

st.download_button(
    label="Download Backtest Results (CSV)",
    data=csv_data,
    file_name="portfolio_backtest_results.csv",
    mime="text/csv",
    use_container_width=True
)

weights_csv = target_weights.to_csv().encode("utf-8")

st.download_button(
    label="Download Historical Portfolio Weights (CSV)",
    data=weights_csv,
    file_name="historical_portfolio_weights.csv",
    mime="text/csv",
    use_container_width=True
)

# ---------------------------------------------------------
# 12. METHODOLOGY
# ---------------------------------------------------------

with st.expander("Backtesting Methodology and Limitations"):

    st.markdown("""
    **Stock selection**

    Stocks are ranked using historical momentum, optionally combined
    with low volatility and liquidity.

    **Portfolio construction**

    The top N stocks are selected at each rebalance date. Portfolio
    weights are either equal-weighted or proportional to positive
    factor scores.

    **Rebalancing**

    The strategy rebalances monthly or quarterly. Signals are calculated
    using data available at the rebalance-date close, and the new
    portfolio weights apply from the next trading day.

    **Transaction costs**

    Transaction costs are estimated using the specified basis-point
    rate multiplied by the absolute change in portfolio weights.

    **Benchmark**

    The portfolio is compared with the downloaded NIFTY 50 benchmark.

    **Important limitations**

    - This is a historical simulation, not a live trading system.
    - The current universe is a development subset, not a verified
      historical NIFTY 50 constituent database.
    - Survivorship bias may affect results.
    - The backtest does not currently model taxes, market impact,
      bid-ask spreads, short selling or liquidity constraints.
    - The benchmark comparison uses the available downloaded price
      series and may not represent a fully point-in-time comparison.
    - The Sharpe ratio assumes a zero risk-free rate.
    - Results are sensitive to the selected period, universe,
      lookback window and transaction-cost assumptions.
    """)