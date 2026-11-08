
import streamlit as st
import pandas as pd
import plotly.express as px

from modules.universe_engine import (
    get_stock_universe,
    get_universe_summary
)

from modules.data_loader import (
    download_market_data,
    download_benchmark
)


st.title("📊 Market Universe")
st.caption("Live market data | Equity universe | Historical price analytics")

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    universe_name = st.selectbox(
        "Stock Universe",
        ["NIFTY 50", "NIFTY NEXT 50", "NIFTY 100"]
    )

with col2:
    period_option = st.selectbox(
        "Historical Period",
        ["1 Year", "2 Years", "3 Years", "5 Years"]
    )

with col3:
    frequency = st.selectbox(
        "Data Frequency",
        ["Daily", "Weekly", "Monthly"]
    )

period_map = {
    "1 Year": "1y",
    "2 Years": "2y",
    "3 Years": "3y",
    "5 Years": "5y"
}

period = period_map[period_option]

st.divider()

# --------------------------------------------------
# DATA DOWNLOAD
# --------------------------------------------------

if st.button("Download Market Data", type="primary"):

    try:
        tickers = get_stock_universe(universe_name)

        with st.spinner("Downloading historical market data..."):

            market_data = download_market_data(
                tickers,
                period=period
            )

            benchmark_prices, benchmark_returns = download_benchmark(
                "^NSEI",
                period=period
            )

            prices = market_data["prices"]
            volumes = market_data["volumes"]
            returns = market_data["returns"]

            if frequency == "Weekly":
                prices = prices.resample("W-FRI").last()
                volumes = volumes.resample("W-FRI").sum()

            elif frequency == "Monthly":
                prices = prices.resample("ME").last()
                volumes = volumes.resample("ME").sum()

            returns = prices.pct_change(fill_method=None)

            summary = get_universe_summary(prices)

            st.session_state["market_prices"] = prices
            st.session_state["market_returns"] = returns
            st.session_state["market_volumes"] = volumes
            st.session_state["market_summary"] = summary
            st.session_state["benchmark_prices"] = benchmark_prices
            st.session_state["benchmark_returns"] = benchmark_returns
            st.session_state["loaded_universe"] = universe_name

        st.success("Market data downloaded successfully!")

    except Exception as error:
        st.error(f"Data download failed: {error}")


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

if "market_prices" in st.session_state:

    prices = st.session_state["market_prices"]
    returns = st.session_state["market_returns"]
    volumes = st.session_state["market_volumes"]
    summary = st.session_state["market_summary"]

    st.subheader("Market Data Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Stocks Loaded",
        len(prices.columns)
    )

    col2.metric(
        "Price Observations",
        len(prices)
    )

    col3.metric(
        "Latest Trading Date",
        str(prices.index.max().date())
    )

    col4.metric(
        "Data Coverage",
        f"{prices.notna().mean().mean() * 100:.1f}%"
    )

    st.divider()

    # --------------------------------------------------
    # STOCK PRICE TABLE
    # --------------------------------------------------

    st.subheader("Stock Price Overview")

    latest_prices = prices.ffill().iloc[-1]

    price_table = pd.DataFrame({
        "Ticker": latest_prices.index,
        "Latest Price": latest_prices.values
    })

    price_table["Latest Price"] = price_table["Latest Price"].round(2)

    st.dataframe(
        price_table,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # PRICE CHART
    # --------------------------------------------------

    st.subheader("Historical Stock Price")

    selected_stock = st.selectbox(
        "Select Stock",
        prices.columns.tolist()
    )

    chart_data = prices[[selected_stock]].dropna().reset_index()

    date_column = chart_data.columns[0]

    fig = px.line(
        chart_data,
        x=date_column,
        y=selected_stock,
        title=f"{selected_stock} Historical Price"
    )

    fig.update_layout(
        template="plotly_dark",
        height=420,
        xaxis_title="Date",
        yaxis_title="Adjusted Price (INR)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------
    # RETURNS
    # --------------------------------------------------

    st.subheader("Daily / Periodic Returns")

    return_data = returns[selected_stock].dropna()

    fig_returns = px.line(
        x=return_data.index,
        y=return_data.values * 100,
        title=f"{selected_stock} Returns"
    )

    fig_returns.update_layout(
        template="plotly_dark",
        height=350,
        xaxis_title="Date",
        yaxis_title="Return (%)"
    )

    st.plotly_chart(fig_returns, use_container_width=True)

    # --------------------------------------------------
    # VOLUME
    # --------------------------------------------------

    st.subheader("Trading Volume")

    volume_data = volumes[selected_stock].dropna()

    fig_volume = px.bar(
        x=volume_data.index,
        y=volume_data.values,
        title=f"{selected_stock} Trading Volume"
    )

    fig_volume.update_layout(
        template="plotly_dark",
        height=350,
        xaxis_title="Date",
        yaxis_title="Volume"
    )

    st.plotly_chart(fig_volume, use_container_width=True)

    # --------------------------------------------------
    # DATA QUALITY
    # --------------------------------------------------

    st.subheader("Data Quality Analysis")

    missing = prices.isna().mean() * 100

    quality = pd.DataFrame({
        "Ticker": missing.index,
        "Missing Data (%)": missing.values
    })

    quality["Missing Data (%)"] = quality["Missing Data (%)"].round(2)

    st.dataframe(
        quality,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------

    st.subheader("NIFTY 50 Benchmark")

    benchmark_prices = st.session_state["benchmark_prices"]

    benchmark_chart = px.line(
        x=benchmark_prices.index,
        y=benchmark_prices.values,
        title="NIFTY 50 Historical Price"
    )

    benchmark_chart.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Date",
        yaxis_title="Index Level"
    )

    st.plotly_chart(
        benchmark_chart,
        use_container_width=True
    )

else:

    st.info(
        "Select your stock universe and click "
        "'Download Market Data' to begin."
    )
