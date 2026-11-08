
import streamlit as st
import pandas as pd
import plotly.express as px

from modules.factor_engine import (
    calculate_all_factors,
    normalize_factors
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.title("Equity Factor Analysis")

st.caption(
    "Construction, normalization and comparison "
    "of systematic equity factors."
)

st.divider()


# =========================================================
# LOAD MARKET DATA
# =========================================================

prices = st.session_state.get("market_prices")
returns = st.session_state.get("market_returns")
volumes = st.session_state.get("market_volumes")

if (
    prices is None
    or returns is None
    or volumes is None
):

    st.warning(
        "Market data not found. Please visit the "
        "Market Universe page and download market data first."
    )

    st.stop()


# =========================================================
# FACTOR DEFINITIONS
# =========================================================

factor_definitions = {

    "Momentum": (
        "Measures historical stock price performance "
        "over different lookback periods."
    ),

    "Value": (
        "Evaluates relative valuation using P/E, P/B, "
        "earnings yield and book-to-price."
    ),

    "Quality": (
        "Measures profitability and financial strength "
        "using ROE, ROA, profit margins and leverage."
    ),

    "Low Volatility": (
        "Identifies stocks with relatively lower "
        "historical return volatility."
    ),

    "Growth": (
        "Evaluates revenue and earnings growth characteristics."
    ),

    "Liquidity": (
        "Measures trading activity using average volume "
        "and average daily traded value."
    )
}


# =========================================================
# RESEARCH CONFIGURATION
# =========================================================

st.subheader("Research Configuration")

col1, col2 = st.columns(2)

with col1:

    selected_factors = st.multiselect(
        "Select factor categories",
        list(factor_definitions.keys()),
        default=[
            "Momentum",
            "Low Volatility",
            "Liquidity"
        ]
    )

with col2:

    normalization = st.selectbox(
        "Normalization Method",
        [
            "Z-Score",
            "Percentile Ranking",
            "Min-Max Scaling"
        ]
    )


# =========================================================
# FUNDAMENTAL DATA
# =========================================================

# Real fundamental data will be integrated separately.
# Do not substitute invented financial values.

tickers = prices.columns.tolist()

fundamentals = pd.DataFrame({
    "Ticker": tickers
})


# =========================================================
# CALCULATE FACTORS
# =========================================================

st.divider()

st.subheader("Factor Calculation")

factor_data = calculate_all_factors(
    prices=prices,
    returns=returns,
    volumes=volumes,
    fundamentals=fundamentals
)

if factor_data.empty:

    st.error("Unable to calculate factor data.")

    st.stop()


# =========================================================
# FACTOR CATEGORY MAPPING
# =========================================================

factor_mapping = {

    "Momentum": [
        "Momentum_1M",
        "Momentum_3M",
        "Momentum_6M",
        "Momentum_12M"
    ],

    "Value": [
        "PE_Ratio",
        "PB_Ratio",
        "Earnings_Yield",
        "Book_to_Price"
    ],

    "Quality": [
        "ROE",
        "ROA",
        "Profit_Margin",
        "Debt_to_Equity",
        "Leverage_Score"
    ],

    "Low Volatility": [
        "Volatility_30D",
        "Volatility_60D",
        "Volatility_90D",
        "Volatility_1Y"
    ],

    "Growth": [
        "Revenue_Growth",
        "Earnings_Growth"
    ],

    "Liquidity": [
        "Average_Daily_Traded_Value",
        "Average_Daily_Volume"
    ]
}


# =========================================================
# SELECT FACTOR COLUMNS
# =========================================================

selected_columns = []

for factor in selected_factors:

    selected_columns.extend(
        factor_mapping[factor]
    )

selected_columns = [
    column for column in selected_columns
    if column in factor_data.columns
]

if not selected_columns:

    st.warning("Select at least one available factor.")

    st.stop()


selected_factor_data = factor_data[selected_columns].copy()


# =========================================================
# NORMALIZE FACTORS
# =========================================================

normalized_data = normalize_factors(
    selected_factor_data,
    method=normalization
)


# =========================================================
# KPI SUMMARY
# =========================================================

st.divider()

st.subheader("Research Overview")

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Stocks Analysed",
    len(factor_data)
)

k2.metric(
    "Factor Measures",
    len(selected_columns)
)

k3.metric(
    "Available Measures",
    int(factor_data[selected_columns].notna().any().sum())
)

k4.metric(
    "Normalization",
    normalization
)


# =========================================================
# RAW FACTOR DATA
# =========================================================

st.divider()

st.subheader("Raw Factor Data")

st.caption(
    "Values are calculated from the selected historical "
    "market data. Fundamental measures remain unavailable "
    "until a fundamental data source is connected."
)

st.dataframe(
    selected_factor_data.style.format("{:.4f}"),
    use_container_width=True
)


# =========================================================
# NORMALIZED FACTOR DATA
# =========================================================

st.divider()

st.subheader("Normalized Factor Data")

st.dataframe(
    normalized_data.style.format("{:.4f}"),
    use_container_width=True
)


# =========================================================
# FACTOR COMPARISON
# =========================================================

st.divider()

st.subheader("Factor Comparison")

available_columns = [
    column for column in normalized_data.columns
    if normalized_data[column].notna().any()
]

if available_columns:

    chart_factor = st.selectbox(
        "Select factor measure for comparison",
        available_columns
    )

    chart_data = (
        normalized_data[[chart_factor]]
        .dropna()
        .sort_values(chart_factor, ascending=False)
        .reset_index()
    )

    chart = px.bar(
        chart_data,
        x="Ticker",
        y=chart_factor,
        title=f"{chart_factor} — Cross-sectional Comparison",
        template="plotly_dark"
    )

    chart.update_layout(
        xaxis_title="Stock",
        yaxis_title="Normalized Factor Score",
        height=450
    )

    st.plotly_chart(
        chart,
        use_container_width=True
    )

else:

    st.info(
        "No non-missing factor values are available "
        "for the selected categories."
    )


# =========================================================
# FACTOR DEFINITIONS
# =========================================================

st.divider()

st.subheader("Factor Methodology")

for factor in selected_factors:

    with st.expander(factor):

        st.write(factor_definitions[factor])

        st.write(
            "**Measures:**",
            ", ".join(factor_mapping[factor])
        )


# =========================================================
# DATA EXPORT
# =========================================================

st.divider()

st.subheader("Export Research Data")

export_col1, export_col2 = st.columns(2)

with export_col1:

    raw_csv = selected_factor_data.to_csv().encode("utf-8")

    st.download_button(
        "Download Raw Factor Data",
        data=raw_csv,
        file_name="raw_factor_data.csv",
        mime="text/csv"
    )

with export_col2:

    normalized_csv = normalized_data.to_csv().encode("utf-8")

    st.download_button(
        "Download Normalized Factor Data",
        data=normalized_csv,
        file_name="normalized_factor_data.csv",
        mime="text/csv"
    )


# =========================================================
# RESEARCH NOTES
# =========================================================

st.divider()

st.info(
    """
    Research Note:

    1. Momentum, volatility and liquidity use historical market data.
    2. Value, Quality and Growth require genuine fundamental data.
    3. Normalization is cross-sectional across the selected stocks.
    4. Normalized scores are not investment recommendations.
    5. Factor direction and composite scoring will be handled
       in the Stock Ranking module.
    """
)
