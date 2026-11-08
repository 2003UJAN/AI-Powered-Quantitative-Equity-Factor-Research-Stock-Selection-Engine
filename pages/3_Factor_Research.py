
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# PAGE HEADER
# =========================================================

st.title("Factor Research & Performance Evaluation")

st.caption(
    "Historical factor evaluation using Information Coefficient, "
    "quintile analysis and long-short portfolio returns."
)

st.divider()


# =========================================================
# LOAD MARKET DATA
# =========================================================

prices = st.session_state.get("market_prices")

if prices is None or prices.empty:

    st.warning(
        "Market data not found. Please download data "
        "from the Market Universe page first."
    )

    st.stop()

prices = prices.sort_index()

if not isinstance(prices.index, pd.DatetimeIndex):
    prices.index = pd.to_datetime(prices.index)


# =========================================================
# RESEARCH CONFIGURATION
# =========================================================

st.subheader("Research Configuration")

col1, col2, col3 = st.columns(3)

with col1:

    factor_name = st.selectbox(
        "Research Factor",
        [
            "1-Month Momentum",
            "3-Month Momentum",
            "6-Month Momentum",
            "12-Month Momentum"
        ]
    )

with col2:

    forward_period = st.selectbox(
        "Forward Return Period",
        [5, 10, 21, 42, 63],
        index=2,
        format_func=lambda x: f"{x} Trading Days"
    )

with col3:

    rebalance_frequency = st.selectbox(
        "Research Frequency",
        ["Monthly", "Weekly", "Daily"],
        index=0
    )


lookback_mapping = {
    "1-Month Momentum": 21,
    "3-Month Momentum": 63,
    "6-Month Momentum": 126,
    "12-Month Momentum": 252
}

lookback_period = lookback_mapping[factor_name]


# =========================================================
# CREATE HISTORICAL FACTOR SIGNALS
# =========================================================

st.divider()

st.subheader("Historical Factor Construction")

# Momentum signal at date t:
# Price(t) / Price(t-lookback) - 1

factor_history = (
    prices / prices.shift(lookback_period) - 1
)

# Forward return:
# Price(t + N) / Price(t) - 1
#
# shift(-N) ensures that future returns are aligned
# with the factor signal known at date t.

forward_returns = (
    prices.shift(-forward_period) / prices - 1
)

# Remove early dates where factor history is unavailable.

factor_history = factor_history.iloc[lookback_period:]
forward_returns = forward_returns.reindex(
    factor_history.index
)


# =========================================================
# SELECT RESEARCH FREQUENCY
# =========================================================

if rebalance_frequency == "Monthly":

    factor_history = factor_history.resample("ME").last()
    forward_returns = forward_returns.reindex(
        factor_history.index
    )

elif rebalance_frequency == "Weekly":

    factor_history = factor_history.resample("W-FRI").last()
    forward_returns = forward_returns.reindex(
        factor_history.index
    )


# =========================================================
# INFORMATION COEFFICIENT
# =========================================================

st.divider()

st.subheader("Information Coefficient Analysis")

# Pearson IC measures linear correlation.
# Rank IC measures Spearman rank correlation.

pearson_ic = factor_history.corrwith(
    forward_returns,
    axis=1,
    method="pearson"
)

rank_ic = factor_history.corrwith(
    forward_returns,
    axis=1,
    method="spearman"
)

ic_data = pd.DataFrame({
    "Pearson_IC": pearson_ic,
    "Rank_IC": rank_ic
}).dropna()

if ic_data.empty:

    st.error(
        "Insufficient observations to calculate IC. "
        "Try a shorter momentum lookback or download more history."
    )

    st.stop()


# =========================================================
# IC STATISTICS
# =========================================================

mean_ic = ic_data["Pearson_IC"].mean()

mean_rank_ic = ic_data["Rank_IC"].mean()

ic_std = ic_data["Rank_IC"].std()

icir = (
    mean_rank_ic / ic_std
    if pd.notna(ic_std) and ic_std > 0
    else np.nan
)

positive_ic = (
    ic_data["Rank_IC"] > 0
).mean() * 100

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Mean Pearson IC",
    f"{mean_ic:.4f}"
)

k2.metric(
    "Mean Rank IC",
    f"{mean_rank_ic:.4f}"
)

k3.metric(
    "IC Information Ratio",
    f"{icir:.4f}" if pd.notna(icir) else "N/A"
)

k4.metric(
    "Positive Rank IC",
    f"{positive_ic:.2f}%"
)


# =========================================================
# IC TIME SERIES
# =========================================================

st.subheader("Information Coefficient Time Series")

ic_chart = go.Figure()

ic_chart.add_trace(
    go.Scatter(
        x=ic_data.index,
        y=ic_data["Pearson_IC"],
        mode="lines",
        name="Pearson IC"
    )
)

ic_chart.add_trace(
    go.Scatter(
        x=ic_data.index,
        y=ic_data["Rank_IC"],
        mode="lines",
        name="Spearman Rank IC"
    )
)

ic_chart.add_hline(
    y=0,
    line_dash="dash"
)

ic_chart.update_layout(
    template="plotly_dark",
    height=450,
    xaxis_title="Research Date",
    yaxis_title="Information Coefficient",
    hovermode="x unified"
)

st.plotly_chart(
    ic_chart,
    use_container_width=True
)


# =========================================================
# QUINTILE ANALYSIS
# =========================================================

st.divider()

st.subheader("Quintile Portfolio Analysis")

st.write(
    "Stocks are sorted into five groups based on their "
    "factor scores. Quintile 5 contains the highest "
    "momentum stocks."
)


def calculate_quintile_returns(factor_data, return_data):

    results = []

    dates = factor_data.index.intersection(
        return_data.index
    )

    for date in dates:

        scores = factor_data.loc[date].dropna()

        returns_for_date = return_data.loc[date]

        common = scores.index.intersection(
            returns_for_date.dropna().index
        )

        scores = scores.loc[common]
        returns_for_date = returns_for_date.loc[common]

        if len(common) < 5:
            continue

        ranked_scores = scores.rank(
            method="first",
            pct=True
        )

        quintiles = pd.cut(
            ranked_scores,
            bins=[
                0,
                0.2,
                0.4,
                0.6,
                0.8,
                1.0
            ],
            labels=[
                "Q1",
                "Q2",
                "Q3",
                "Q4",
                "Q5"
            ],
            include_lowest=True
        )

        temporary = pd.DataFrame({
            "Quintile": quintiles,
            "Forward_Return": returns_for_date
        })

        grouped = temporary.groupby(
            "Quintile",
            observed=False
        )["Forward_Return"].mean()

        row = {
            "Date": date,
            "Q1": grouped.get("Q1", np.nan),
            "Q2": grouped.get("Q2", np.nan),
            "Q3": grouped.get("Q3", np.nan),
            "Q4": grouped.get("Q4", np.nan),
            "Q5": grouped.get("Q5", np.nan)
        }

        results.append(row)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).set_index("Date")


quintile_returns = calculate_quintile_returns(
    factor_history,
    forward_returns
)


if quintile_returns.empty:

    st.warning(
        "Insufficient stocks for quintile analysis."
    )

else:

    average_quintile_returns = (
        quintile_returns.mean() * 100
    )

    quintile_summary = pd.DataFrame({
        "Quintile": average_quintile_returns.index,
        "Average Forward Return (%)":
            average_quintile_returns.values
    })

    st.dataframe(
        quintile_summary,
        use_container_width=True,
        hide_index=True
    )

    quintile_chart = px.bar(
        quintile_summary,
        x="Quintile",
        y="Average Forward Return (%)",
        title="Average Forward Returns by Quintile",
        template="plotly_dark"
    )

    quintile_chart.update_layout(
        height=400
    )

    st.plotly_chart(
        quintile_chart,
        use_container_width=True
    )


# =========================================================
# LONG-SHORT PORTFOLIO
# =========================================================

st.divider()

st.subheader("Long-Short Portfolio Analysis")

if not quintile_returns.empty:

    # Long highest-factor quintile.
    # Short lowest-factor quintile.

    long_short_returns = (
        quintile_returns["Q5"]
        - quintile_returns["Q1"]
    ).dropna()

    average_long_short = (
        long_short_returns.mean()
    )

    if rebalance_frequency == "Monthly":
        periods_per_year = 12
    elif rebalance_frequency == "Weekly":
        periods_per_year = 52
    else:
        periods_per_year = 252

    annualized_long_short = (
        (1 + average_long_short) ** periods_per_year - 1
        if average_long_short > -1
        else np.nan
    )

    ls1, ls2, ls3 = st.columns(3)

    ls1.metric(
        "Average Long-Short Return",
        f"{average_long_short * 100:.3f}%"
    )

    ls2.metric(
        "Annualized Long-Short Return",
        (
            f"{annualized_long_short * 100:.2f}%"
            if pd.notna(annualized_long_short)
            else "N/A"
        )
    )

    ls3.metric(
        "Research Observations",
        len(long_short_returns)
    )

    # Cumulative compounded returns.
    cumulative_long_short = (
        1 + long_short_returns
    ).cumprod() - 1

    long_short_chart = px.line(
        x=cumulative_long_short.index,
        y=cumulative_long_short.values * 100,
        title="Cumulative Long-Short Return",
        labels={
            "x": "Date",
            "y": "Cumulative Return (%)"
        },
        template="plotly_dark"
    )

    long_short_chart.add_hline(
        y=0,
        line_dash="dash"
    )

    long_short_chart.update_layout(
        height=450
    )

    st.plotly_chart(
        long_short_chart,
        use_container_width=True
    )


# =========================================================
# IC INFORMATION TABLE
# =========================================================

st.divider()

st.subheader("Detailed IC Statistics")

ic_statistics = pd.DataFrame({
    "Metric": [
        "Mean Pearson IC",
        "Mean Spearman Rank IC",
        "Rank IC Standard Deviation",
        "IC Information Ratio",
        "Positive Rank IC Frequency",
        "Number of Observations"
    ],

    "Value": [
        mean_ic,
        mean_rank_ic,
        ic_std,
        icir,
        positive_ic,
        len(ic_data)
    ]
})

st.dataframe(
    ic_statistics,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# EXPORT RESEARCH RESULTS
# =========================================================

st.divider()

st.subheader("Export Research Results")

col1, col2, col3 = st.columns(3)

with col1:

    st.download_button(
        "Download IC Data",
        data=ic_data.to_csv().encode("utf-8"),
        file_name="factor_ic_analysis.csv",
        mime="text/csv"
    )

with col2:

    st.download_button(
        "Download Quintile Data",
        data=quintile_returns.to_csv().encode("utf-8"),
        file_name="factor_quintile_returns.csv",
        mime="text/csv"
    )

with col3:

    if not quintile_returns.empty:

        st.download_button(
            "Download Long-Short Data",
            data=long_short_returns.to_csv().encode("utf-8"),
            file_name="long_short_returns.csv",
            mime="text/csv"
        )


# =========================================================
# METHODOLOGY NOTES
# =========================================================

st.divider()

with st.expander("Research Methodology & Limitations"):

    st.markdown(
        """
        **Factor construction**

        Momentum is calculated using the historical price
        relative to its price at the selected lookback period.

        **Forward returns**

        Forward returns are aligned to the factor observation
        date. The factor uses prices available at date t, while
        the outcome uses prices at date t + N.

        **Information Coefficient**

        Pearson IC measures linear correlation between factor
        scores and forward returns.

        Spearman Rank IC measures rank correlation.

        **Quintile analysis**

        Stocks are ranked into five groups. Q5 contains the
        highest factor scores and Q1 the lowest.

        **Long-short strategy**

        Long Q5 and short Q1, using equal-weighted average
        forward returns.

        **Important limitations**

        - The current universe contains only a small development
          subset of NIFTY 50 stocks.
        - The quintile results are not yet suitable for
          investment conclusions.
        - Forward return windows overlap when observations are
          sampled more frequently than the forward horizon.
        - Transaction costs, slippage, survivorship bias and
          historical index constituents are not yet incorporated.
        - The reported annualized long-short return is a simple
          annualization of the average periodic return, not a
          validated portfolio CAGR.
        """
    )


st.caption(
    "Quantitative Equity Factor Research Engine | "
    "Historical factor evaluation prototype"
)
