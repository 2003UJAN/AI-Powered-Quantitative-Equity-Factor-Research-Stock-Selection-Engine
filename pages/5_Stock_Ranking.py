
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from modules.factor_engine import calculate_all_factors

from modules.ranking_engine import (
    DEFAULT_WEIGHTS,
    calculate_factor_scores,
    calculate_composite_score,
    calculate_factor_contributions,
    rank_stocks
)


# =========================================================
# PAGE HEADER
# =========================================================

st.title("Quantitative Stock Ranking Engine")

st.caption(
    "Multi-factor equity ranking using momentum, value, "
    "quality, growth, volatility and liquidity."
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
        "Market data not found. Please visit Market Universe "
        "and download market data first."
    )

    st.stop()


# =========================================================
# PREPARE DATA
# =========================================================

prices = prices.copy().sort_index()
returns = returns.copy().sort_index()
volumes = volumes.copy().sort_index()

tickers = prices.columns.tolist()

# Fundamental data will be connected separately.
# No financial values are fabricated.

fundamentals = pd.DataFrame({
    "Ticker": tickers
})


# =========================================================
# FACTOR CALCULATION
# =========================================================

st.subheader("Factor Calculation")

try:

    factor_data = calculate_all_factors(
        prices=prices,
        returns=returns,
        volumes=volumes,
        fundamentals=fundamentals
    )

    factor_scores = calculate_factor_scores(
        factor_data
    )

except Exception as error:

    st.error(
        f"Factor calculation failed: {error}"
    )

    st.stop()


# =========================================================
# AVAILABLE FACTORS
# =========================================================

available_factors = [
    factor
    for factor in DEFAULT_WEIGHTS
    if (
        factor in factor_scores.columns
        and factor_scores[factor].notna().any()
    )
]

unavailable_factors = [
    factor
    for factor in DEFAULT_WEIGHTS
    if factor not in available_factors
]


st.subheader("Factor Availability")

availability_col1, availability_col2 = st.columns(2)

with availability_col1:

    st.metric(
        "Available Factor Categories",
        len(available_factors)
    )

with availability_col2:

    st.metric(
        "Unavailable Categories",
        len(unavailable_factors)
    )


if unavailable_factors:

    st.warning(
        "The following categories need genuine fundamental "
        "data or additional observations: "
        + ", ".join(unavailable_factors)
    )


# =========================================================
# FACTOR WEIGHT CONFIGURATION
# =========================================================

st.divider()

st.subheader("Factor Weight Configuration")

st.write(
    "Adjust the importance of each factor category. "
    "Weights are automatically normalized to sum to 100% "
    "across available categories."
)

weight_col1, weight_col2 = st.columns(2)

weights = {}

for index, factor in enumerate(DEFAULT_WEIGHTS):

    column = (
        weight_col1
        if index % 2 == 0
        else weight_col2
    )

    with column:

        weights[factor] = st.slider(
            factor,
            min_value=0,
            max_value=100,
            value=int(DEFAULT_WEIGHTS[factor] * 100),
            step=5,
            disabled=factor not in available_factors,
            key=f"ranking_weight_{factor}"
        ) / 100


active_weights = {
    factor: weight
    for factor, weight in weights.items()
    if factor in available_factors
}

active_total = sum(active_weights.values())

if active_total <= 0:

    st.error(
        "Please assign a positive weight to at least "
        "one available factor."
    )

    st.stop()


normalized_weights = {
    factor: weight / active_total
    for factor, weight in active_weights.items()
}


# =========================================================
# DISPLAY NORMALIZED WEIGHTS
# =========================================================

st.markdown("**Effective factor weights**")

weight_display = pd.DataFrame({
    "Factor": normalized_weights.keys(),
    "Effective Weight (%)": [
        value * 100
        for value in normalized_weights.values()
    ]
})

st.dataframe(
    weight_display,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# COMPOSITE RANKING
# =========================================================

st.divider()

st.subheader("Generate Stock Rankings")

try:

    ranking = rank_stocks(
        factor_data=factor_data,
        weights=weights,
        require_all_factors=False
    )

except Exception as error:

    st.error(
        f"Unable to generate rankings: {error}"
    )

    st.stop()


# =========================================================
# RANKING SUMMARY
# =========================================================

valid_ranking = ranking.dropna(
    subset=["Composite_Score"]
)

if valid_ranking.empty:

    st.warning(
        "No valid composite scores could be calculated."
    )

    st.stop()


top_stock = valid_ranking.index[0]

top_score = valid_ranking.iloc[0]["Composite_Score"]

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Stocks Ranked",
    len(valid_ranking)
)

k2.metric(
    "Active Factors",
    len(available_factors)
)

k3.metric(
    "Top Ranked Stock",
    str(top_stock)
)

k4.metric(
    "Top Composite Score",
    f"{top_score:.2f}/100"
)


# =========================================================
# COMPLETE STOCK RANKING TABLE
# =========================================================

st.divider()

st.subheader("Stock Ranking Table")

display_columns = [
    factor
    for factor in available_factors
]

display_columns += [
    "Composite_Score",
    "Rank"
]

ranking_display = ranking[display_columns].copy()

ranking_display.index.name = "Ticker"

st.dataframe(
    ranking_display.style.format("{:.2f}"),
    use_container_width=True
)


# =========================================================
# TOP N STOCKS
# =========================================================

st.divider()

st.subheader("Top Ranked Stocks")

top_n = st.slider(
    "Number of stocks to display",
    min_value=1,
    max_value=len(valid_ranking),
    value=min(5, len(valid_ranking))
)

top_stocks = valid_ranking.head(top_n)

top_chart_data = (
    top_stocks[["Composite_Score"]]
    .reset_index()
)

top_chart = px.bar(
    top_chart_data,
    x="Ticker",
    y="Composite_Score",
    title="Top Ranked Stocks by Composite Score",
    template="plotly_dark",
    text="Composite_Score"
)

top_chart.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside"
)

top_chart.update_layout(
    height=450,
    yaxis_title="Composite Score (0–100)"
)

st.plotly_chart(
    top_chart,
    use_container_width=True
)


# =========================================================
# INDIVIDUAL STOCK FACTOR PROFILE
# =========================================================

st.divider()

st.subheader("Individual Stock Factor Profile")

selected_stock = st.selectbox(
    "Select stock",
    valid_ranking.index.tolist()
)

stock_profile = factor_scores.loc[
    selected_stock,
    available_factors
]

profile_data = pd.DataFrame({
    "Factor": stock_profile.index,
    "Score": stock_profile.values
}).dropna()

profile_chart = px.bar(
    profile_data,
    x="Factor",
    y="Score",
    title=f"Factor Profile: {selected_stock}",
    template="plotly_dark"
)

profile_chart.update_layout(
    height=400,
    yaxis_title="Factor Score (0–100)"
)

st.plotly_chart(
    profile_chart,
    use_container_width=True
)


# =========================================================
# FACTOR CONTRIBUTION ANALYSIS
# =========================================================

st.divider()

st.subheader("Factor Contribution Analysis")

contributions = calculate_factor_contributions(
    factor_scores,
    weights=weights
)

stock_contribution = contributions.loc[
    selected_stock
].dropna()

if not stock_contribution.empty:

    contribution_data = pd.DataFrame({
        "Factor": stock_contribution.index,
        "Contribution": stock_contribution.values
    })

    contribution_chart = px.bar(
        contribution_data,
        x="Factor",
        y="Contribution",
        title=f"Weighted Contributions: {selected_stock}",
        template="plotly_dark"
    )

    contribution_chart.update_layout(
        height=400,
        yaxis_title="Contribution to Composite Score"
    )

    st.plotly_chart(
        contribution_chart,
        use_container_width=True
    )


# =========================================================
# FACTOR SCORE HEATMAP
# =========================================================

st.divider()

st.subheader("Cross-Sectional Factor Heatmap")

heatmap_data = factor_scores[
    available_factors
].copy()

heatmap_chart = px.imshow(
    heatmap_data,
    text_auto=".1f",
    aspect="auto",
    color_continuous_scale="RdYlGn",
    title="Factor Scores Across Stocks",
    labels={
        "x": "Factor",
        "y": "Stock",
        "color": "Score"
    }
)

heatmap_chart.update_layout(
    height=500
)

st.plotly_chart(
    heatmap_chart,
    use_container_width=True
)


# =========================================================
# EXPORT RANKING RESULTS
# =========================================================

st.divider()

st.subheader("Export Ranking Results")

export_data = ranking.copy()

export_data.index.name = "Ticker"

st.download_button(
    "Download Complete Stock Rankings",
    data=export_data.to_csv().encode("utf-8"),
    file_name="quantitative_stock_rankings.csv",
    mime="text/csv",
    use_container_width=True
)


# =========================================================
# SAVE RANKING IN SESSION STATE
# =========================================================

st.session_state["stock_ranking"] = ranking
st.session_state["factor_scores"] = factor_scores
st.session_state["factor_weights"] = weights


# =========================================================
# METHODOLOGY
# =========================================================

st.divider()

with st.expander("Ranking Methodology & Limitations"):

    st.markdown(
        """
        **1. Factor normalization**

        Each individual factor measure is converted into a
        cross-sectional percentile score from 0 to 100.

        **2. Factor direction**

        Higher momentum, earnings yield, profitability,
        growth and liquidity are preferred.

        Lower P/E, P/B, leverage and volatility are preferred.

        **3. Category scores**

        Individual measure scores are averaged within each
        factor category.

        **4. Composite score**

        Category scores are combined using user-defined
        weights. Missing categories are excluded and the
        remaining weights are renormalized for each stock.

        **5. Ranking**

        Stocks are ranked in descending order of composite
        scores.

        **Important limitations**

        - Value, Quality and Growth require genuine financial
          statement data.
        - Current rankings use only available factor categories.
        - Percentile scores are relative to this selected universe.
        - A small universe makes percentile rankings unstable.
        - This is a latest-date cross-sectional ranking, not a
          historical backtest.
        - Rankings are research outputs, not investment advice.
        """
    )


st.caption(
    "Quantitative Equity Factor Research Engine | "
    "Multi-Factor Stock Ranking"
)
