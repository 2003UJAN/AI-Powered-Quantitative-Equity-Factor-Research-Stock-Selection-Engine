
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from modules.ml_engine import (
    train_regression_model,
    train_classification_model
)


# =========================================================
# PAGE HEADER
# =========================================================

st.title("ML-Based Equity Stock Selection")

st.caption(
    "Machine learning models for forecasting equity returns "
    "and identifying potential stock opportunities."
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


prices = prices.copy().sort_index()
returns = returns.copy().sort_index()
volumes = volumes.copy().sort_index()

prices.index = pd.to_datetime(prices.index)
returns.index = pd.to_datetime(returns.index)
volumes.index = pd.to_datetime(volumes.index)


# =========================================================
# MODEL CONFIGURATION
# =========================================================

st.subheader("Model Configuration")

col1, col2, col3 = st.columns(3)

with col1:

    task = st.selectbox(
        "Prediction Task",
        [
            "Regression",
            "Classification"
        ]
    )

with col2:

    if task == "Regression":

        model_name = st.selectbox(
            "Machine Learning Model",
            [
                "Linear Regression",
                "Random Forest",
                "Gradient Boosting",
                "XGBoost"
            ]
        )

    else:

        model_name = st.selectbox(
            "Machine Learning Model",
            [
                "Logistic Regression",
                "Random Forest"
            ]
        )

with col3:

    forward_horizon = st.selectbox(
        "Prediction Horizon",
        [21, 63],
        index=0,
        format_func=lambda x: f"{x} Trading Days"
    )


col4, col5 = st.columns(2)

with col4:

    test_size = st.slider(
        "Test Data Percentage",
        min_value=10,
        max_value=40,
        value=20,
        step=5
    ) / 100

with col5:

    st.metric(
        "Sampling Frequency",
        "Monthly"
    )


# =========================================================
# FEATURE ENGINEERING
# =========================================================

st.divider()

st.subheader("Feature Engineering")


def create_ml_dataset(
    prices,
    returns,
    volumes,
    forward_horizon=21
):
    """
    Construct stock-level historical features.

    Features at date t use information available up to date t.

    Target:
        Price(t + horizon) / Price(t) - 1

    Observations are sampled monthly using the final
    available trading day of each month.
    """

    prices = prices.sort_index()
    returns = returns.sort_index()
    volumes = volumes.sort_index()

    common_tickers = (
        prices.columns
        .intersection(returns.columns)
        .intersection(volumes.columns)
    )

    prices = prices[common_tickers]
    returns = returns[common_tickers]
    volumes = volumes[common_tickers]

    feature_frames = {}

    for ticker in common_tickers:

        stock_prices = prices[ticker]
        stock_returns = returns[ticker]
        stock_volumes = volumes[ticker]

        features = pd.DataFrame(
            index=prices.index
        )

        # Momentum

        features["Momentum_1M"] = (
            stock_prices / stock_prices.shift(21) - 1
        )

        features["Momentum_3M"] = (
            stock_prices / stock_prices.shift(63) - 1
        )

        features["Momentum_6M"] = (
            stock_prices / stock_prices.shift(126) - 1
        )

        features["Momentum_12M"] = (
            stock_prices / stock_prices.shift(252) - 1
        )

        # Volatility

        features["Volatility_21D"] = (
            stock_returns.rolling(21).std() * np.sqrt(252)
        )

        features["Volatility_63D"] = (
            stock_returns.rolling(63).std() * np.sqrt(252)
        )

        # Liquidity

        traded_value = (
            stock_prices * stock_volumes
        )

        features["Average_Traded_Value_21D"] = (
            traded_value.rolling(21).mean()
        )

        features["Average_Traded_Value_63D"] = (
            traded_value.rolling(63).mean()
        )

        features["Average_Volume_21D"] = (
            stock_volumes.rolling(21).mean()
        )

        feature_frames[ticker] = features

    # Forward return target

    future_returns = (
        prices.shift(-forward_horizon) / prices - 1
    )

    # Last available trading date of each calendar month

    sample_dates = (
        prices.groupby(
            prices.index.to_period("M")
        )
        .tail(1)
        .index
    )

    dataset_frames = []

    for ticker in common_tickers:

        ticker_features = feature_frames[ticker].loc[
            sample_dates
        ].copy()

        ticker_features["Target"] = (
            future_returns[ticker].reindex(sample_dates)
        )

        ticker_features["Ticker"] = ticker

        # Remove observations with unavailable future returns.

        ticker_features = ticker_features.dropna(
            subset=["Target"]
        )

        dataset_frames.append(ticker_features)

    if not dataset_frames:
        return pd.DataFrame(), pd.Series(dtype=float)

    dataset = pd.concat(dataset_frames)

    dataset.index.name = "Date"

    dataset = dataset.reset_index()

    dataset = dataset.set_index(
        ["Date", "Ticker"]
    ).sort_index()

    feature_columns = [
        column for column in dataset.columns
        if column != "Target"
    ]

    # Remove rows with no usable feature information.

    dataset = dataset.dropna(
        subset=feature_columns,
        how="all"
    )

    X = dataset[feature_columns]
    y = dataset["Target"]

    return X, y


# =========================================================
# FEATURE DESCRIPTION
# =========================================================

feature_names = [
    "Momentum_1M",
    "Momentum_3M",
    "Momentum_6M",
    "Momentum_12M",
    "Volatility_21D",
    "Volatility_63D",
    "Average_Traded_Value_21D",
    "Average_Traded_Value_63D",
    "Average_Volume_21D"
]

with st.expander("View Model Features"):

    feature_description = pd.DataFrame({

        "Feature": feature_names,

        "Category": [
            "Momentum",
            "Momentum",
            "Momentum",
            "Momentum",
            "Risk",
            "Risk",
            "Liquidity",
            "Liquidity",
            "Liquidity"
        ],

        "Description": [
            "21-day historical return",
            "63-day historical return",
            "126-day historical return",
            "252-day historical return",
            "21-day annualized volatility",
            "63-day annualized volatility",
            "21-day average traded value",
            "63-day average traded value",
            "21-day average trading volume"
        ]
    })

    st.dataframe(
        feature_description,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# GENERATE DATASET
# =========================================================

st.divider()

st.subheader("Training Dataset")

X, y = create_ml_dataset(
    prices=prices,
    returns=returns,
    volumes=volumes,
    forward_horizon=forward_horizon
)

if X.empty or y.empty:

    st.error(
        "Unable to construct the training dataset. "
        "Please download a longer historical period."
    )

    st.stop()


# Remove completely empty feature columns.

X = X.dropna(axis=1, how="all")

if X.empty:

    st.error("No usable model features are available.")

    st.stop()


# Dataset statistics

dataset_col1, dataset_col2, dataset_col3 = st.columns(3)

dataset_col1.metric(
    "Training Observations",
    len(X)
)

dataset_col2.metric(
    "Stocks Included",
    X.index.get_level_values("Ticker").nunique()
)

dataset_col3.metric(
    "Available Features",
    X.shape[1]
)


st.caption(
    f"Historical sample: "
    f"{X.index.get_level_values('Date').min().date()} "
    f"to "
    f"{X.index.get_level_values('Date').max().date()}"
)


with st.expander("Preview Training Dataset"):

    preview = X.copy()
    preview["Future_Return"] = y

    st.dataframe(
        preview.head(30),
        use_container_width=True
    )


# =========================================================
# MODEL TRAINING
# =========================================================

st.divider()

st.subheader("Model Training")

st.write(
    "Train a machine learning model using historical "
    "momentum, volatility and liquidity features."
)

run_training = st.button(
    "Train Machine Learning Model",
    type="primary",
    use_container_width=True
)


if run_training:

    try:

        with st.spinner("Training model..."):

            if task == "Regression":

                model, metrics, results = (
                    train_regression_model(
                        X=X,
                        y=y,
                        model_name=model_name,
                        test_size=test_size,
                        gap=1
                    )
                )

            else:

                classification_target = (
                    y > 0
                ).astype(int)

                if classification_target.nunique() < 2:

                    st.error(
                        "The target contains only one class. "
                        "Classification requires positive and "
                        "non-positive forward returns."
                    )

                    st.stop()

                model, metrics, results = (
                    train_classification_model(
                        X=X,
                        y=classification_target,
                        model_name=model_name,
                        test_size=test_size,
                        gap=1
                    )
                )

            st.session_state["ml_model"] = model
            st.session_state["ml_metrics"] = metrics
            st.session_state["ml_results"] = results
            st.session_state["ml_task"] = task
            st.session_state["ml_model_name"] = model_name
            st.session_state["ml_features"] = X.columns.tolist()

            st.session_state["ml_config"] = {
                "task": task,
                "model": model_name,
                "horizon": forward_horizon,
                "test_size": test_size
            }

            st.success(
                "Model training completed successfully."
            )

    except Exception as error:

        st.error(
            f"Model training failed: {error}"
        )


# =========================================================
# MODEL EVALUATION
# =========================================================

saved_config = st.session_state.get("ml_config")

current_config = {
    "task": task,
    "model": model_name,
    "horizon": forward_horizon,
    "test_size": test_size
}

if (
    saved_config == current_config
    and "ml_metrics" in st.session_state
):

    metrics = st.session_state["ml_metrics"]
    results = st.session_state["ml_results"]

    st.divider()

    st.subheader("Model Evaluation")

    st.caption(
        f"Model: {model_name} | Task: {task}"
    )

    metric_columns = st.columns(len(metrics))

    for column, (metric_name, metric_value) in zip(
        metric_columns,
        metrics.items()
    ):

        column.metric(
            metric_name,
            f"{metric_value:.4f}"
        )


    # =====================================================
    # REGRESSION RESULTS
    # =====================================================

    if task == "Regression":

        st.subheader("Actual vs Predicted Returns")

        chart_data = results.reset_index()

        result_chart = px.scatter(
            chart_data,
            x="Actual",
            y="Predicted",
            title="Actual vs Predicted Future Returns",
            template="plotly_dark"
        )

        minimum = min(
            results["Actual"].min(),
            results["Predicted"].min()
        )

        maximum = max(
            results["Actual"].max(),
            results["Predicted"].max()
        )

        result_chart.add_trace(
            go.Scatter(
                x=[minimum, maximum],
                y=[minimum, maximum],
                mode="lines",
                name="Perfect Prediction"
            )
        )

        result_chart.update_layout(
            height=450
        )

        st.plotly_chart(
            result_chart,
            use_container_width=True
        )


    # =====================================================
    # CLASSIFICATION RESULTS
    # =====================================================

    else:

        st.subheader("Classification Results")

        classification_summary = results.copy()

        classification_summary["Actual Direction"] = (
            classification_summary["Actual"]
            .map({
                0: "Non-positive",
                1: "Positive"
            })
        )

        classification_summary["Predicted Direction"] = (
            classification_summary["Predicted"]
            .map({
                0: "Non-positive",
                1: "Positive"
            })
        )

        st.dataframe(
            classification_summary,
            use_container_width=True
        )


    # =====================================================
    # OUT-OF-SAMPLE RESULTS
    # =====================================================

    st.divider()

    st.subheader("Out-of-Sample Predictions")

    st.dataframe(
        results,
        use_container_width=True
    )

    st.download_button(
        "Download Model Predictions",
        data=results.to_csv().encode("utf-8"),
        file_name="ml_stock_predictions.csv",
        mime="text/csv"
    )


elif "ml_metrics" in st.session_state:

    st.info(
        "Your model settings have changed. "
        "Train the model again to generate updated results."
    )


# =========================================================
# METHODOLOGY AND LIMITATIONS
# =========================================================

st.divider()

with st.expander("Model Methodology & Limitations"):

    st.markdown(
        """
        **Feature engineering**

        - Historical momentum
        - Realized volatility
        - Average traded value
        - Average trading volume

        **Regression**

        Predicts the numerical future stock return.

        **Classification**

        Predicts whether the future stock return is positive.

        **Validation**

        Uses chronological train/test splitting with a
        one-observation purge gap.

        **Limitations**

        - Current stock universe is a small development subset.
        - Monthly observations are used for initial model training.
        - Only market-derived features are currently included.
        - Historical fundamentals are not yet integrated.
        - A one-observation gap is not a complete purged
          walk-forward validation framework.
        - Transaction costs and portfolio constraints are not
          included in model evaluation.
        - Results are experimental, not investment recommendations.
        """
    )


st.caption(
    "Quantitative Equity Factor Research Engine | "
    "Machine Learning Stock Selection"
)
