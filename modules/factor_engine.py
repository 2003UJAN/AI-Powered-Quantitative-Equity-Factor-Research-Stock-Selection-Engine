
import numpy as np
import pandas as pd


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def _get_numeric_column(data, column):
    """Return a numeric Series, even if the column is missing."""

    if column in data.columns:
        return pd.to_numeric(data[column], errors="coerce")

    return pd.Series(np.nan, index=data.index, dtype=float)


def _prepare_fundamentals(fundamentals):
    """Prepare fundamental data with Ticker as the index."""

    if fundamentals is None or fundamentals.empty:
        return pd.DataFrame()

    data = fundamentals.copy()

    if "Ticker" in data.columns:
        data = data.drop_duplicates(subset="Ticker")
        data = data.set_index("Ticker")

    return data


# =========================================================
# 1. MOMENTUM FACTOR
# =========================================================

def calculate_momentum(prices):

    if prices is None or prices.empty:
        return pd.DataFrame()

    prices = prices.sort_index()

    latest = prices.iloc[-1]

    result = pd.DataFrame(index=prices.columns)

    lookback_periods = {
        "Momentum_1M": 21,
        "Momentum_3M": 63,
        "Momentum_6M": 126,
        "Momentum_12M": 252
    }

    for factor, period in lookback_periods.items():

        if len(prices) > period:

            previous = prices.iloc[-(period + 1)]

            result[factor] = (
                latest / previous.where(previous > 0) - 1
            )

        else:
            result[factor] = np.nan

    return result.replace([np.inf, -np.inf], np.nan)


# =========================================================
# 2. VALUE FACTOR
# =========================================================

def calculate_value(fundamentals):

    data = _prepare_fundamentals(fundamentals)

    if data.empty:
        return pd.DataFrame()

    result = pd.DataFrame(index=data.index)

    result["PE_Ratio"] = _get_numeric_column(
        data, "PE Ratio"
    )

    result["PB_Ratio"] = _get_numeric_column(
        data, "PB Ratio"
    )

    result["Earnings_Yield"] = (
        1 / result["PE_Ratio"].where(
            result["PE_Ratio"] > 0
        )
    )

    result["Book_to_Price"] = (
        1 / result["PB_Ratio"].where(
            result["PB_Ratio"] > 0
        )
    )

    return result.replace([np.inf, -np.inf], np.nan)


# =========================================================
# 3. QUALITY FACTOR
# =========================================================

def calculate_quality(fundamentals):

    data = _prepare_fundamentals(fundamentals)

    if data.empty:
        return pd.DataFrame()

    result = pd.DataFrame(index=data.index)

    result["ROE"] = _get_numeric_column(
        data, "ROE"
    )

    result["ROA"] = _get_numeric_column(
        data, "ROA"
    )

    result["Profit_Margin"] = _get_numeric_column(
        data, "Profit Margin"
    )

    result["Debt_to_Equity"] = _get_numeric_column(
        data, "Debt to Equity"
    )

    # Lower debt-to-equity is preferred.
    result["Leverage_Score"] = (
        -result["Debt_to_Equity"]
    )

    return result.replace([np.inf, -np.inf], np.nan)


# =========================================================
# 4. GROWTH FACTOR
# =========================================================

def calculate_growth(fundamentals):

    data = _prepare_fundamentals(fundamentals)

    if data.empty:
        return pd.DataFrame()

    result = pd.DataFrame(index=data.index)

    result["Revenue_Growth"] = _get_numeric_column(
        data, "Revenue Growth"
    )

    result["Earnings_Growth"] = _get_numeric_column(
        data, "Earnings Growth"
    )

    return result.replace([np.inf, -np.inf], np.nan)


# =========================================================
# 5. LOW VOLATILITY FACTOR
# =========================================================

def calculate_low_volatility(returns):

    if returns is None or returns.empty:
        return pd.DataFrame()

    returns = returns.sort_index()

    result = pd.DataFrame(index=returns.columns)

    lookback_periods = {
        "Volatility_30D": 30,
        "Volatility_60D": 60,
        "Volatility_90D": 90,
        "Volatility_1Y": 252
    }

    for factor, period in lookback_periods.items():

        historical_returns = returns.tail(period)

        minimum_observations = max(
            2, int(period * 0.8)
        )

        volatility = (
            historical_returns.std(ddof=1) * np.sqrt(252)
        )

        valid_counts = historical_returns.count()

        volatility[valid_counts < minimum_observations] = np.nan

        result[factor] = volatility

    return result.replace([np.inf, -np.inf], np.nan)


# =========================================================
# 6. LIQUIDITY FACTOR
# =========================================================

def calculate_liquidity(prices, volumes):

    if (
        prices is None or volumes is None
        or prices.empty or volumes.empty
    ):
        return pd.DataFrame()

    common = prices.columns.intersection(
        volumes.columns
    )

    if len(common) == 0:
        return pd.DataFrame()

    prices = prices[common]
    volumes = volumes[common]

    traded_value = prices * volumes

    result = pd.DataFrame(index=common)

    result["Average_Daily_Traded_Value"] = (
        traded_value.tail(60).mean()
    )

    result["Average_Daily_Volume"] = (
        volumes.tail(60).mean()
    )

    return result.replace([np.inf, -np.inf], np.nan)


# =========================================================
# 7. COMBINE ALL SIX FACTOR CATEGORIES
# =========================================================

def calculate_all_factors(
    prices,
    returns,
    volumes,
    fundamentals
):

    factors = [
        calculate_momentum(prices),
        calculate_value(fundamentals),
        calculate_quality(fundamentals),
        calculate_growth(fundamentals),
        calculate_low_volatility(returns),
        calculate_liquidity(prices, volumes)
    ]

    combined = pd.concat(
        factors,
        axis=1
    )

    combined.index.name = "Ticker"

    return combined.replace(
        [np.inf, -np.inf], np.nan
    )


# =========================================================
# 8. FACTOR NORMALIZATION
# =========================================================

def normalize_factors(
    factor_data,
    method="Z-Score"
):

    if factor_data is None or factor_data.empty:
        return pd.DataFrame()

    result = factor_data.copy()

    for column in result.columns:

        values = pd.to_numeric(
            result[column],
            errors="coerce"
        )

        if method == "Z-Score":

            std = values.std()

            if pd.notna(std) and std > 0:

                result[column] = (
                    values - values.mean()
                ) / std

            else:
                result[column] = np.nan

        elif method == "Percentile Ranking":

            result[column] = values.rank(
                pct=True
            )

        elif method == "Min-Max Scaling":

            minimum = values.min()
            maximum = values.max()

            if (
                pd.notna(minimum)
                and pd.notna(maximum)
                and maximum > minimum
            ):

                result[column] = (
                    values - minimum
                ) / (maximum - minimum)

            else:
                result[column] = np.nan

        else:
            raise ValueError(
                f"Unsupported normalization method: {method}"
            )

    return result.replace(
        [np.inf, -np.inf], np.nan
    )
