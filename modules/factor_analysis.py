
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


def calculate_information_coefficient(
    factor_scores,
    forward_returns,
    method="spearman"
):
    """
    Cross-sectional Information Coefficient.
    """

    data = pd.concat(
        [
            factor_scores.rename("Factor"),
            forward_returns.rename("Forward_Return")
        ],
        axis=1
    ).dropna()

    if len(data) < 3:
        return np.nan

    if method.lower() == "spearman":
        result = spearmanr(
            data["Factor"], data["Forward_Return"]
        )

    elif method.lower() == "pearson":
        result = pearsonr(
            data["Factor"], data["Forward_Return"]
        )

    else:
        raise ValueError("Method must be spearman or pearson.")

    return float(result.statistic)


def calculate_ic_series(
    factor_history,
    forward_return_history,
    method="spearman"
):
    """
    Calculate cross-sectional IC at each date.

    Inputs must have dates as rows and stocks as columns.
    """

    common_dates = factor_history.index.intersection(
        forward_return_history.index
    )

    results = {}

    for dt in common_dates:

        results[dt] = calculate_information_coefficient(
            factor_history.loc[dt],
            forward_return_history.loc[dt],
            method
        )

    return pd.Series(results, name="IC").sort_index()


def calculate_ic_statistics(ic_series):

    ic = ic_series.dropna()

    if ic.empty:
        return {}

    mean_ic = ic.mean()
    std_ic = ic.std()

    return {
        "Mean IC": mean_ic,
        "IC Standard Deviation": std_ic,
        "IC Information Ratio": (
            mean_ic / std_ic if std_ic > 0 else np.nan
        ),
        "Positive IC Percentage": (ic > 0).mean() * 100,
        "Observations": len(ic)
    }


def calculate_quintile_returns(
    factor_scores,
    forward_returns,
    quintiles=5
):
    """
    Assign stocks to factor-score quintiles
    and calculate their mean forward returns.
    """

    data = pd.concat(
        [
            factor_scores.rename("Factor"),
            forward_returns.rename("Forward_Return")
        ],
        axis=1
    ).dropna()

    if len(data) < quintiles:
        raise ValueError("Insufficient observations for quintile analysis.")

    data["Quintile"] = pd.qcut(
        data["Factor"].rank(method="first"),
        q=quintiles,
        labels=False
    ) + 1

    result = data.groupby("Quintile")["Forward_Return"].agg(
        ["mean", "median", "count"]
    )

    result.columns = [
        "Average Forward Return",
        "Median Forward Return",
        "Number of Stocks"
    ]

    return result


def calculate_long_short_return(quintile_returns):

    if 1 not in quintile_returns.index or 5 not in quintile_returns.index:
        return np.nan

    return (
        quintile_returns.loc[5, "Average Forward Return"]
        - quintile_returns.loc[1, "Average Forward Return"]
    )
