
import numpy as np
import pandas as pd


# =========================================================
# FACTOR DEFINITIONS AND DIRECTIONS
# =========================================================

FACTOR_GROUPS = {
    "Momentum": {
        "Momentum_1M": 1,
        "Momentum_3M": 1,
        "Momentum_6M": 1,
        "Momentum_12M": 1
    },

    "Value": {
        "PE_Ratio": -1,
        "PB_Ratio": -1,
        "Earnings_Yield": 1,
        "Book_to_Price": 1
    },

    "Quality": {
        "ROE": 1,
        "ROA": 1,
        "Profit_Margin": 1,
        "Leverage_Score": 1
    },

    "Growth": {
        "Revenue_Growth": 1,
        "Earnings_Growth": 1
    },

    "Low Volatility": {
        "Volatility_30D": -1,
        "Volatility_60D": -1,
        "Volatility_90D": -1,
        "Volatility_1Y": -1
    },

    "Liquidity": {
        "Average_Daily_Traded_Value": 1,
        "Average_Daily_Volume": 1
    }
}


DEFAULT_WEIGHTS = {
    "Momentum": 0.25,
    "Value": 0.20,
    "Quality": 0.20,
    "Growth": 0.15,
    "Low Volatility": 0.10,
    "Liquidity": 0.10
}


# =========================================================
# 1. CALCULATE INDIVIDUAL FACTOR SCORES
# =========================================================

def calculate_factor_scores(factor_data):
    """
    Convert raw factor measures into percentile scores.

    Scores range from 0 to 100.
    Higher scores always represent a more preferred outcome.

    Each factor category is calculated only from available
    measures. Missing categories remain NaN.
    """

    if factor_data is None or factor_data.empty:
        raise ValueError("Factor data is empty.")

    result = pd.DataFrame(index=factor_data.index)

    for group, definitions in FACTOR_GROUPS.items():

        component_scores = []

        for column, direction in definitions.items():

            if column not in factor_data.columns:
                continue

            values = pd.to_numeric(
                factor_data[column],
                errors="coerce"
            )

            values = values.replace(
                [np.inf, -np.inf], np.nan
            )

            if values.notna().sum() < 2:
                continue

            # Positive direction: higher is better.
            # Negative direction: lower is better.
            ranked = values.rank(
                pct=True,
                ascending=(direction == 1),
                method="average"
            ) * 100

            component_scores.append(ranked)

        if component_scores:

            group_scores = pd.concat(
                component_scores,
                axis=1
            )

            result[group] = group_scores.mean(
                axis=1,
                skipna=True
            )

        else:

            result[group] = np.nan

    return result


# =========================================================
# 2. NORMALIZE FACTOR WEIGHTS
# =========================================================

def normalize_weights(weights, available_factors=None):
    """
    Normalize weights to sum to 1.

    If some factor categories are unavailable, their weights
    are excluded and the remaining weights are rescaled.
    """

    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    cleaned = {}

    for factor, weight in weights.items():

        weight = float(weight)

        if not np.isfinite(weight) or weight < 0:
            raise ValueError(
                f"Invalid weight for {factor}."
            )

        cleaned[factor] = weight

    if available_factors is not None:

        cleaned = {
            factor: weight
            for factor, weight in cleaned.items()
            if factor in available_factors
        }

    total = sum(cleaned.values())

    if total <= 0:
        raise ValueError(
            "At least one available factor must have a positive weight."
        )

    return {
        factor: weight / total
        for factor, weight in cleaned.items()
    }


# =========================================================
# 3. CALCULATE COMPOSITE STOCK SCORES
# =========================================================

def calculate_composite_score(
    factor_scores,
    weights=None,
    require_all_factors=False
):
    """
    Calculate weighted composite scores.

    Missing factor categories are excluded for each stock,
    and remaining weights are renormalized for that stock.

    The default does not require all six factor categories.
    """

    if factor_scores is None or factor_scores.empty:
        raise ValueError("Factor scores are empty.")

    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    available_factors = [
        factor
        for factor in weights
        if (
            factor in factor_scores.columns
            and factor_scores[factor].notna().any()
        )
    ]

    if not available_factors:
        raise ValueError(
            "No available factor categories to calculate ranking."
        )

    normalized = normalize_weights(
        weights,
        available_factors
    )

    score = pd.Series(
        0.0,
        index=factor_scores.index
    )

    weight_sum = pd.Series(
        0.0,
        index=factor_scores.index
    )

    for factor, weight in normalized.items():

        values = factor_scores[factor]

        valid = values.notna()

        score.loc[valid] += (
            values.loc[valid] * weight
        )

        weight_sum.loc[valid] += weight

    if require_all_factors:

        valid_rows = factor_scores[
            available_factors
        ].notna().all(axis=1)

        score.loc[~valid_rows] = np.nan

    else:

        score = score.div(
            weight_sum.where(weight_sum > 0)
        )

    return score.rename("Composite_Score")


# =========================================================
# 4. RANK STOCKS
# =========================================================

def rank_stocks(
    factor_data,
    weights=None,
    require_all_factors=False
):
    """
    Generate a stock ranking using available factor data.

    Returns a DataFrame containing factor scores,
    composite score and rank.
    """

    factor_scores = calculate_factor_scores(
        factor_data
    )

    composite_scores = calculate_composite_score(
        factor_scores,
        weights=weights,
        require_all_factors=require_all_factors
    )

    ranking = factor_scores.copy()

    ranking["Composite_Score"] = composite_scores

    ranking["Rank"] = (
        ranking["Composite_Score"]
        .rank(
            ascending=False,
            method="min"
        )
    )

    ranking = ranking.sort_values(
        "Composite_Score",
        ascending=False,
        na_position="last"
    )

    return ranking


# =========================================================
# 5. FACTOR CONTRIBUTION ANALYSIS
# =========================================================

def calculate_factor_contributions(
    factor_scores,
    weights=None
):
    """
    Calculate each category's weighted contribution
    to each stock's composite score.

    Contributions are normalized using available factors
    for each stock.
    """

    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    available_factors = [
        factor
        for factor in weights
        if (
            factor in factor_scores.columns
            and factor_scores[factor].notna().any()
        )
    ]

    normalized_weights = normalize_weights(
        weights,
        available_factors
    )

    contribution = pd.DataFrame(
        index=factor_scores.index
    )

    weight_sum = pd.Series(
        0.0,
        index=factor_scores.index
    )

    for factor in available_factors:

        weight_sum += (
            factor_scores[factor].notna()
            * normalized_weights[factor]
        )

    for factor in available_factors:

        contribution[factor] = (
            factor_scores[factor]
            * normalized_weights[factor]
            / weight_sum.where(weight_sum > 0)
        )

    return contribution
