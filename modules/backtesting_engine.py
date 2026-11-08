# modules/backtesting_engine.py

import numpy as np
import pandas as pd


def _get_rebalance_dates(index, frequency="Monthly"):
    """Return the last available trading date in each rebalance period."""

    index = pd.DatetimeIndex(index)

    if frequency == "Monthly":
        periods = index.to_period("M")
    elif frequency == "Quarterly":
        periods = index.to_period("Q")
    else:
        raise ValueError("Frequency must be Monthly or Quarterly.")

    dates = pd.Series(index, index=index).groupby(periods).last()

    return list(dates.values)


def _calculate_historical_scores(
    prices,
    volumes,
    date,
    strategy="Momentum + Low Volatility",
    momentum_lookback=126,
    volatility_lookback=63
):
    """Calculate cross-sectional stock scores using data available up to date."""

    historical_prices = prices.loc[:date]

    if len(historical_prices) <= max(momentum_lookback, volatility_lookback):
        return pd.Series(dtype=float)

    # Momentum: trailing price return
    momentum = (
        historical_prices.iloc[-1]
        / historical_prices.iloc[-momentum_lookback - 1]
    ) - 1

    # Volatility: annualised historical daily volatility
    daily_returns = historical_prices.pct_change()

    volatility = (
        daily_returns.tail(volatility_lookback)
        .std()
        * np.sqrt(252)
    )

    score_components = []

    # Higher momentum is preferred
    momentum_score = momentum.rank(pct=True)
    score_components.append(momentum_score.rename("Momentum"))

    if strategy in [
        "Momentum + Low Volatility",
        "Momentum + Low Volatility + Liquidity"
    ]:
        # Lower volatility is preferred
        volatility_score = volatility.rank(
            ascending=False,
            pct=True
        )
        score_components.append(
            volatility_score.rename("Low Volatility")
        )

    if strategy == "Momentum + Low Volatility + Liquidity":
        if volumes is not None and not volumes.empty:
            historical_volumes = volumes.loc[:date].reindex(
                columns=prices.columns
            )

            traded_value = (
                historical_prices.tail(63)
                * historical_volumes.tail(63)
            ).mean()

            liquidity_score = traded_value.rank(pct=True)

            score_components.append(
                liquidity_score.rename("Liquidity")
            )

    score_frame = pd.concat(score_components, axis=1)

    # A stock must have valid momentum to be considered.
    scores = score_frame.mean(axis=1, skipna=True)
    scores = scores.where(momentum.notna())

    return scores.replace([np.inf, -np.inf], np.nan).dropna()


def calculate_portfolio_weights(
    scores,
    top_n=5,
    weighting_method="Equal Weight"
):
    """Select the top stocks and calculate their portfolio weights."""

    scores = scores.dropna().sort_values(ascending=False)

    if scores.empty:
        return pd.Series(dtype=float)

    selected = scores.head(min(top_n, len(scores)))

    if weighting_method == "Equal Weight":
        weights = pd.Series(
            1.0 / len(selected),
            index=selected.index
        )

    elif weighting_method == "Score Weighted":
        positive_scores = selected.clip(lower=0)

        if positive_scores.sum() == 0:
            weights = pd.Series(
                1.0 / len(selected),
                index=selected.index
            )
        else:
            weights = positive_scores / positive_scores.sum()

    else:
        raise ValueError(
            "Weighting method must be Equal Weight or Score Weighted."
        )

    return weights


def calculate_performance_metrics(returns, periods_per_year=252):
    """Calculate standard portfolio performance metrics."""

    returns = pd.Series(returns).dropna()

    if returns.empty:
        return {
            "Total Return": np.nan,
            "CAGR": np.nan,
            "Annualised Volatility": np.nan,
            "Sharpe Ratio": np.nan,
            "Maximum Drawdown": np.nan,
            "Positive Days (%)": np.nan
        }

    equity_curve = (1 + returns).cumprod()

    total_return = equity_curve.iloc[-1] - 1

    years = len(returns) / periods_per_year

    if years > 0 and equity_curve.iloc[-1] > 0:
        cagr = equity_curve.iloc[-1] ** (1 / years) - 1
    else:
        cagr = np.nan

    annualised_volatility = (
        returns.std(ddof=1) * np.sqrt(periods_per_year)
        if len(returns) > 1 else np.nan
    )

    if (
        pd.notna(annualised_volatility)
        and annualised_volatility > 0
    ):
        sharpe_ratio = (
            returns.mean() * periods_per_year
        ) / annualised_volatility
    else:
        sharpe_ratio = np.nan

    running_peak = equity_curve.cummax()
    drawdown = (equity_curve / running_peak) - 1

    max_drawdown = drawdown.min()

    positive_days = (returns > 0).mean() * 100

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualised Volatility": annualised_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown": max_drawdown,
        "Positive Days (%)": positive_days
    }


def run_backtest(
    prices,
    volumes=None,
    benchmark_prices=None,
    top_n=5,
    rebalance_frequency="Monthly",
    weighting_method="Equal Weight",
    strategy="Momentum + Low Volatility",
    transaction_cost_bps=10,
    momentum_lookback=126,
    volatility_lookback=63
):
    """
    Run a historical stock-selection portfolio backtest.

    Signals are calculated using closing prices on each rebalance date.
    Target weights become active on the following trading day.

    Transaction costs are applied to the rebalance-date return.
    """

    if prices is None or prices.empty:
        raise ValueError("Price data is empty.")

    prices = prices.copy().sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.dropna(axis=1, how="all")

    if prices.empty or len(prices) < momentum_lookback + 10:
        raise ValueError(
            "Insufficient price history for the selected momentum lookback."
        )

    prices = prices.ffill()
    returns = prices.pct_change().replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    if volumes is not None and not volumes.empty:
        volumes = volumes.reindex(
            index=prices.index,
            columns=prices.columns
        ).apply(pd.to_numeric, errors="coerce")

    rebalance_dates = _get_rebalance_dates(
        prices.index,
        rebalance_frequency
    )

    target_weights = pd.DataFrame(
        0.0,
        index=prices.index,
        columns=prices.columns
    )

    turnover = pd.Series(0.0, index=prices.index)
    previous_weights = pd.Series(0.0, index=prices.columns)

    for date in rebalance_dates:

        scores = _calculate_historical_scores(
            prices=prices,
            volumes=volumes,
            date=date,
            strategy=strategy,
            momentum_lookback=momentum_lookback,
            volatility_lookback=volatility_lookback
        )

        new_weights = calculate_portfolio_weights(
            scores=scores,
            top_n=top_n,
            weighting_method=weighting_method
        )

        current_weights = pd.Series(
            0.0,
            index=prices.columns
        )

        if not new_weights.empty:
            current_weights.loc[new_weights.index] = new_weights.values

        target_weights.loc[date] = current_weights

        # Gross traded notional as a proportion of portfolio value.
        turnover.loc[date] = (
            current_weights - previous_weights
        ).abs().sum()

        previous_weights = current_weights.copy()

    # Signals from the close of date t are applied from date t+1.
    daily_weights = target_weights.replace(
        0, np.nan
    ).ffill().fillna(0).shift(1).fillna(0)

    # Restore the intended portfolio weights after each rebalance.
    # Forward-fill target weights, rather than individual zero entries.
    daily_weights = target_weights.ffill().shift(1).fillna(0)

    gross_portfolio_returns = (
        daily_weights * returns
    ).sum(axis=1)

    transaction_cost_rate = transaction_cost_bps / 10000

    transaction_costs = turnover * transaction_cost_rate

    net_portfolio_returns = (
        gross_portfolio_returns - transaction_costs
    )

    portfolio_equity = (1 + net_portfolio_returns).cumprod()

    running_peak = portfolio_equity.cummax()
    portfolio_drawdown = (
        portfolio_equity / running_peak
    ) - 1

    results = pd.DataFrame({
        "Portfolio Return": net_portfolio_returns,
        "Gross Portfolio Return": gross_portfolio_returns,
        "Turnover": turnover,
        "Transaction Cost": transaction_costs,
        "Portfolio Equity": portfolio_equity,
        "Portfolio Drawdown": portfolio_drawdown
    })

    # Benchmark comparison
    if benchmark_prices is not None and not benchmark_prices.empty:

        if isinstance(benchmark_prices, pd.Series):
            benchmark_series = benchmark_prices.copy()
        else:
            benchmark_series = benchmark_prices.iloc[:, 0].copy()

        benchmark_series = pd.to_numeric(
            benchmark_series,
            errors="coerce"
        ).dropna()

        benchmark_returns = benchmark_series.pct_change()

        benchmark_returns = benchmark_returns.reindex(
            prices.index
        ).fillna(0)

        benchmark_equity = (
            1 + benchmark_returns
        ).cumprod()

        results["Benchmark Return"] = benchmark_returns
        results["Benchmark Equity"] = benchmark_equity

        benchmark_running_peak = benchmark_equity.cummax()

        results["Benchmark Drawdown"] = (
            benchmark_equity / benchmark_running_peak
        ) - 1

    portfolio_metrics = calculate_performance_metrics(
        results["Portfolio Return"]
    )

    metrics = {"Portfolio": portfolio_metrics}

    if "Benchmark Return" in results.columns:
        metrics["NIFTY 50"] = calculate_performance_metrics(
            results["Benchmark Return"]
        )

    metrics["Average Daily Turnover"] = (
        results["Turnover"].mean()
    )

    metrics["Total Rebalances"] = int(
        (results["Turnover"] > 0).sum()
    )

    return {
        "results": results,
        "target_weights": target_weights,
        "daily_weights": daily_weights,
        "metrics": metrics,
        "rebalance_dates": rebalance_dates
    }