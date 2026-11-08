
import pandas as pd

NIFTY_50 = [
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "BHARTIARTL.NS",
    "INFY.NS",
    "TCS.NS",
    "SBIN.NS",
    "ITC.NS",
    "HINDUNILVR.NS",
    "LT.NS"
]


def get_stock_universe(name="NIFTY 50"):
    if name == "NIFTY 50":
        return NIFTY_50.copy()

    raise ValueError(f"Unknown universe: {name}")


def get_universe_dataframe(name="NIFTY 50"):
    tickers = get_stock_universe(name)

    return pd.DataFrame({
        "Ticker": tickers,
        "Company": [ticker.replace(".NS", "") for ticker in tickers]
    })


def get_universe_summary(prices):
    if prices is None or prices.empty:
        return {
            "stocks": 0,
            "observations": 0,
            "start_date": None,
            "end_date": None
        }

    return {
        "stocks": prices.shape[1],
        "observations": prices.shape[0],
        "start_date": prices.index.min(),
        "end_date": prices.index.max()
    }
