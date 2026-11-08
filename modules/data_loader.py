
import yfinance as yf
import pandas as pd
import numpy as np


def download_stock_data(tickers, period="5y", interval="1d"):

    if not tickers:
        raise ValueError("Ticker list cannot be empty.")

    data = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True
    )

    if data.empty:
        raise ValueError("No market data received from Yahoo Finance.")

    return data


def prepare_stock_data(data, tickers):

    prices = {}
    volumes = {}

    for ticker in tickers:

        try:
            if isinstance(data.columns, pd.MultiIndex):
                ticker_data = data[ticker]
            else:
                ticker_data = data

            prices[ticker] = ticker_data["Close"]
            volumes[ticker] = ticker_data["Volume"]

        except (KeyError, TypeError):
            continue

    price_df = pd.DataFrame(prices)
    volume_df = pd.DataFrame(volumes)

    price_df = price_df.sort_index().dropna(how="all")
    volume_df = volume_df.sort_index().dropna(how="all")

    returns_df = price_df.pct_change(fill_method=None)

    return {
        "prices": price_df,
        "returns": returns_df,
        "volumes": volume_df
    }


def download_market_data(tickers, period="5y", interval="1d"):

    raw_data = download_stock_data(
        tickers=tickers,
        period=period,
        interval=interval
    )

    return prepare_stock_data(raw_data, tickers)


def download_fundamental_data(tickers):

    records = []

    for ticker in tickers:

        try:
            info = yf.Ticker(ticker).info

            records.append({
                "Ticker": ticker,
                "Market Cap": info.get("marketCap"),
                "PE Ratio": info.get("trailingPE"),
                "Forward PE": info.get("forwardPE"),
                "PB Ratio": info.get("priceToBook"),
                "ROE": info.get("returnOnEquity"),
                "ROA": info.get("returnOnAssets"),
                "Profit Margin": info.get("profitMargins"),
                "Revenue Growth": info.get("revenueGrowth"),
                "Earnings Growth": info.get("earningsGrowth"),
                "Debt to Equity": info.get("debtToEquity"),
                "Sector": info.get("sector"),
                "Average Volume": info.get("averageVolume")
            })

        except Exception as error:
            print(f"Fundamental data unavailable for {ticker}: {error}")
            records.append({"Ticker": ticker})

    return pd.DataFrame(records)


def download_benchmark(ticker="^NSEI", period="5y"):

    data = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError("Benchmark data unavailable.")

    prices = data["Close"]

    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]

    returns = prices.pct_change(fill_method=None).dropna()

    return prices, returns


def calculate_returns(prices, frequency="Daily"):

    if frequency == "Weekly":
        prices = prices.resample("W-FRI").last()

    elif frequency == "Monthly":
        prices = prices.resample("ME").last()

    return prices.pct_change(fill_method=None).dropna(how="all")
