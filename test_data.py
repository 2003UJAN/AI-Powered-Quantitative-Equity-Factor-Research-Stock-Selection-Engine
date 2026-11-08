
from modules.universe_engine import get_stock_universe
from modules.data_loader import download_market_data, download_benchmark

print("Downloading NIFTY 50 stock data...")

tickers = get_stock_universe("NIFTY 50")

# Start with five stocks to test the pipeline.
test_tickers = tickers[:5]

market_data = download_market_data(
    test_tickers,
    period="1y"
)

prices = market_data["prices"]
returns = market_data["returns"]
volumes = market_data["volumes"]

print("\nSTOCK PRICES")
print(prices.tail())

print("\nSTOCK RETURNS")
print(returns.tail())

print("\nTRADING VOLUME")
print(volumes.tail())

print("\nDATA SUMMARY")
print(prices.describe())

print("\nDOWNLOADING NIFTY 50 BENCHMARK")

benchmark_prices, benchmark_returns = download_benchmark(
    "^NSEI",
    period="1y"
)

print("\nBENCHMARK PRICES")
print(benchmark_prices.tail())

print("\nBENCHMARK RETURNS")
print(benchmark_returns.tail())

print("\nDATA PIPELINE TEST COMPLETED")
