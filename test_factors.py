
import pandas as pd

from modules.factor_engine import (
    calculate_momentum,
    calculate_low_volatility,
    calculate_liquidity,
    calculate_all_factors,
    normalize_factors
)

from modules.data_loader import download_market_data
from modules.universe_engine import get_stock_universe


# =========================================================
# STEP 1: DOWNLOAD MARKET DATA
# =========================================================

print("\nDownloading market data...")

tickers = get_stock_universe("NIFTY 50")[:5]

market_data = download_market_data(
    tickers,
    period="2y",
    interval="1d"
)


# =========================================================
# STEP 2: EXTRACT DATA
# =========================================================

prices = market_data["prices"]
returns = market_data["returns"]
volumes = market_data["volumes"]


print("\n========== DATA VALIDATION ==========")

print("Prices:", prices.shape)
print("Returns:", returns.shape)
print("Volumes:", volumes.shape)

print("\nSelected stocks:")
print(tickers)


# =========================================================
# STEP 3: TEST MOMENTUM
# =========================================================

print("\n========== MOMENTUM ==========")

momentum = calculate_momentum(prices)

print(momentum)


# =========================================================
# STEP 4: TEST LOW VOLATILITY
# =========================================================

print("\n========== LOW VOLATILITY ==========")

volatility = calculate_low_volatility(returns)

print(volatility)


# =========================================================
# STEP 5: TEST LIQUIDITY
# =========================================================

print("\n========== LIQUIDITY ==========")

liquidity = calculate_liquidity(
    prices,
    volumes
)

print(liquidity)


# =========================================================
# STEP 6: COMBINE ALL FACTORS
# =========================================================

print("\n========== COMBINED FACTORS ==========")

# Fundamental data will be integrated separately.
# Empty fundamental fields are intentional for this test.

fundamentals = pd.DataFrame({
    "Ticker": tickers
})

factor_data = calculate_all_factors(
    prices,
    returns,
    volumes,
    fundamentals
)

print(factor_data)


# =========================================================
# STEP 7: NORMALIZE FACTORS
# =========================================================

print("\n========== NORMALIZED FACTORS ==========")

normalized_data = normalize_factors(
    factor_data,
    method="Z-Score"
)

print(normalized_data)


# =========================================================
# STEP 8: BASIC VALIDATION
# =========================================================

print("\n========== VALIDATION ==========")

print("Momentum calculated:", not momentum.empty)
print("Volatility calculated:", not volatility.empty)
print("Liquidity calculated:", not liquidity.empty)
print("Combined factors calculated:", not factor_data.empty)
print("Normalization completed:", not normalized_data.empty)

print("\nFactor engine testing completed successfully.")
