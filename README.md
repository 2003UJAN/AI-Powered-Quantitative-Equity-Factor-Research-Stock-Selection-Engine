# Quantitative Equity Factor Research & Stock Selection Engine

### Systematic Equity Research | Financial Factor Investing | Machine Learning | Portfolio Backtesting

## Overview

The **Quantitative Equity Factor Research & Stock Selection Engine** is a Python-based quantitative research platform designed to evaluate equities using systematic financial factors, statistical analysis and machine learning.

The platform aims to identify stocks with attractive quantitative characteristics by analyzing momentum, value, quality, growth, liquidity and volatility factors. It incorporates factor evaluation, predictive modelling, composite stock ranking and portfolio backtesting against market benchmarks such as the NIFTY 50.

## Key Features

### 1. Market Universe
- NIFTY 50, NIFTY NEXT 50 and NIFTY 100 stock universes.
- Historical stock price and trading volume collection.
- Benchmark index comparison.
- Fundamental data retrieval.

### 2. Equity Factor Analysis
Implementation of six major equity factors:

| Factor | Key Metrics |
|---|---|
| Momentum | 1M, 3M, 6M and 12M returns |
| Value | P/E, P/B, Earnings Yield, Book-to-Price |
| Quality | ROE, ROA, Profit Margin, Leverage |
| Low Volatility | 30D, 60D, 90D and 1Y volatility |
| Growth | Revenue and Earnings Growth |
| Liquidity | Average Daily Volume and Traded Value |

### 3. Factor Research
- Information Coefficient (IC).
- Spearman Rank Correlation and Pearson Correlation.
- IC Information Ratio.
- Quintile portfolio analysis.
- Long-short factor return evaluation.

### 4. Machine Learning Stock Selection
- Linear Regression.
- Random Forest.
- Gradient Boosting.
- XGBoost.
- Logistic Regression.
- Chronological train-test validation.

### 5. Quantitative Stock Ranking
- Cross-sectional factor normalization.
- Customizable factor weights.
- Composite factor score calculation.
- Systematic stock ranking.

### 6. Portfolio Backtesting
- Equal-weighted portfolio construction.
- Monthly, quarterly and annual rebalancing.
- Benchmark-relative performance analysis.
- Transaction cost estimation.
- CAGR, Sharpe Ratio, volatility and maximum drawdown.

### 7. AI Research Assistant
- Gemini API integration.
- Interpretation of factor research results.
- Quantitative stock-ranking explanations.
- AI-assisted financial research insights.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Dashboard | Streamlit |
| Market Data | Yahoo Finance (yfinance) |
| Data Processing | Pandas, NumPy |
| Statistical Analysis | SciPy |
| Machine Learning | Scikit-learn, XGBoost |
| Visualization | Plotly |
| AI Integration | Google Gemini API |
| Version Control | Git, GitHub |

## Project Structure

```text
quantitative-equity-factor-engine/
│
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml
│
├── modules/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── universe_engine.py
│   ├── factor_engine.py
│   ├── factor_analysis.py
│   ├── ml_engine.py
│   ├── ranking_engine.py
│   ├── backtesting_engine.py
│   └── gemini_assistant.py
│
├── pages/
│   ├── 1_Market_Universe.py
│   ├── 2_Factor_Analysis.py
│   ├── 3_Factor_Research.py
│   ├── 4_ML_Stock_Selection.py
│   ├── 5_Stock_Ranking.py
│   └── 6_Backtesting.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/2003UJAN/AI-Powered-Quantitative-Equity-Factor-Research-Stock-Selection-Engine.git
cd quantitative-equity-factor-engine
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Gemini API

Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "YOUR_API_KEY"
GEMINI_MODEL = "gemini-flash-latest"
```

Never commit API keys or secrets to GitHub.

### 5. Run the Application

```bash
streamlit run app.py
```

## Quantitative Methodology

The research framework follows a systematic workflow:

1. Collect historical stock prices, volumes and fundamental information.
2. Calculate and normalize equity factor metrics.
3. Evaluate factor predictive power using Information Coefficient and quintile analysis.
4. Train machine learning models using historical factor observations.
5. Generate composite factor scores and rank stocks.
6. Construct systematic portfolios and evaluate historical performance against benchmarks.

## Research Considerations

The framework is designed with attention to:

- Look-ahead bias.
- Survivorship bias.
- Chronological validation.
- Missing financial data.
- Transaction costs and portfolio turnover.
- Out-of-sample model performance.

Historical backtesting and predictive modelling require careful validation before drawing investment conclusions.

## Future Enhancements

- Historical NIFTY constituent database.
- Sector-neutral factor construction.
- Multi-factor portfolio optimization.
- Walk-forward model validation.
- Advanced portfolio risk attribution.
- Additional financial data provider integrations.
- Enhanced AI-generated quantitative research reports.

## Disclaimer

This project is intended for educational, quantitative research and analytical purposes only. It does not constitute investment advice or a recommendation to buy or sell securities. Historical performance does not guarantee future results.

## Author

**Ujan Pradhan**

B.Tech Computer Science | Quantitative Finance | Machine Learning | Financial Analytics

GitHub: [2003UJAN](https://github.com/2003UJAN)