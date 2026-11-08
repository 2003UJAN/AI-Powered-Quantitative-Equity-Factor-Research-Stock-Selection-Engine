
import streamlit as st

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Quantitative Equity Research",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM DESIGN
# ==========================================================

st.markdown("""
<style>

/* Main application */
.stApp {
    background: #0B1120;
    color: #E2E8F0;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111B2E;
    border-right: 1px solid #263750;
}

[data-testid="stSidebar"] * {
    color: #CBD5E1;
}

/* Headings */
h1, h2, h3 {
    color: #F8FAFC !important;
    font-weight: 650 !important;
}

/* Paragraphs */
p, label {
    color: #CBD5E1;
}

/* Hero */
.hero {
    background: linear-gradient(110deg, #14243D, #102D43);
    border: 1px solid #284765;
    border-radius: 14px;
    padding: 28px 30px;
    margin-bottom: 25px;
}

.hero-title {
    font-size: 31px;
    font-weight: 750;
    color: #F8FAFC;
}

.hero-subtitle {
    color: #AFC3D8;
    font-size: 14px;
    margin-top: 10px;
}

/* KPI cards */
.kpi-card {
    background: #17253A;
    border: 1px solid #2B405B;
    border-radius: 12px;
    padding: 20px;
    min-height: 125px;
}

.kpi-label {
    color: #94A9C2;
    font-size: 13px;
}

.kpi-value {
    color: #38BDF8;
    font-size: 25px;
    font-weight: 750;
    margin-top: 10px;
}

.kpi-description {
    color: #94A9C2;
    font-size: 12px;
    margin-top: 5px;
}

/* Research module cards */
.module-card {
    background: #142137;
    border: 1px solid #2B405B;
    border-radius: 12px;
    padding: 20px;
    min-height: 150px;
    margin-bottom: 12px;
}

.module-title {
    color: #F1F5F9;
    font-size: 17px;
    font-weight: 650;
    margin-bottom: 10px;
}

.module-description {
    color: #A9B9CC;
    font-size: 13px;
    line-height: 1.6;
}

/* Dividers */
hr {
    border-color: #293B54 !important;
}

/* Buttons */
.stButton > button {
    background: #0284C7;
    color: white;
    border: none;
    border-radius: 7px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #0369A1;
    color: white;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #2B405B;
    border-radius: 9px;
}

/* Sidebar radio and navigation readability */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #CBD5E1 !important;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# SIDEBAR BRANDING
# ==========================================================

st.sidebar.markdown("""
<div style="padding: 12px 0 20px 0;">
    <div style="font-size:22px;font-weight:750;color:#38BDF8;">
        QUANT RESEARCH
    </div>
    <div style="font-size:12px;color:#94A9C2;margin-top:5px;">
        EQUITY FACTOR RESEARCH PLATFORM
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()

st.sidebar.markdown("**RESEARCH CONFIGURATION**")

universe = st.sidebar.selectbox(
    "Stock Universe",
    ["NIFTY 50", "NIFTY NEXT 50", "NIFTY 100"]
)

benchmark = st.sidebar.selectbox(
    "Benchmark",
    ["NIFTY 50", "NIFTY BANK", "NIFTY 100"]
)

period = st.sidebar.selectbox(
    "Historical Period",
    ["1 Year", "2 Years", "3 Years", "5 Years"]
)

st.sidebar.divider()

st.sidebar.caption(
    "Systematic Equity Research | Financial Econometrics | Machine Learning"
)


# ==========================================================
# MAIN HEADER
# ==========================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">
        Quantitative Equity Research Terminal
    </div>
    <div class="hero-subtitle">
        SYSTEMATIC STOCK SELECTION &nbsp;|&nbsp;
        FACTOR INVESTING &nbsp;|&nbsp;
        MACHINE LEARNING &nbsp;|&nbsp;
        PORTFOLIO ANALYTICS
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# MARKET CONFIGURATION
# ==========================================================

st.markdown("### Research Overview")

col1, col2, col3, col4 = st.columns(4)

def kpi(label, value, description):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-description">{description}</div>
    </div>
    """, unsafe_allow_html=True)

with col1:
    kpi("STOCK UNIVERSE", universe, "Selected equity universe")

with col2:
    kpi("BENCHMARK", benchmark, "Reference market index")

with col3:
    kpi("HISTORICAL PERIOD", period, "Research lookback")

with col4:
    kpi("MODEL FRAMEWORK", "6 FACTORS", "Systematic factor framework")


st.divider()


# ==========================================================
# RESEARCH MODULES
# ==========================================================

st.markdown("### Quantitative Research Modules")

st.caption(
    "Navigate through the research modules using the pages in the sidebar."
)

modules = [
    (
        "01",
        "Market Universe",
        "Equity universe selection, historical prices, trading volume and fundamental data."
    ),
    (
        "02",
        "Factor Analysis",
        "Momentum, Value, Quality, Growth, Low Volatility and Liquidity."
    ),
    (
        "03",
        "Factor Research",
        "Information Coefficient, IC Information Ratio and quintile portfolio analysis."
    ),
    (
        "04",
        "ML Stock Selection",
        "Regression, classification and machine learning-based equity selection."
    ),
    (
        "05",
        "Stock Ranking",
        "Composite factor scores, customizable weights and quantitative stock rankings."
    ),
    (
        "06",
        "Portfolio Backtesting",
        "Portfolio construction, rebalancing, transaction costs and benchmark comparison."
    )
]

for start in range(0, len(modules), 3):

    cols = st.columns(3)

    for col, item in zip(cols, modules[start:start + 3]):

        number, title, description = item

        with col:
            st.markdown(f"""
            <div class="module-card">
                <div style="color:#38BDF8;font-size:12px;font-weight:700;">
                    MODULE {number}
                </div>
                <div class="module-title">{title}</div>
                <div class="module-description">
                    {description}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ==========================================================
# FACTOR FRAMEWORK
# ==========================================================

st.divider()

st.markdown("### Equity Factor Framework")

factor_data = [
    ("Momentum", "Historical price performance"),
    ("Value", "Relative valuation"),
    ("Quality", "Profitability and financial strength"),
    ("Low Volatility", "Historical price risk"),
    ("Growth", "Revenue and earnings growth"),
    ("Liquidity", "Trading activity and market liquidity")
]

cols = st.columns(3)

for index, (factor, description) in enumerate(factor_data):

    with cols[index % 3]:
        st.markdown(f"""
        <div style="
            background:#111D30;
            border:1px solid #293B54;
            border-radius:8px;
            padding:13px;
            margin-bottom:10px;">
            <div style="color:#38BDF8;font-weight:650;">
                {factor}
            </div>
            <div style="font-size:12px;color:#A9B9CC;margin-top:5px;">
                {description}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.markdown("""
<div style="text-align:center;padding:10px;color:#71839B;font-size:12px;">
    QUANTITATIVE EQUITY FACTOR RESEARCH ENGINE
    <br>
    Python | Financial Econometrics | Machine Learning | Streamlit
    <br><br>
    Research and educational purposes only. Not investment advice.
</div>
""", unsafe_allow_html=True)
