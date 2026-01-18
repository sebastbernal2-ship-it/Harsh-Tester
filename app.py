import streamlit as st
import pandas as pd
import yfinance as yf
from harsh_tester import HarshTester
import json

st.set_page_config(page_title="Harsh Strategy Tester", layout="wide")
st.title("🎯 Harsh Strategy Tester")

# Sidebar: Data inputs
st.sidebar.header("Data Setup")
symbol = st.sidebar.text_input("Symbol (e.g., SPY)", "SPY")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-31"))

# Load data
@st.cache_data
def load_data(symbol, start, end):
    return yf.download(symbol, start=start, end=end)["Close"]

data = load_data(symbol, start_date, end_date)
st.sidebar.success(f"Loaded {len(data)} days of {symbol} data")

# Main: Strategy code input
st.header("1️. Paste Your Strategy Code")
strategy_code = st.text_area(
    "Paste your Strategy class (must inherit BaseStrategy, implement generate_signals)",
    height=300,
    value="""from base_strategy import BaseStrategy

class Strategy(BaseStrategy):
    def generate_signals(self):
        fast = self.data.rolling(self.params["fast"]).mean()
        slow = self.data.rolling(self.params["slow"]).mean()
        entries = (fast > slow).shift(1)
        exits = (fast < slow).shift(1)
        return entries, exits
"""
)

# Parameters
st.header("2️. Set Parameter Grid")
col1, col2 = st.columns(2)

with col1:
    param_1_name = st.text_input("Param 1 Name", "fast")
    param_1_vals = st.text_input("Param 1 Values (comma-separated)", "63,126")
    param_1_list = [int(x.strip()) for x in param_1_vals.split(",")]

with col2:
    param_2_name = st.text_input("Param 2 Name", "slow")
    param_2_vals = st.text_input("Param 2 Values (comma-separated)", "126,252")
    param_2_list = [int(x.strip()) for x in param_2_vals.split(",")]

param_grid = {param_1_name: param_1_list, param_2_name: param_2_list}

# Run tests
st.header("3️. Run Harsh Tests")
if st.button("🚀 Run Backtest"):
    with st.spinner("Running tests..."):
        tester = HarshTester(data, init_cash=100000)
        results_df = tester.stress_test(strategy_code, param_grid)
    
    # Display results
    st.subheader("Results Table")
    st.dataframe(results_df.sort_values("sharpe", ascending=False), width='stretch')

    
    # Statistics
    col1, col2, col3 = st.columns(3)
    col1.metric("Best Sharpe", f"{results_df['sharpe'].max():.2f}")
    col2.metric("Avg Sharpe", f"{results_df['sharpe'].mean():.2f}")
    col3.metric("Min Max DD", f"{results_df['max_dd'].min():.2%}")
    
    # Charts
    st.subheader("Visualizations")
    st.line_chart(results_df.set_index(param_1_name)["sharpe"])
