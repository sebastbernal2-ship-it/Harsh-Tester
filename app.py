# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
from harsh_tester import HarshTester
import json
import numpy as np

st.set_page_config(page_title="Harsh Strategy Tester", layout="wide")
st.title("🎯 Harsh Strategy Tester")

# Sidebar: Data inputs
st.sidebar.header("Data Setup")
symbols_input = st.sidebar.text_area(
    "Asset Universe (comma-separated, e.g., SPY,TLT,GLD,QQQ)",
    "SPY,TLT,GLD",
    height=100
)
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-31"))

# Load data
@st.cache_data
def load_data(symbols_list, start, end):
    data = yf.download(symbols_list, start=start, end=end)["Close"]
    if len(symbols_list) == 1:
        data = data.to_frame(name=symbols_list[0])
    return data.dropna()

try:
    data = load_data(symbols, start_date, end_date)
    st.sidebar.success(f"Loaded {len(data)} days for {len(symbols)} assets: {', '.join(symbols)}")
except Exception as e:
    st.sidebar.error(f"Error loading data: {e}")
    st.stop()

# Main: Strategy code input
st.header("1️⃣ Paste Your Strategy Code")
strategy_code = st.text_area(
    "Paste your Strategy class (must inherit BaseStrategy, implement generate_signals)",
    height=300,
    value="""from base_strategy import BaseStrategy
import pandas as pd
import numpy as np

class Strategy(BaseStrategy):
    def generate_signals(self):
        lookback = self.params.get("lookback", 20)
        z_entry = self.params.get("z_entry", 1.0)
        z_exit = self.params.get("z_exit", 0.5)
        
        rolling_mean = self.data.rolling(lookback).mean()
        rolling_std = self.data.rolling(lookback).std()
        z_score = (self.data - rolling_mean) / (rolling_std + 1e-6)
        
        entries = (z_score < -z_entry).shift(1).fillna(False).astype(bool)
        exits = (z_score > -z_exit).shift(1).fillna(False).astype(bool)
        
        return entries, exits
"""
)

# Parameters
st.header("2️⃣ Optimization Grid")
col1, col2, col3 = st.columns(3)

with col1:
    param_1_name = st.text_input("Param 1 Name", "lookback")
    param_1_vals = st.text_input("Param 1 Values (comma-separated)", "10,20,30")
    try:
        param_1_list = [int(x.strip()) for x in param_1_vals.split(",") if x.strip()]
    except:
        param_1_list = [10, 20, 30]

with col2:
    param_2_name = st.text_input("Param 2 Name", "z_entry")
    param_2_vals = st.text_input("Param 2 Values (comma-separated)", "1.0,1.5,2.0")
    try:
        param_2_list = [float(x.strip()) for x in param_2_vals.split(",") if x.strip()]
    except:
        param_2_list = [1.0, 1.5, 2.0]

with col3:
    param_3_name = st.text_input("Param 3 Name", "z_exit")
    param_3_vals = st.text_input("Param 3 Values (comma-separated)", "0.0,0.5,1.0")
    try:
        param_3_list = [float(x.strip()) for x in param_3_vals.split(",") if x.strip()]
    except:
        param_3_list = [0.0, 0.5, 1.0]

param_grid = {param_1_name: param_1_list, param_2_name: param_2_list, param_3_name: param_3_list}

# Strategy-specific tunable parameters
st.header("2b️⃣ Strategy-Specific Thresholds & Parameters")
st.write("Define thresholds, z-scores, or any other tunable parameters as JSON")

strategy_params_input = st.text_area(
    "Strategy Parameters (JSON format)",
    value='{"threshold": [0.0, 0.5, 1.0]}',
    height=100
)

try:
    strategy_param_grid = json.loads(strategy_params_input)
    param_grid.update(strategy_param_grid)
except json.JSONDecodeError:
    st.error("Invalid JSON. Use format: {\"param_name\": [value1, value2, value3]}")
    strategy_param_grid = {}

# Portfolio settings
st.header("3️⃣ Portfolio Setup")
col1, col2 = st.columns(2)

with col1:
    init_cash = st.number_input("Initial Capital ($)", value=100000, min_value=1000)
    
with col2:
    allocation_method = st.selectbox(
        "Allocation Method",
        ["equal_weight", "signal_weight"]
    )

# Run all tests
st.header("4️⃣ Run All Harsh Tests")
if st.button("🚀 Run Complete Harsh Test Suite"):
    tester = HarshTester(data, init_cash=init_cash, allocation_method=allocation_method)
    
    # Test 1: Grid Search
    st.subheader("📊 Test 1: Grid Search Backtest")
    with st.spinner("Running grid search..."):
        grid_results = tester.stress_test(strategy_code, param_grid)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Sharpe", f"{grid_results['sharpe'].max():.2f}")
    col2.metric("Avg Sharpe", f"{grid_results['sharpe'].mean():.2f}")
    col3.metric("Min Max DD", f"{grid_results['max_dd'].min():.2%}")
    col4.metric("Best Return", f"{grid_results['total_return'].max():.2%}")
    
    st.dataframe(grid_results.sort_values("sharpe", ascending=False), width='stretch')
    st.line_chart(grid_results.set_index(param_1_name)["sharpe"])
    
    # Test 2: Walk-Forward Analysis
    st.subheader("📈 Test 2: Walk-Forward Validation")
    with st.spinner("Running walk-forward analysis..."):
        try:
            wf_results, wf_summary = tester.walk_forward_test(strategy_code, param_grid, train_years=2, test_years=1)
            
            if len(wf_results) > 0:
                st.write("**Walk-Forward Results by Window:**")
                st.dataframe(wf_results, width='stretch')
                
                st.write("**Walk-Forward Summary Statistics:**")
                st.dataframe(wf_summary, width='stretch')
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Avg Test Sharpe", f"{wf_summary['avg_test_sharpe'].values[0]:.2f}")
                col2.metric("Sharpe Std Dev", f"{wf_summary['std_test_sharpe'].values[0]:.2f}")
                col3.metric("Avg Max DD", f"{wf_summary['avg_max_dd'].values[0]:.2%}")
                col4.metric("Sharpe Degradation", f"{wf_summary['sharpe_degradation'].values[0]:.1%}")
                
                wf_chart_data = wf_results[['window', 'train_sharpe', 'sharpe']].set_index('window')
                st.line_chart(wf_chart_data)
            else:
                st.warning("Insufficient data for walk-forward analysis. Try shorter date range or fewer test years.")
        except Exception as e:
            st.warning(f"Walk-forward test skipped: {e}")
    
    # Test 3: Monte Carlo Simulation
    st.subheader("🎲 Test 3: Monte Carlo Stress Test (1000 Simulations)")
    with st.spinner("Running Monte Carlo simulations..."):
        try:
            mc_results = tester.monte_carlo_test(strategy_code, param_grid, n_sims=1000)
            
            st.write("**Monte Carlo Distribution of Returns:**")
            st.dataframe(mc_results, width='stretch')
            
            # Create meaningful visualization
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Sharpe Ratio: 95% Confidence Interval**")
                mc_sharpe_ci = pd.DataFrame({
                    '5th Percentile': mc_results['mc_sharpe_5pct'],
                    'Mean': mc_results['mc_sharpe_mean'],
                    '95th Percentile': mc_results['mc_sharpe_95pct'],
                    'Base Sharpe': mc_results['base_sharpe']
                })
                st.bar_chart(mc_sharpe_ci.head(10))
                
                st.write("**Percentile Values:**")
                st.dataframe(mc_sharpe_ci.head(10), width='stretch')
            
            with col2:
                st.write("**Max Drawdown: 95% Confidence Interval**")
                mc_dd_ci = pd.DataFrame({
                    '5th Percentile': mc_results['mc_dd_5pct'],
                    'Mean': mc_results['mc_dd_mean'],
                    '95th Percentile': mc_results['mc_dd_95pct'],
                    'Base Max DD': mc_results['base_max_dd']
                })
                st.bar_chart(mc_dd_ci.head(10))
                
                st.write("**Percentile Values:**")
                st.dataframe(mc_dd_ci.head(10), width='stretch')
            
            # Summary statistics
            st.write("**Monte Carlo Summary (across all parameter combos):**")
            mc_summary = pd.DataFrame({
                'Metric': ['Sharpe 5%', 'Sharpe Mean', 'Sharpe 95%', 'Max DD 5%', 'Max DD Mean', 'Max DD 95%'],
                'Min': [
                    mc_results['mc_sharpe_5pct'].min(),
                    mc_results['mc_sharpe_mean'].min(),
                    mc_results['mc_sharpe_95pct'].min(),
                    mc_results['mc_dd_5pct'].min(),
                    mc_results['mc_dd_mean'].min(),
                    mc_results['mc_dd_95pct'].min()
                ],
                'Mean': [
                    mc_results['mc_sharpe_5pct'].mean(),
                    mc_results['mc_sharpe_mean'].mean(),
                    mc_results['mc_sharpe_95pct'].mean(),
                    mc_results['mc_dd_5pct'].mean(),
                    mc_results['mc_dd_mean'].mean(),
                    mc_results['mc_dd_95pct'].mean()
                ],
                'Max': [
                    mc_results['mc_sharpe_5pct'].max(),
                    mc_results['mc_sharpe_mean'].max(),
                    mc_results['mc_sharpe_95pct'].max(),
                    mc_results['mc_dd_5pct'].max(),
                    mc_results['mc_dd_mean'].max(),
                    mc_results['mc_dd_95pct'].max()
                ]
            })
            st.dataframe(mc_summary, width='stretch')
            
        except Exception as e:
            st.warning(f"Monte Carlo test error: {e}")
    
    # Test 4: Parameter Sensitivity
    st.subheader("⚙️ Test 4: Parameter Sensitivity Analysis (±20%)")
    with st.spinner("Running parameter sensitivity tests..."):
        try:
            sensitivity_results = tester.parameter_sensitivity_test(strategy_code, param_grid, perturbation=0.2)
            
            st.write("**Sensitivity Results:**")
            st.dataframe(sensitivity_results, width='stretch')
            
            sensitivity_summary = sensitivity_results.groupby('perturbed_param').agg({
                'sharpe_change': ['mean', 'std'],
                'dd_change': ['mean', 'std']
            }).round(3)
            
            st.write("**Parameter Robustness Summary (avg % change per ±20% perturbation):**")
            st.dataframe(sensitivity_summary, width='stretch')
            
            sensitivity_pivot = sensitivity_results.pivot_table(
                values='sharpe_change',
                index='perturbed_param',
                columns='perturbation',
                aggfunc='mean'
            )
            st.bar_chart(sensitivity_pivot)
        except Exception as e:
            st.warning(f"Parameter sensitivity test error: {e}")
    
    # Test 5: Historical Stress Tests
    st.subheader("📉 Test 5: Historical Crisis Stress Tests")
    with st.spinner("Running historical stress tests..."):
        try:
            stress_results = tester.stress_test_historical(strategy_code, param_grid)
            
            if len(stress_results) > 0:
                st.write("**Performance During Historical Crises:**")
                st.dataframe(stress_results, width='stretch')
                
                crisis_summary = stress_results.groupby('crisis')[['sharpe', 'max_dd', 'total_return']].mean()
                st.write("**Average Metrics by Crisis:**")
                st.dataframe(crisis_summary, width='stretch')
                
                crisis_chart = stress_results.pivot_table(
                    values='sharpe',
                    index='crisis',
                    aggfunc='mean'
                )
                st.bar_chart(crisis_chart)
            else:
                st.warning("No historical crisis data available in selected date range.")
        except Exception as e:
            st.warning(f"Historical stress test error: {e}")

st.info("""
### What These Tests Tell You:

1. **Grid Search**: Which parameter combinations work best in-sample
2. **Walk-Forward**: How stable is your edge out-of-sample? Does Sharpe degrade significantly?
3. **Monte Carlo**: What's the tail risk? 95% confidence intervals on returns
4. **Parameter Sensitivity**: How fragile is the strategy to small changes? Robust strategies stay stable ±20%
5. **Historical Crises**: Does the strategy survive real market shocks? Or does it blow up?

**Red Flags:**
- Walk-forward Sharpe much lower than in-sample → overfitting
- Monte Carlo 95% CI includes negative returns → tail risk
- Parameter sensitivity >50% change → fragile, not robust
- Crashes during 2008/2020 → not truly diversified
""")
