import warnings
import os
import pandas as pd
import numpy as np
from itertools import product

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# Import from harsh_tester
from harsh_tester import HarshTester

# ============================================================================
# PREDEFINED STRATEGIES
# ============================================================================

PREDEFINED_STRATEGIES = {
    "Momentum (Fast/Slow MA)": """
import pandas as pd
import numpy as np

class Strategy:
    def __init__(self, data, params=None):
        self.data = data
        self.params = params or {}
    
    def generate_signals(self):
        fast = self.params.get('fast', 63)
        slow = self.params.get('slow', 126)
        
        fast_ma = self.data.rolling(fast).mean()
        slow_ma = self.data.rolling(slow).mean()
        
        entries = (fast_ma > slow_ma).shift(1).fillna(False).astype(bool)
        exits = (fast_ma <= slow_ma).shift(1).fillna(False).astype(bool)
        
        return entries, exits
""",
    "Mean Reversion (Z-Score)": """
import pandas as pd
import numpy as np

class Strategy:
    def __init__(self, data, params=None):
        self.data = data
        self.params = params or {}
    
    def generate_signals(self):
        lookback = self.params.get('lookback', 20)
        z_entry = self.params.get('z_entry', 1.0)
        z_exit = self.params.get('z_exit', 0.5)
        
        rolling_mean = self.data.rolling(lookback).mean()
        rolling_std = self.data.rolling(lookback).std()
        z_score = (self.data - rolling_mean) / (rolling_std + 1e-6)
        
        entries = (z_score < -z_entry).shift(1).fillna(False).astype(bool)
        exits = (z_score < -z_exit).shift(1).fillna(False).astype(bool)
        
        return entries, exits
""",
    "RSI (Overbought/Oversold)": """
import pandas as pd
import numpy as np

class Strategy:
    def __init__(self, data, params=None):
        self.data = data
        self.params = params or {}
    
    def generate_signals(self):
        period = self.params.get('period', 14)
        oversold = self.params.get('oversold', 30)
        overbought = self.params.get('overbought', 70)
        
        delta = self.data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        
        rs = gain / (loss + 1e-6)
        rsi = 100 - (100 / (1 + rs))
        
        entries = (rsi < oversold).shift(1).fillna(False).astype(bool)
        exits = (rsi > overbought).shift(1).fillna(False).astype(bool)
        
        return entries, exits
""",
    "Bollinger Bands Breakout": """
import pandas as pd
import numpy as np

class Strategy:
    def __init__(self, data, params=None):
        self.data = data
        self.params = params or {}
    
    def generate_signals(self):
        period = self.params.get('period', 20)
        num_std = self.params.get('num_std', 2.0)
        
        sma = self.data.rolling(period).mean()
        std = self.data.rolling(period).std()
        
        upper_band = sma + num_std * std
        lower_band = sma - num_std * std
        
        entries = (self.data < lower_band).shift(1).fillna(False).astype(bool)
        exits = (self.data > upper_band).shift(1).fillna(False).astype(bool)
        
        return entries, exits
""",
    "MACD Crossover": """
import pandas as pd
import numpy as np

class Strategy:
    def __init__(self, data, params=None):
        self.data = data
        self.params = params or {}
    
    def generate_signals(self):
        fast_period = self.params.get('fast_period', 12)
        slow_period = self.params.get('slow_period', 26)
        signal_period = self.params.get('signal_period', 9)
        
        ema_fast = self.data.ewm(span=fast_period).mean()
        ema_slow = self.data.ewm(span=slow_period).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period).mean()
        
        entries = (macd > signal).shift(1).fillna(False).astype(bool)
        exits = (macd < signal).shift(1).fillna(False).astype(bool)
        
        return entries, exits
""",
}

# ============================================================================
# STREAMLIT PAGE CONFIG
# ============================================================================

st.set_page_config(page_title="Harsh Strategy Tester", layout="wide")
st.title("🔬 Harsh Strategy Tester - Production Grade v4.1 (All 10 Tests)")

# ============================================================================
# SIDEBAR: DATA SETUP
# ============================================================================

st.sidebar.header("📊 Data Setup")

symbols_input = st.sidebar.text_area(
    "Asset Universe (comma-separated, e.g., SPY,TLT,GLD)",
    value="SPY,TLT,GLD",
    height=80
)
symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime('2024-12-31'))

# Load data
@st.cache_data
def load_data(symbols_list, start, end):
    data = yf.download(symbols_list, start=start, end=end)['Close']
    if len(symbols_list) == 1:
        data = data.to_frame(name=symbols_list[0])
    return data.dropna()

try:
    data = load_data(symbols, start_date, end_date)
    st.sidebar.success(f"✅ Loaded {len(data)} days for {len(symbols)} assets")
except Exception as e:
    st.sidebar.error(f"❌ Error loading data: {e}")
    st.stop()

# ============================================================================
# MAIN: STRATEGY SELECTION
# ============================================================================

st.header("1️⃣ Strategy Selection")

strategy_choice = st.selectbox(
    "Select strategy from dropdown",
    options=list(PREDEFINED_STRATEGIES.keys()),
    index=0,
    help="Choose a predefined strategy"
)

strategy_code = PREDEFINED_STRATEGIES[strategy_choice]

with st.expander(f"View {strategy_choice}"):
    st.code(strategy_code, language='python')

# ============================================================================
# OPTIMIZATION GRID
# ============================================================================

st.header("2️⃣ Optimization Grid")

col1, col2, col3 = st.columns(3)

with col1:
    param1_name = st.text_input("Param 1 Name", value="fast")
    param1_vals = st.text_input("Param 1 Values (comma-separated)", value="63,126")
    try:
        param1_list = [int(x.strip()) for x in param1_vals.split(',') if x.strip()]
    except:
        param1_list = [63, 126]

with col2:
    param2_name = st.text_input("Param 2 Name", value="slow")
    param2_vals = st.text_input("Param 2 Values (comma-separated)", value="126,252")
    try:
        param2_list = [int(x.strip()) for x in param2_vals.split(',') if x.strip()]
    except:
        param2_list = [126, 252]

with col3:
    param3_name = st.text_input("Param 3 Name", value="threshold")
    param3_vals = st.text_input("Param 3 Values (comma-separated)", value="0")
    try:
        param3_list = [float(x.strip()) for x in param3_vals.split(',') if x.strip()]
    except:
        param3_list = [0]

param_grid = {
    param1_name: param1_list,
    param2_name: param2_list,
    param3_name: param3_list
}

# ============================================================================
# PORTFOLIO SETUP
# ============================================================================

st.header("3️⃣ Portfolio Setup")

col1, col2, col3 = st.columns(3)

with col1:
    init_cash = st.number_input("Initial Capital ($)", value=100000, min_value=1000)

with col2:
    allocation_method = st.selectbox("Allocation Method", ["equal_weight"])

with col3:
    fee_bps = st.slider("Transaction Fee (bps)", 0, 50, 3)
    fee = fee_bps / 10000

# ============================================================================
# RUN TESTS
# ============================================================================

st.header("4️⃣ Run Tests")

col1, col2 = st.columns([3, 1])
with col1:
    run_tests = st.button("🚀 Run Complete Test Suite (All 10 Tests)", use_container_width=True)
with col2:
    debug_mode = st.checkbox("🐛 Debug Output")

if run_tests:
    tester = HarshTester(data, init_cash=init_cash, fee=fee, allocation_method=allocation_method)
    tester.debug = debug_mode

    # ======================================================================
    # TEST 1: STRESS TEST
    # ======================================================================
    st.subheader("Test 1️⃣: Grid Search Backtest")
    
    with st.spinner("Running grid search..."):
        grid_results = tester.stress_test(strategy_code, param_grid)
    
    if not grid_results.empty:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Best Sharpe", f"{grid_results['sharpe'].max():.2f}")
        col2.metric("Best Sortino", f"{grid_results['sortino'].max():.2f}" if grid_results['sortino'].max() < 1e10 else "EXTREME")
        col3.metric("Avg Sharpe", f"{grid_results['sharpe'].mean():.2f}")
        col4.metric("Best Return", f"{grid_results['total_return'].max()*100:.2f}%")
        col5.metric("Min Max DD", f"{grid_results['max_dd'].min():.2%}")
        
        st.dataframe(grid_results.sort_values('sharpe', ascending=False), use_container_width=True)
        
        if hasattr(tester, 'best_stats') and tester.best_stats is not None:
            st.write("**Equity Curve - Best Performer:**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=tester.best_stats['dates'],
                y=tester.best_stats['equity_curve'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#1f77b4', width=2)
            ))
            fig.update_layout(
                title="Equity Curve",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Grid search returned no results")

    # ======================================================================
    # TEST 2: WALK-FORWARD VALIDATION
    # ======================================================================
    st.subheader("Test 2️⃣: Walk-Forward Validation")
    
    with st.spinner("Running walk-forward analysis..."):
        try:
            wf_results = tester.walk_forward_test(strategy_code, param_grid, train_years=2, test_years=1)
            if not wf_results.empty:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Avg Test Sharpe", f"{wf_results['sharpe'].mean():.2f}")
                col2.metric("Sharpe Std Dev", f"{wf_results['sharpe'].std():.2f}")
                col3.metric("Avg Max DD", f"{wf_results['max_dd'].mean():.2%}")
                col4.metric("Windows", wf_results['window'].nunique())
                st.dataframe(wf_results, use_container_width=True)
            else:
                st.info("Insufficient data for walk-forward analysis")
        except Exception as e:
            st.warning(f"Walk-forward test error: {str(e)[:100]}")

    # ======================================================================
    # TEST 3: MONTE CARLO STRESS TEST
    # ======================================================================
    st.subheader("Test 3️⃣: Monte Carlo Stress Test (500 Sims)")
    
    with st.spinner("Running Monte Carlo simulations..."):
        try:
            mc_results = tester.monte_carlo_test(strategy_code, param_grid, n_sims=500)
            if not mc_results.empty:
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                col1.metric("MC Sharpe 5%", f"{mc_results['mc_sharpe_5pct'].mean():.2f}")
                col2.metric("MC Sharpe Mean", f"{mc_results['mc_sharpe_mean'].mean():.2f}")
                col3.metric("MC Sharpe 95%", f"{mc_results['mc_sharpe_95pct'].mean():.2f}")
                col4.metric("MC DD 5%", f"{mc_results['mc_dd_5pct'].mean():.2%}")
                col5.metric("MC DD Mean", f"{mc_results['mc_dd_mean'].mean():.2%}")
                col6.metric("MC DD 95%", f"{mc_results['mc_dd_95pct'].mean():.2%}")
                st.dataframe(mc_results, use_container_width=True)
            else:
                st.info("No Monte Carlo results")
        except Exception as e:
            st.warning(f"Monte Carlo test error: {str(e)[:100]}")

    # ======================================================================
    # TEST 4: PARAMETER SENSITIVITY
    # ======================================================================
    st.subheader("Test 4️⃣: Parameter Sensitivity (±20%)")
    
    with st.spinner("Running parameter sensitivity tests..."):
        try:
            sens = tester.parameter_sensitivity_test(strategy_code, param_grid, perturbation=0.2)
            if not sens.empty:
                st.dataframe(sens, use_container_width=True)
            else:
                st.warning("No sensitivity results")
        except Exception as e:
            st.error(f"Test 4 Error: {str(e)[:150]}")

    # ======================================================================
    # TEST 5: TRANSACTION COST IMPACT
    # ======================================================================
    st.subheader("Test 5️⃣: Transaction Cost Impact")
    
    with st.spinner("Analyzing transaction costs..."):
        try:
            cost_results = tester.transaction_cost_test(strategy_code, param_grid)
            if not cost_results.empty:
                st.dataframe(cost_results, use_container_width=True)
            else:
                st.info("No transaction cost results")
        except Exception as e:
            st.warning(f"Transaction cost test error: {str(e)[:100]}")

    # ======================================================================
    # TEST 6: ROLLING METRICS
    # ======================================================================
    st.subheader("Test 6️⃣: Rolling Metrics (1-Year Windows)")
    
    with st.spinner("Calculating rolling metrics..."):
        try:
            rolling_results = tester.rolling_metrics_test(strategy_code, param_grid, window_years=1)
            if not rolling_results.empty:
                st.write(f"Periods analyzed: {rolling_results['period'].nunique()}")
                st.dataframe(rolling_results, use_container_width=True)
            else:
                st.info("No rolling metrics results")
        except Exception as e:
            st.warning(f"Rolling metrics error: {str(e)[:100]}")

    # ======================================================================
    # TEST 7: CRISIS STRESS TESTS
    # ======================================================================
    st.subheader("Test 7️⃣: Crisis Stress Tests")
    
    with st.spinner("Running historical crisis tests..."):
        try:
            crisis_results = tester.crisis_stress_test(strategy_code, param_grid)
            if not crisis_results.empty:
                st.write(f"Crisis periods tested: {crisis_results['crisis'].nunique()}")
                st.dataframe(crisis_results, use_container_width=True)
            else:
                st.info("No crisis data in selected date range")
        except Exception as e:
            st.error(f"Crisis stress test error: {str(e)[:200]}")

    # ======================================================================
    # TEST 8: MONTHLY CONSISTENCY
    # ======================================================================
    st.subheader("Test 8️⃣: Monthly Returns Consistency")
    
    with st.spinner("Analyzing monthly consistency..."):
        monthly_results, best_monthly_stats = tester.monthly_consistency_test(strategy_code, param_grid)
    
    if not monthly_results.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Monthly Return", f"{monthly_results['avg_monthly_return'].mean()*100:.2f}%")
        col2.metric("Monthly Volatility", f"{monthly_results['monthly_volatility'].mean()*100:.2f}%")
        col3.metric("Best Month Return", f"{monthly_results['avg_monthly_return'].max()*100:.2f}%")
        
        st.dataframe(monthly_results, use_container_width=True)
        
        # ✅ FIXED 2D HEATMAP - NO MORE xaxis DICT ERROR
        if best_monthly_stats is not None and best_monthly_stats.get('equity_curve') is not None and len(best_monthly_stats['equity_curve']) > 21:
            st.write("**Monthly Returns Heatmap - Best Performer:**")
            equity = np.array(best_monthly_stats['equity_curve'], dtype=float)
            monthly_rets = []
            for i in range(0, len(equity) - 21, 21):
                month_start = equity[i]
                month_end = equity[min(i + 21, len(equity) - 1)]
                if month_start > 0:
                    ret = ((month_end - month_start) / month_start * 100)
                    monthly_rets.append(ret)
            
            if len(monthly_rets) > 0:
                # RESHAPE TO 2D: (years, months) for better visualization
                n_months_per_year = 12
                n_years = (len(monthly_rets) + n_months_per_year - 1) // n_months_per_year
                padded_rets = monthly_rets + [np.nan] * (n_years * n_months_per_year - len(monthly_rets))
                heatmap_data = np.array(padded_rets).reshape(n_years, n_months_per_year)
                
                fig = go.Figure(data=go.Heatmap(
                    z=heatmap_data,
                    colorscale='RdYlGn',
                    zmid=0,
                    text=[[f"{v:.2f}%" if not np.isnan(v) else "" for v in row] for row in heatmap_data],
                    texttemplate="%{text}",
                    textfont={"size": 9},
                    hoverongaps=False,
                    showscale=True,
                    colorbar=dict(title="Return %")
                ))
                fig.update_layout(
                    height=200 + n_years * 20,
                    xaxis_title="Month",
                    yaxis_title="Year",
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                fig.update_xaxes(side="bottom")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No monthly data available for heatmap")
    else:
        st.info("No monthly consistency results")

    # ======================================================================
    # TEST 9: DRAWDOWN ANALYSIS
    # ======================================================================
    st.subheader("Test 9️⃣: Drawdown Analysis (Best Performer)")
    
    with st.spinner("Analyzing drawdowns..."):
        dd_results, best_dd_stats = tester.drawdown_analysis_test(strategy_code, param_grid)
    
    if not dd_results.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sharpe", f"{dd_results['sharpe'].values[0]:.2f}")
        sortino_val = dd_results['sortino'].values[0]
        col2.metric("Sortino", f"{sortino_val:.2f}" if sortino_val < 1e10 else "EXTREME")
        col3.metric("Max DD", f"{dd_results['max_dd'].values[0]:.2%}")
        col4.metric("Total Return", f"{dd_results['total_return'].values[0]:.2%}")
        
        st.dataframe(dd_results, use_container_width=True)
        
        if best_dd_stats is not None and best_dd_stats.get('drawdowns') is not None and len(best_dd_stats['drawdowns']) > 0:
            st.write("**Drawdown Chart - Best Performer:**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=best_dd_stats['dates'],
                y=best_dd_stats['drawdowns'],
                fill='tozeroy',
                name='Drawdown',
                line=dict(color='#d62728'),
                fillcolor='rgba(214, 39, 40, 0.5)'
            ))
            fig.update_layout(
                title="Drawdown Over Time",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No drawdown analysis results")

    # ======================================================================
    # TEST 10: KELLY CRITERION
    # ======================================================================
    st.subheader("Test 🔟: Kelly Criterion (Optimal Position Sizing)")
    
    with st.spinner("Calculating Kelly criterion..."):
        try:
            kelly = tester.kelly_criterion_test(strategy_code, param_grid)
            if not kelly.empty:
                col1, col2, col3 = st.columns(3)
                col1.metric("Full Kelly", f"{kelly['full_kelly'].values[0]:.1f}%")
                col2.metric("Half Kelly", f"{kelly['half_kelly'].values[0]:.1f}%")
                col3.metric("Quarter Kelly", f"{kelly['quarter_kelly'].values[0]:.1f}%")
                st.write(f"**Status:** {kelly['status'].values[0]}")
                st.dataframe(kelly, use_container_width=True)
            else:
                st.info("No Kelly criterion results")
        except Exception as e:
            st.warning(f"Kelly criterion error: {str(e)[:100]}")

    st.success("✅ All 10 tests completed!")
