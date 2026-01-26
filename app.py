# app.py
import warnings
import os
import pandas as pd
import numpy as np

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from harsh_tester import HarshTester

# ============================================================================
# PREDEFINED STRATEGIES (cast window params to int for safety)
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
        fast = int(max(1, round(self.params.get('fast', 63))))
        slow = int(max(1, round(self.params.get('slow', 126))))
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
        lookback = int(max(1, round(self.params.get('lookback', 20))))
        z_entry = float(self.params.get('z_entry', 1.0))
        z_exit = float(self.params.get('z_exit', 0.5))
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
        period = int(max(1, round(self.params.get('period', 14))))
        oversold = float(self.params.get('oversold', 30))
        overbought = float(self.params.get('overbought', 70))
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
        period = int(max(1, round(self.params.get('period', 20))))
        num_std = float(self.params.get('num_std', 2.0))
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
        fast_period = int(max(1, round(self.params.get('fast_period', 12))))
        slow_period = int(max(1, round(self.params.get('slow_period', 26))))
        signal_period = int(max(1, round(self.params.get('signal_period', 9))))
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
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="Harsh Strategy Tester", layout="wide")
st.title("🔬 Harsh Strategy Tester — 20 Tests (Tradability-Focused)")

# ============================================================================
# SIDEBAR: DATA
# ============================================================================
st.sidebar.header("📊 Data Setup")
symbols_input = st.sidebar.text_area("Assets (comma-separated, e.g., SPY,TLT,GLD)", value="SPY,TLT,GLD", height=80)
symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime('2024-12-31'))

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
# STRATEGY SELECTION (with Custom)
# ============================================================================
st.header("1️⃣ Strategy Selection")

strategy_options = list(PREDEFINED_STRATEGIES.keys()) + ["Custom"]
choice = st.selectbox("Select or paste a strategy", options=strategy_options, index=0)

DEFAULT_CUSTOM = """
import pandas as pd
import numpy as np
class Strategy:
    def __init__(self, data, params=None):
        self.data = data
        self.params = params or {}
    def generate_signals(self):
        fast = int(max(1, round(self.params.get('fast', 20))))
        slow = int(max(1, round(self.params.get('slow', 50))))
        fast_ma = self.data.rolling(fast).mean()
        slow_ma = self.data.rolling(slow).mean()
        entries = (fast_ma > slow_ma).shift(1).fillna(False).astype(bool)
        exits = (fast_ma <= slow_ma).shift(1).fillna(False).astype(bool)
        return entries, exits
"""
strategy_code = PREDEFINED_STRATEGIES[choice] if choice != "Custom" else st.text_area(
    "Paste your custom Strategy code (must define class Strategy with generate_signals)",
    value=DEFAULT_CUSTOM, height=300
)
with st.expander(f"View {choice}"):
    st.code(strategy_code, language='python')

# ============================================================================
# PARAM GRID
# ============================================================================
st.header("2️⃣ Optimization Grid")
c1, c2, c3 = st.columns(3)
with c1:
    p1_name = st.text_input("Param 1 Name", value="fast")
    p1_vals = st.text_input("Param 1 Values", value="63,126")
    try:
        p1_list = [int(x.strip()) for x in p1_vals.split(',') if x.strip()]
    except:
        p1_list = [63, 126]
with c2:
    p2_name = st.text_input("Param 2 Name", value="slow")
    p2_vals = st.text_input("Param 2 Values", value="126,252")
    try:
        p2_list = [int(x.strip()) for x in p2_vals.split(',') if x.strip()]
    except:
        p2_list = [126, 252]
with c3:
    p3_name = st.text_input("Param 3 Name", value="threshold")
    p3_vals = st.text_input("Param 3 Values", value="0")
    try:
        p3_list = [float(x.strip()) for x in p3_vals.split(',') if x.strip()]
    except:
        p3_list = [0.0]
param_grid = {p1_name: p1_list, p2_name: p2_list, p3_name: p3_list}

# ============================================================================
# PORTFOLIO SETUP
# ============================================================================
st.header("3️⃣ Portfolio Setup")
c1, c2, c3 = st.columns(3)
with c1:
    init_cash = st.number_input("Initial Capital ($)", value=100000, min_value=1000)
with c2:
    allocation_method = st.selectbox("Allocation Method", ["equal_weight"])
with c3:
    fee_bps = st.slider("Transaction Fee (bps)", 0, 50, 3)
    fee = fee_bps / 10000.0

# ============================================================================
# RUN
# ============================================================================
st.header("4️⃣ Run Tests")
c1, c2 = st.columns([3, 1])
with c1:
    run_tests = st.button("🚀 Run Complete Suite (Tests 1–20)", type="primary")
with c2:
    debug_mode = st.checkbox("🐛 Debug Output")

def show_df(df: pd.DataFrame, formatters: dict | None = None):
    if df.empty:
        st.info("No results")
        return
    if formatters:
        st.dataframe(df.style.format(formatters), width='stretch')
    else:
        st.dataframe(df, width='stretch')

if run_tests:
    tester = HarshTester(data, init_cash=init_cash, fee=fee, allocation_method=allocation_method)
    tester.debug = debug_mode

    # Test 1
    st.subheader("Test 1️⃣: Grid Search Backtest")
    with st.spinner("Running grid search..."):
        t1 = tester.stress_test(strategy_code, param_grid)
    if not t1.empty:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Best Sharpe", f"{t1['sharpe'].max():.2f}")
        m2.metric("Best Sortino", f"{t1['sortino'].max():.2f}")
        m3.metric("Avg Sharpe", f"{t1['sharpe'].mean():.2f}")
        m4.metric("Best Return", f"{t1['total_return'].max()*100:.2f}%")
        m5.metric("Min Max DD", f"{t1['max_dd'].min():.2%}")
        show_df(t1.sort_values('sharpe', ascending=False))
        if tester.best_stats is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tester.best_stats['dates'], y=tester.best_stats['equity_curve'], mode='lines', name='Equity'))
            fig.update_layout(title="Equity Curve (Best Performer)", height=400)
            st.plotly_chart(fig, width='stretch')
    else:
        st.warning("Grid search returned no results")

    # Test 2
    st.subheader("Test 2️⃣: Walk-Forward Validation (simple)")
    with st.spinner("Running walk-forward analysis..."):
        try:
            t2 = tester.walk_forward_test(strategy_code, param_grid, train_years=2, test_years=1)
            show_df(t2)
        except Exception as e:
            st.warning(f"Walk-forward test error: {str(e)[:100]}")

    # Test 3
    st.subheader("Test 3️⃣: Monte Carlo Stress Test (500 Sims)")
    with st.spinner("Running Monte Carlo simulations..."):
        try:
            t3 = tester.monte_carlo_test(strategy_code, param_grid, n_sims=500)
            show_df(t3)
        except Exception as e:
            st.warning(f"Monte Carlo test error: {str(e)[:100]}")

    # Test 4
    st.subheader("Test 4️⃣: Parameter Sensitivity (±20%)")
    with st.spinner("Running parameter sensitivity..."):
        try:
            t4 = tester.parameter_sensitivity_test(strategy_code, param_grid, perturbation=0.2)
            show_df(t4)
        except Exception as e:
            st.error(f"Test 4 Error: {str(e)[:150]}")

    # Test 5
    st.subheader("Test 5️⃣: Transaction Cost Impact")
    with st.spinner("Analyzing transaction costs..."):
        try:
            t5 = tester.transaction_cost_test(strategy_code, param_grid)
            show_df(t5)
        except Exception as e:
            st.warning(f"Transaction cost test error: {str(e)[:100]}")

    # Test 6
    st.subheader("Test 6️⃣: Rolling Metrics (1-Year Windows)")
    with st.spinner("Calculating rolling metrics..."):
        try:
            t6 = tester.rolling_metrics_test(strategy_code, param_grid, window_years=1)
            show_df(t6)
        except Exception as e:
            st.warning(f"Rolling metrics error: {str(e)[:100]}")

    # Test 7
    st.subheader("Test 7️⃣: Crisis Stress Tests")
    with st.spinner("Running historical crisis tests..."):
        try:
            t7 = tester.crisis_stress_test(strategy_code, param_grid)
            show_df(t7)
        except Exception as e:
            st.error(f"Crisis stress test error: {str(e)[:200]}")

    # Test 8
    st.subheader("Test 8️⃣: Monthly Returns Consistency")
    with st.spinner("Analyzing monthly consistency..."):
        t8, best_monthly = tester.monthly_consistency_test(strategy_code, param_grid)
    if not t8.empty:
        show_df(t8)
        if best_monthly and len(best_monthly.get('equity_curve', [])) > 21:
            eq = np.array(best_monthly['equity_curve'], dtype=float)
            monthly_rets = []
            for i in range(0, len(eq) - 21, 21):
                start_val = eq[i]; end_val = eq[min(i + 21, len(eq) - 1)]
                if start_val > 0:
                    monthly_rets.append((end_val - start_val) / start_val * 100.0)
            if monthly_rets:
                n_months_per_year = 12
                n_years = (len(monthly_rets) + n_months_per_year - 1) // n_months_per_year
                padded = monthly_rets + [np.nan] * (n_years * n_months_per_year - len(monthly_rets))
                heat = np.array(padded).reshape(n_years, n_months_per_year)
                fig = go.Figure(data=go.Heatmap(z=heat, colorscale='RdYlGn', zmid=0, showscale=True))
                fig.update_layout(title="Monthly Returns Heatmap (Best)", height=200 + n_years * 20)
                st.plotly_chart(fig, width='stretch')
    else:
        st.info("No monthly consistency results")

    # Test 9
    st.subheader("Test 9️⃣: Drawdown Analysis (Best Performer)")
    with st.spinner("Analyzing drawdowns..."):
        t9, best_dd = tester.drawdown_analysis_test(strategy_code, param_grid)
    if not t9.empty:
        show_df(t9)
        if best_dd and best_dd.get('drawdowns'):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=best_dd['dates'], y=best_dd['drawdowns'], fill='tozeroy', name='Drawdown'))
            fig.update_layout(title="Drawdown Over Time", height=400)
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("No drawdown analysis results")

    # Test 10
    st.subheader("Test 🔟: Kelly Criterion (Optimal Sizing)")
    with st.spinner("Computing Kelly..."):
        try:
            t10 = tester.kelly_criterion_test(strategy_code, param_grid)
            show_df(t10)
        except Exception as e:
            st.warning(f"Kelly criterion error: {str(e)[:100]}")

    st.success("✅ First 10 tests completed!")

    # Test 11
    st.subheader("Test 1️⃣1️⃣: Conditional P/L Decomposition (EV Breakdown)")
    with st.spinner("Decomposing EV..."):
        try:
            t11 = tester.test_conditional_decomposition(strategy_code, param_grid)
            show_df(t11)
        except Exception as e:
            st.warning(f"Conditional decomposition error: {str(e)[:120]}")

    # Test 12
    st.subheader("Test 1️⃣2️⃣: Time Slice Stability")
    with st.spinner("Evaluating time-slice stability..."):
        try:
            t12 = tester.test_time_slice_stability(strategy_code, param_grid, n_slices=4)
            show_df(t12)
        except Exception as e:
            st.warning(f"Time-slice stability error: {str(e)[:120]}")

    # Test 13
    st.subheader("Test 1️⃣3️⃣: Regime Dependency (Volatility Terciles)")
    with st.spinner("Analyzing regime dependency..."):
        try:
            t13 = tester.test_regime_dependency(strategy_code, param_grid, regime_var='volatility')
            show_df(t13)
        except Exception as e:
            st.warning(f"Regime dependency error: {str(e)[:120]}")

    # Test 14
    st.subheader("Test 1️⃣4️⃣: Win/Loss Distribution Stability")
    with st.spinner("Measuring distribution stability..."):
        try:
            t14 = tester.test_win_loss_distribution_stability(strategy_code, param_grid, n_slices=2, min_samples=5)
            show_df(
                t14,
                formatters={
                    'winner_kde_divergence': lambda v: 'N/A' if pd.isna(v) else f'{v:.4f}',
                    'loser_kde_divergence': lambda v: 'N/A' if pd.isna(v) else f'{v:.4f}',
                    'combined_distribution_score': lambda v: 'N/A' if pd.isna(v) else f'{v:.4f}',
                }
            )
        except Exception as e:
            st.warning(f"Distribution stability error: {str(e)[:120]}")

    # Test 15
    st.subheader("Test 1️⃣5️⃣: Drawdown Under Realistic Sizing")
    with st.spinner("Simulating sized drawdowns..."):
        try:
            t15 = tester.test_sized_drawdown_simulation(strategy_code, param_grid, kelly_fraction=0.25, initial_capital=init_cash)
            show_df(t15)
        except Exception as e:
            st.warning(f"Sized drawdown simulation error: {str(e)[:120]}")

    # Test 16
    st.subheader("Test 1️⃣6️⃣: Parameter Robustness")
    with st.spinner("Assessing parameter robustness..."):
        try:
            t16 = tester.test_parameter_robustness(strategy_code, param_grid)
            show_df(t16)
        except Exception as e:
            st.warning(f"Parameter robustness error: {str(e)[:120]}")

    # Test 17
    st.subheader("Test 1️⃣7️⃣: Market Regime Drift (Beta Drift)")
    with st.spinner("Computing beta drift..."):
        try:
            t17 = tester.test_market_regime_drift(strategy_code, param_grid)
            show_df(t17)
        except Exception as e:
            st.warning(f"Market regime drift error: {str(e)[:120]}")

    # Test 18
    st.subheader("Test 1️⃣8️⃣: Slippage & Commissions Impact")
    with st.spinner("Applying realistic costs..."):
        try:
            t18 = tester.test_slippage_impact(strategy_code, param_grid, slippage_bps=1, commission_bps=0.5)
            show_df(t18)
        except Exception as e:
            st.warning(f"Slippage impact error: {str(e)[:120]}")

    # Test 19
    st.subheader("Test 1️⃣9️⃣: Drawdown Severity & Loss Streaks")
    with st.spinner("Analyzing loss streaks and recovery..."):
        try:
            t19 = tester.test_drawdown_severity(strategy_code, param_grid)
            show_df(t19)
        except Exception as e:
            st.warning(f"Drawdown severity error: {str(e)[:120]}")

    # Test 20
    st.subheader("Test 2️⃣0️⃣: Out-of-Sample Walk-Forward Validation")
    with st.spinner("Running train/test folds..."):
        try:
            t20 = tester.test_walk_forward_validation(strategy_code, param_grid, train_pct=0.7, step_size=0.1)
            if not t20.empty:
                t20['overfitting_ratio'] = t20['overfitting_ratio'].replace([np.inf, -np.inf], np.nan)
                show_df(
                    t20,
                    formatters={
                        'overfitting_ratio': lambda v: 'N/A' if pd.isna(v) else f'{v:.4f}',
                        'delta_sharpe_pct': lambda v: 'N/A' if pd.isna(v) else f'{v:.2%}',
                        'train_ev': '{:.6f}',
                        'test_ev': '{:.6f}',
                        'train_sharpe': '{:.4f}',
                        'test_sharpe': '{:.4f}',
                    }
                )
            else:
                st.info("No walk-forward validation results")
        except Exception as e:
            st.warning(f"Walk-forward validation error: {str(e)[:120]}")

    st.success("✅ All 20 tests completed!")# Add new tests
st.subheader("Test 21️⃣: Multi-Asset Correlation Risk")
with st.spinner("Running multi-asset correlation risk test..."):
    t21 = tester.test_multi_asset_correlation_risk(strategy_code, param_grid)
    show_df(t21)

st.subheader("Test 22️⃣: Correlation Clustering")
with st.spinner("Running correlation clustering test..."):
    t22 = tester.test_correlation_clustering(strategy_code, param_grid)
    show_df(t22)

st.subheader("Test 23️⃣: Factor Attribution")
with st.spinner("Running factor attribution test..."):
    t23 = tester.test_factor_attribution(strategy_code, param_grid)
    show_df(t23)

st.subheader("Test 24️⃣: Liquidity Impact")
with st.spinner("Running liquidity impact test..."):
    t24 = tester.test_liquidity_impact(strategy_code, param_grid)
    show_df(t24)

st.subheader("Test 25️⃣: Dynamic Position Sizing")
with st.spinner("Running dynamic position sizing test..."):
    t25 = tester.test_dynamic_position_sizing(strategy_code, param_grid)
    show_df(t25)

# Add UI/UX improvements
st.subheader("Export Results")
export_button = st.button("Export Results to CSV")
if export_button:
    results_df = pd.concat([t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25])
    results_df.to_csv("results.csv", index=False)
    st.success("Results exported to results.csv")

st.subheader("Compare Multiple Strategies")
compare_button = st.button("Compare Multiple Strategies")
if compare_button:
    # Allow users to input multiple strategies and compare their performance
    strategies = []
    for i in range(5):
        strategy_name = st.text_input(f"Strategy {i+1} Name")
        strategy_code = st.text_area(f"Strategy {i+1} Code")
        strategies.append((strategy_name, strategy_code))
    comparison_results = []
    for strategy_name, strategy_code in strategies:
        results = tester.stress_test(strategy_code, param_grid)
        comparison_results.append((strategy_name, results))
    comparison_df = pd.DataFrame(comparison_results)
    st.dataframe(comparison_df)

st.subheader("Parameter Optimization Heatmaps")
heatmaps_button = st.button("View Parameter Optimization Heatmaps")
if heatmaps_button:
    # Display heatmaps to visualize the impact of different parameter combinations
    param_grid_heatmap = pd.DataFrame(param_grid)
    st.dataframe(param_grid_heatmap)

st.subheader("Live Trading Integration")
live_trading_button = st.button("Enable Live Trading")
if live_trading_button:
    # Integrate the backtester with a live trading platform
    st.info("Live trading integration is not yet implemented")

st.subheader("Risk Dashboard")
risk_dashboard_button = st.button("View Risk Dashboard")
if risk_dashboard_button:
    # Create a risk dashboard to provide users with a comprehensive view of their portfolio's risk profile
    st.info("Risk dashboard is not yet implemented")
