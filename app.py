# app.py
import os
import warnings
warnings.filterwarnings('ignore')

import ast
import re
import hashlib
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from harsh_tester import HarshTester

st.set_page_config(page_title="Harsh Strategy Tester", layout="wide")
os.environ['PYTHONWARNINGS'] = 'ignore'

# ---------------------------
# Title/Header
# ---------------------------
st.title("🔬 Harsh Strategy Tester — Robustness Suite")

# ---------------------------
# Sidebar: Data setup
# ---------------------------
st.sidebar.header("📊 Data Setup")
symbols_input = st.sidebar.text_area(
    "Assets (comma-separated, e.g., SPY,TLT,GLD)",
    value="SPY,TLT,GLD",
    height=80,
)
symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime('2024-12-31'))

@st.cache_data
def load_data(symbols_list, start, end):
    data = yf.download(symbols_list, start=start, end=end)['Close']
    if isinstance(data, pd.Series):
        data = data.to_frame(name=symbols_list[0] if symbols_list else 'Asset')
    return data.dropna()

try:
    data = load_data(symbols, start_date, end_date)
    st.sidebar.success(f"✅ Loaded {len(data)} days for {len(symbols)} assets")
except Exception as e:
    st.sidebar.error(f"❌ Error loading data: {e}")
    st.stop()

# ---------------------------
# Utilities and visualization helpers
# ---------------------------
def show_df(df: pd.DataFrame, formatters: dict | None = None):
    if df is None or df.empty:
        st.info("No results")
        return
    if formatters:
        st.dataframe(df.style.format(formatters), width='stretch')
    else:
        st.dataframe(df, width='stretch')

def plot_equity_curve(stats_dict, title, key_suffix=None):
    eq = stats_dict.get('equity_curve', [])
    dates = stats_dict.get('dates', [])
    if len(eq) > 0 and len(dates) == len(eq):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=eq, mode='lines', name='Equity'))
        fig.update_layout(title=title, height=350)
        st.plotly_chart(fig, width='stretch', key=f"equity_curve_{(key_suffix or title)}")

def best_params_from_t1(t1: pd.DataFrame, param_grid: dict) -> dict | None:
    if t1 is None or t1.empty:
        return None
    row = t1.loc[t1['sharpe'].idxmax()]
    params = {}
    for k in param_grid.keys():
        if k in row.index:
            params[k] = row[k]
    return params

def compute_daily_returns_from_stats(stats: dict) -> np.ndarray:
    eq = np.array(stats.get('equity_curve', []), dtype=float)
    if len(eq) < 2:
        return np.array([])
    eq = np.where(eq <= 0, 1e-6, eq)
    r = np.diff(eq) / eq[:-1]
    r = np.where(np.isfinite(r), r, 0.0)
    return r

def detect_param_columns(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return []
    metric_cols = {
        'sharpe', 'sortino', 'calmar', 'max_dd', 'total_return', 'final_value',
        'window', 'fold_num', 'train_ev', 'test_ev', 'train_sharpe', 'test_sharpe',
        'overfitting_ratio', 'is_overfit', 'period_start', 'crisis', 'days',
        'avg_monthly_return', 'monthly_volatility'
    }
    return [c for c in df.columns if c not in metric_cols]

def plot_param_heatmap_quickview(t1: pd.DataFrame, x_name: str, y_name: str, contour=False, title="Parameter Heatmap (Sharpe, best over others)"):
    if t1 is None or t1.empty or x_name not in t1.columns or y_name not in t1.columns:
        st.info("Run optimization first and ensure the axis parameter names are present.")
        return
    idx = t1.groupby([y_name, x_name])['sharpe'].idxmax()
    best = t1.loc[idx, [x_name, y_name, 'sharpe', 'sortino', 'total_return', 'max_dd']]
    z_sharpe = best.pivot(index=y_name, columns=x_name, values='sharpe')
    sortino_mat = best.pivot(index=y_name, columns=x_name, values='sortino')
    ret_mat = best.pivot(index=y_name, columns=x_name, values='total_return')
    dd_mat = best.pivot(index=y_name, columns=x_name, values='max_dd')

    txt = np.empty(z_sharpe.shape, dtype=object)
    for i, iy in enumerate(z_sharpe.index):
        for j, jx in enumerate(z_sharpe.columns):
            srt = sortino_mat.loc[iy, jx] if (iy in sortino_mat.index and jx in sortino_mat.columns and pd.notna(sortino_mat.loc[iy, jx])) else np.nan
            ret = ret_mat.loc[iy, jx] if (iy in ret_mat.index and jx in ret_mat.columns and pd.notna(ret_mat.loc[iy, jx])) else np.nan
            dd = dd_mat.loc[iy, jx] if (iy in dd_mat.index and jx in dd_mat.columns and pd.notna(dd_mat.loc[iy, jx])) else np.nan
            txt[i, j] = f"So:{srt:.2f}\nR:{ret*100:.1f}%\nDD:{dd:.1%}" if (iy in z_sharpe.index and jx in z_sharpe.columns and pd.notna(z_sharpe.loc[iy, jx])) else ""

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z_sharpe.values,
        x=z_sharpe.columns.astype(str),
        y=z_sharpe.index.astype(str),
        colorscale='Viridis',
        colorbar=dict(title='Sharpe'),
        text=txt,
        hovertemplate=f"{x_name}=%{{x}}<br>{y_name}=%{{y}}<br>Sharpe=%{{z:.2f}}<br>%{{text}}<extra></extra>",
    ))
    if contour and z_sharpe.shape[0] >= 4 and z_sharpe.shape[1] >= 4:
        fig.add_trace(go.Contour(
            z=z_sharpe.values,
            x=z_sharpe.columns.astype(str),
            y=z_sharpe.index.astype(str),
            contours=dict(coloring='none', showlabels=False),
            line=dict(color='white', width=1),
            showscale=False,
            hoverinfo='skip',
        ))
    fig.update_layout(title=title, height=520)
    st.plotly_chart(fig, width='stretch', key=f"param_heatmap_{x_name}_{y_name}_{contour}")

def plot_param_heatmap_slice(t1: pd.DataFrame, x_name: str, y_name: str, slice_name: str, slice_value, contour=False,
                             title="Parameter Heatmap (Sharpe; fixed third param)"):
    if t1 is None or t1.empty:
        st.info("No optimization results to plot.")
        return
    if any(n not in t1.columns for n in [x_name, y_name, slice_name]):
        st.info("Selected parameter names are not present in results.")
        return
    df = t1.copy()
    try:
        df = df[df[slice_name] == slice_value]
    except Exception:
        df[slice_name] = df[slice_name].astype(str)
        df = df[df[slice_name] == str(slice_value)]
    if df.empty:
        st.info("No rows match the selected slice value.")
        return
    plot_param_heatmap_quickview(df, x_name, y_name, contour=contour, title=title)

def plot_cost_survival_curve(t5: pd.DataFrame, t18: pd.DataFrame):
    if t5 is None or t5.empty:
        st.info("Run Transaction Cost Impact to see the cost survival curve.")
        return
    x_bps, y = [], []
    for _, r in t5.iterrows():
        try:
            fee = float(r['fee_level'])
            x_bps.append(fee * 10000.0)
            y.append(r['best_sharpe'])
        except Exception:
            continue
    if not x_bps:
        st.info("No valid cost levels to plot.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_bps, y=y, mode='lines+markers', name='Best Sharpe vs Fee (bps)'))
    if t18 is not None and not t18.empty:
        try:
            s0 = float(np.nanmean(pd.to_numeric(t18['sharpe_no_cost'], errors='coerce').replace([np.inf, -np.inf], np.nan)))
            s1 = float(np.nanmean(pd.to_numeric(t18['sharpe_with_cost'], errors='coerce').replace([np.inf, -np.inf], np.nan)))
            xline = [min(x_bps), max(x_bps)]
            fig.add_trace(go.Scatter(x=xline, y=[s0, s0], mode='lines', name='Sharpe (no cost)', line=dict(dash='dot', color='green')))
            fig.add_trace(go.Scatter(x=xline, y=[s1, s1], mode='lines', name='Sharpe (with slip+comm)', line=dict(dash='dot', color='red')))
        except Exception:
            pass
    fig.update_layout(title="Cost Survival Curve", xaxis_title="Trading Cost (bps)", yaxis_title="Sharpe", height=420)
    st.plotly_chart(fig, width='stretch', key="cost_survival_curve")

def plot_monte_carlo_fan(tester: HarshTester, n_paths=200, p_lo=5, p_hi=95):
    stats = tester.best_stats
    if not stats or 'equity_curve' not in stats:
        st.info("Run Optimization to populate best equity curve for fan chart.")
        return
    eq = np.array(stats['equity_curve'], dtype=float)
    eq = np.where(eq <= 0, 1e-6, eq)
    base_returns = np.diff(eq) / eq[:-1]
    base_returns = np.where(np.isfinite(base_returns), base_returns, 0.0)
    if len(base_returns) < 10:
        st.info("Not enough returns to build a Monte Carlo fan.")
        return
    paths = np.zeros((n_paths, len(eq)))
    for i in range(n_paths):
        rs = np.random.choice(base_returns, size=len(base_returns), replace=True)
        paths[i, 0] = eq[0]
        paths[i, 1:] = eq[0] * np.cumprod(1 + rs)
    lo = np.percentile(paths, p_lo, axis=0)
    med = np.percentile(paths, 50, axis=0)
    hi = np.percentile(paths, p_hi, axis=0)
    dates = stats['dates']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=hi, line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=dates, y=lo, fill='tonexty', name=f'{p_lo}-{p_hi} percentile', line=dict(width=0), hoverinfo='skip', fillcolor='rgba(0,100,200,0.15)'))
    fig.add_trace(go.Scatter(x=dates, y=med, name='Median', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=dates, y=eq, name='Actual', line=dict(color='black')))
    fig.update_layout(title="Monte Carlo Fan Chart (bootstrapped paths)", height=420)
    st.plotly_chart(fig, width='stretch', key="monte_carlo_fan")

def rolling_sharpe_ribbon(stats: dict, window=63):
    if not stats or 'equity_curve' not in stats:
        st.info("Run Optimization to get an equity curve for rolling Sharpe.")
        return
    r = compute_daily_returns_from_stats(stats)
    if len(r) < window + 5:
        st.info("Not enough data for rolling Sharpe window.")
        return
    s = pd.Series(r)
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std()
    roll_sharpe = (mu / (sd.replace(0, np.nan))) * np.sqrt(252)
    band = roll_sharpe.rolling(20).std()
    x = stats['dates'][1:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=(roll_sharpe - band), line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x, y=(roll_sharpe + band), fill='tonexty', name='±1σ', line=dict(width=0), hoverinfo='skip', fillcolor='rgba(200,150,0,0.15)'))
    fig.add_trace(go.Scatter(x=x, y=roll_sharpe, name='Rolling Sharpe', line=dict(color='orange')))
    fig.update_layout(title=f"Rolling Sharpe (window={window})", height=420)
    st.plotly_chart(fig, width='stretch', key=f"rolling_sharpe_{window}")

def drawdown_episodes_scatter(stats: dict):
    if not stats or 'equity_curve' not in stats:
        st.info("Run Optimization to get an equity curve for drawdown episodes.")
        return
    eq = np.array(stats['equity_curve'], dtype=float)
    runmax = np.maximum.accumulate(eq)
    dd = (eq - runmax) / np.where(runmax > 0, runmax, 1.0)
    depths, durations = [], []
    in_dd = False
    start_i = 0
    cur_min = 0.0
    for i, d in enumerate(dd):
        if d < 0 and not in_dd:
            in_dd = True
            start_i = i
            cur_min = d
        elif d < 0 and in_dd:
            cur_min = min(cur_min, d)
        elif d >= 0 and in_dd:
            in_dd = False
            depths.append(cur_min)
            durations.append(i - start_i)
            cur_min = 0.0
    if in_dd:
        depths.append(cur_min)
        durations.append(len(dd) - start_i)
    if not depths:
        st.info("No drawdown episodes detected.")
        return
    fig = go.Figure(data=go.Scatter(x=durations, y=np.array(depths) * 100.0, mode='markers',
                                    marker=dict(size=8, color=np.abs(depths), colorscale='Reds'),
                                    name='Episodes'))
    fig.update_layout(title="Drawdown Pain Map (Depth vs Duration)",
                      xaxis_title="Duration (days)", yaxis_title="Depth (%)", height=420)
    st.plotly_chart(fig, width='stretch', key="drawdown_pain_map")

def train_test_scatter(t20: pd.DataFrame):
    if t20 is None or t20.empty:
        st.info("Run Walk-Forward Overfitting to view train vs test scatter.")
        return
    x = t20['train_sharpe']
    y = t20['test_sharpe']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Folds'))
    xymin = float(min(x.min(), y.min()))
    xymax = float(max(x.max(), y.max()))
    fig.add_trace(go.Scatter(x=[xymin, xymax], y=[xymin, xymax], mode='lines', name='45°', line=dict(dash='dash')))
    fig.update_layout(title="Train vs Test Sharpe (Overfitting Funnel)", xaxis_title="Train Sharpe", yaxis_title="Test Sharpe", height=420)
    st.plotly_chart(fig, width='stretch', key="train_test_scatter")

def regime_quadrant(t17: pd.DataFrame):
    if t17 is None or t17.empty:
        st.info("Run Market Regime Drift to view bull vs bear quadrant.")
        return
    x = t17['bull_market_sharpe']
    y = t17['bear_market_sharpe']
    size = 10 + 40 * (t17['rolling_beta_std'] - t17['rolling_beta_std'].min()) / (t17['rolling_beta_std'].max() - t17['rolling_beta_std'].min() + 1e-9)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(size=size, color=t17['rolling_beta_std'], colorscale='Bluered'),
                             name='Param sets', text=[str(p) for p in t17['param_set']]))
    lim = float(max(abs(x).max(), abs(y).max(), 1.0))
    fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode='lines', name='Parity', line=dict(dash='dash')))
    fig.update_layout(title="Regime Quadrant (Bull vs Bear Sharpe, size=color=beta drift)", xaxis_title="Bull Sharpe", yaxis_title="Bear Sharpe", height=420)
    st.plotly_chart(fig, width='stretch', key="regime_quadrant")

# Asset-level helpers
def plot_asset_vs_benchmark(data: pd.DataFrame, tester_fee: float,
                            strategy_code_local: str, best_params: dict | None,
                            asset_names: list[str], asset_idx: int, initial_capital: float,
                            chart_key: str | None = None):
    if not best_params or len(asset_names) == 0:
        st.info("No best parameters or assets available. Run Optimization first.")
        return
    StrategyClass = HarshTester(data).load_strategy_class(strategy_code_local)
    strat_full = StrategyClass(data, best_params)
    e_full, x_full = strat_full.generate_signals()

    sym = asset_names[asset_idx]
    data_i = data[[sym]]
    e_i = e_full[[sym]] if isinstance(e_full, pd.DataFrame) and sym in e_full.columns else e_full
    x_i = x_full[[sym]] if isinstance(x_full, pd.DataFrame) and sym in x_full.columns else x_full

    tester_single = HarshTester(data_i, init_cash=initial_capital, fee=tester_fee, allocation_method="equal_weight")
    stats_i = tester_single.backtest_signals(e_i, x_i, data=data_i)
    eq_strategy = np.array(stats_i.get('equity_curve', []), dtype=float)
    dates = stats_i.get('dates', data_i.index)
    if len(eq_strategy) == 0 or len(dates) != len(eq_strategy):
        st.info("Strategy equity for this asset is unavailable for plotting.")
        return

    ret_i = data_i.astype(float).pct_change().fillna(0.0).values.flatten()
    eq_bench = initial_capital * np.cumprod(1.0 + ret_i)
    if len(eq_bench) != len(eq_strategy):
        eq_bench = np.concatenate([[initial_capital], eq_bench[1:len(eq_strategy)]])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=eq_strategy, name=f"Strategy ({sym})", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=dates, y=eq_bench, name=f"Benchmark ({sym} buy & hold)", line=dict(color="green")))
    fig.update_layout(title=f"Asset-Level: Strategy vs Benchmark — {sym}", height=420, legend=dict(orientation="h"))
    st.plotly_chart(fig, width='stretch', key=chart_key or f"plot_asset_vs_bench_{sym}_{asset_idx}")

    strat_ret = (eq_strategy[-1] / eq_strategy[0] - 1.0) if len(eq_strategy) > 1 else 0.0
    bench_ret = (eq_bench[-1] / eq_bench[0] - 1.0) if len(eq_bench) > 1 else 0.0
    c1, c2 = st.columns(2)
    c1.metric(f"{sym} Strategy total return (net)", f"{strat_ret*100:.2f}%")
    c2.metric(f"{sym} Benchmark total return", f"{bench_ret*100:.2f}%")

def plot_outperformance_contribution(data: pd.DataFrame, tester_fee: float,
                                     strategy_code_local: str, best_params: dict | None,
                                     asset_names: list[str], initial_capital: float,
                                     chart_key: str | None = None):
    if not best_params or len(asset_names) == 0:
        st.info("No best parameters or assets available. Run Optimization first.")
        return
    StrategyClass = HarshTester(data).load_strategy_class(strategy_code_local)
    strat_full = StrategyClass(data, best_params)
    e_full, x_full = strat_full.generate_signals()

    contributions = []
    for sym in asset_names:
        data_i = data[[sym]]
        e_i = e_full[[sym]] if isinstance(e_full, pd.DataFrame) and sym in e_full.columns else e_full
        x_i = x_full[[sym]] if isinstance(x_full, pd.DataFrame) and sym in x_full.columns else x_full
        tester_single = HarshTester(data_i, init_cash=initial_capital, fee=tester_fee, allocation_method="equal_weight")
        stats_i = tester_single.backtest_signals(e_i, x_i, data=data_i)
        eq_strategy = np.array(stats_i.get('equity_curve', []), dtype=float)
        if len(eq_strategy) == 0:
            contributions.append({"asset": sym, "delta$": 0.0})
            continue
        ret_i = data_i.astype(float).pct_change().fillna(0.0).values.flatten()
        eq_bench = initial_capital * np.cumprod(1.0 + ret_i)
        if len(eq_bench) != len(eq_strategy):
            eq_bench = np.concatenate([[initial_capital], eq_bench[1:len(eq_strategy)]])
        delta_dollars = float(eq_strategy[-1] - eq_bench[-1])
        contributions.append({"asset": sym, "delta$": delta_dollars})

    dfc = pd.DataFrame(contributions)
    if dfc.empty:
        st.info("No contribution data to display.")
        return
    total_abs = float(np.sum(np.abs(dfc["delta$"])))
    dfc["percent_of_outperformance"] = (dfc["delta$"] / total_abs * 100.0) if total_abs > 1e-12 else 0.0
    colors = ["#2ECC71" if d > 0 else "#E74C3C" for d in dfc["delta$"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dfc["asset"],
        y=dfc["delta$"],
        marker_color=colors,
        name="Outperformance ($)",
        hovertemplate="Asset=%{x}<br>Δ$=%{y:.0f}<br>% of outperformance=%{customdata:.1f}%<extra></extra>",
        customdata=dfc["percent_of_outperformance"]
    ))
    fig.update_layout(
        title="Outperformance Contribution by Asset (Δ$ vs benchmark; green = outperformed, red = underperformed)",
        xaxis_title="Asset",
        yaxis_title="Δ$ (Strategy − Buy & Hold)",
        height=420
    )
    st.plotly_chart(fig, width='stretch', key=chart_key or "plot_outperformance_contrib")
    st.caption("Percent of outperformance by asset (signed; underperformers shown as negative).")
    st.dataframe(
        dfc[["asset", "delta$", "percent_of_outperformance"]].rename(columns={"delta$": "delta_dollars"}),
        width='stretch'
    )

# ---------- AI: Strategy code parameter analysis and auto-grid ----------
COMMON_PARAM_HINTS = {
    'fast': 'window', 'slow': 'window', 'period': 'window', 'lookback': 'window',
    'fast_period': 'window', 'slow_period': 'window', 'signal_period': 'window',
    'oversold': 'threshold_low', 'overbought': 'threshold_high',
    'z_entry': 'z_high', 'z_exit': 'z_low',
    'num_std': 'std_mult',
}

def _extract_params_via_ast(code: str) -> dict:
    """Return {name: default_value} from self.params.get('name', default) and [] from self.params['name'] patterns."""
    params = {}
    try:
        tree = ast.parse(code)
        class GetVisitor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call):
                try:
                    if isinstance(node.func, ast.Attribute) and node.func.attr == 'get':
                        if isinstance(node.func.value, ast.Attribute) and node.func.value.attr == 'params':
                            if len(node.args) >= 1 and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                                name = node.args[0].value
                                default = None
                                if len(node.args) >= 2:
                                    if isinstance(node.args[1], ast.Constant):
                                        default = node.args[1].value
                                    elif isinstance(node.args[1], (ast.Num, ast.Str)):
                                        default = getattr(node.args[1], 'n', None) or getattr(node.args[1], 's', None)
                                params[name] = default
                except Exception:
                    pass
                self.generic_visit(node)
            def visit_Subscript(self, node: ast.Subscript):
                try:
                    if isinstance(node.value, ast.Attribute) and node.value.attr == 'params':
                        sl = node.slice.value if hasattr(node.slice, 'value') else node.slice
                        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                            name = sl.value
                            if name not in params:
                                params[name] = None
                except Exception:
                    pass
                self.generic_visit(node)
        GetVisitor().visit(tree)
    except Exception:
        pass
    # regex fallback
    for m in re.finditer(r"self\.params\.get\(\s*['\"]([A-Za-z0-9_]+)['\"]\s*,\s*([0-9\.]+)\s*\)", code):
        name = m.group(1)
        default = float(m.group(2)) if '.' in m.group(2) else int(m.group(2))
        params[name] = default
    for m in re.finditer(r"self\.params\[\s*['\"]([A-Za-z0-9_]+)['\"]\s*\]", code):
        name = m.group(1)
        params.setdefault(name, None)
    return params

def _window_grid(default: int | float | None, data_len: int):
    max_cap = max(10, min(252, data_len // 2))
    base = int(default) if isinstance(default, (int, float)) else 63
    base = max(5, min(base, max_cap))
    candidates = sorted(set([
        max(5, int(round(base * 0.5))),
        max(6, int(round(base * 0.75))),
        base,
        min(max_cap, int(round(base * 1.25))),
        min(max_cap, int(round(base * 1.5))),
    ]))
    return candidates

def _threshold_high_grid(default: float | None):
    base = float(default) if isinstance(default, (int, float)) else 70.0
    values = [max(50, min(90, v)) for v in [60, 65, 70, 75, 80]]
    return sorted(set(values + [max(50, min(90, round(base)))]))

def _threshold_low_grid(default: float | None):
    base = float(default) if isinstance(default, (int, float)) else 30.0
    values = [max(10, min(50, v)) for v in [20, 25, 30, 35, 40]]
    return sorted(set(values + [max(10, min(50, round(base)))]))

def _z_high_grid(default: float | None):
    base = float(default) if isinstance(default, (int, float)) else 1.5
    values = [1.0, 1.5, 2.0, 2.5]
    return sorted(set(values + [round(base, 2)]))

def _z_low_grid(default: float | None):
    base = float(default) if isinstance(default, (int, float)) else 0.75
    values = [0.5, 0.75, 1.0, 1.25]
    return sorted(set(values + [round(base, 2)]))

def _std_mult_grid(default: float | None):
    base = float(default) if isinstance(default, (int, float)) else 2.0
    values = [1.5, 2.0, 2.5, 3.0]
    return sorted(set(values + [round(base, 2)]))

def suggest_param_grid(strategy_code: str, data_len: int) -> dict:
    found = _extract_params_via_ast(strategy_code)
    if not found:
        # Generic fallback
        return {'period': [20, 40, 60, 80, 100], 'threshold': [0.01, 0.02, 0.05]}
    grid = {}
    for name, default in found.items():
        hint = COMMON_PARAM_HINTS.get(name)
        if hint == 'window':
            grid[name] = _window_grid(default, data_len)
        elif hint == 'threshold_high':
            grid[name] = _threshold_high_grid(default)
        elif hint == 'threshold_low':
            grid[name] = _threshold_low_grid(default)
        elif hint == 'z_high':
            grid[name] = _z_high_grid(default)
        elif hint == 'z_low':
            grid[name] = _z_low_grid(default)
        elif hint == 'std_mult':
            grid[name] = _std_mult_grid(default)
        else:
            lname = name.lower()
            if any(k in lname for k in ['period', 'lookback', 'fast', 'slow', 'span']):
                grid[name] = _window_grid(default, data_len)
            else:
                base = float(default) if isinstance(default, (int, float)) else 1.0
                grid[name] = sorted(set([round(base * f, 3) for f in [0.5, 0.75, 1.0, 1.25, 1.5]]))
    # constraints
    if 'fast' in grid and 'slow' in grid:
        grid['fast'] = [v for v in grid['fast'] if v < max(grid['slow'])]
    if 'fast_period' in grid and 'slow_period' in grid:
        grid['fast_period'] = [v for v in grid['fast_period'] if v < max(grid['slow_period'])]
    if 'oversold' in grid and 'overbought' in grid:
        grid['overbought'] = [v for v in grid['overbought'] if v > min(grid['oversold'])]
    if 'z_exit' in grid and 'z_entry' in grid:
        grid['z_exit'] = [v for v in grid['z_exit'] if v < max(grid['z_entry'])]
    # Limit to 3 params for initial suggestion
    if len(grid) > 3:
        order = [n for n in ['fast', 'slow', 'period', 'lookback', 'overbought', 'oversold', 'z_entry', 'z_exit', 'num_std', 'fast_period', 'slow_period', 'signal_period'] if n in grid]
        keep = order[:3] if order else list(grid.keys())[:3]
        grid = {k: grid[k] for k in keep}
    return grid

def plot_cost_factored_vs_benchmark(data: pd.DataFrame, tester: HarshTester,
                                    strategy_code_local: str, best_params: dict | None,
                                    initial_capital: float, title: str = "Strategy (net of costs) vs Benchmark"):
    if not best_params:
        st.info("No best parameters available. Run Optimization first.")
        return
    StrategyClass = tester.load_strategy_class(strategy_code_local)
    strat = StrategyClass(data, best_params)
    e, x = strat.generate_signals()
    stats = tester.backtest_signals(e, x, data=data)
    eq_strategy = np.array(stats.get('equity_curve', []), dtype=float)
    dates = stats.get('dates', data.index)
    if len(eq_strategy) == 0 or len(dates) != len(eq_strategy):
        st.info("Strategy equity is unavailable for plotting.")
        return

    # Benchmark: equal-weight buy-and-hold of the loaded assets
    px = data.astype(float).copy()
    if px.shape[1] > 1:
        bench_ret = px.pct_change().mean(axis=1).fillna(0.0).values
    else:
        bench_ret = px.pct_change().fillna(0.0).values.flatten()
    eq_bench = initial_capital * np.cumprod(1.0 + bench_ret)
    if len(eq_bench) != len(eq_strategy):
        eq_bench = np.concatenate([[initial_capital], eq_bench[1:len(eq_strategy)]])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=eq_strategy, name="Strategy (net of costs)", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=dates, y=eq_bench, name="Benchmark (equal-weight buy & hold)", line=dict(color="green")))
    fig.update_layout(title=title, height=420, legend=dict(orientation="h"))
    st.plotly_chart(fig, width='stretch', key=f"cost_vs_bench_{hash(title)}")

    strat_ret = (eq_strategy[-1] / eq_strategy[0] - 1.0) if len(eq_strategy) > 1 else 0.0
    bench_ret = (eq_bench[-1] / eq_bench[0] - 1.0) if len(eq_bench) > 1 else 0.0
    c1, c2 = st.columns(2)
    c1.metric("Strategy total return (net)", f"{strat_ret*100:.2f}%")
    c2.metric("Benchmark total return", f"{bench_ret*100:.2f}%")

# ---------------------------
# Strategy Selection (Single or Compare)
# ---------------------------
st.header("Strategy Selection")
mode = st.radio("Mode", ["Single Strategy", "Compare Two Strategies"], horizontal=True)

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
        exits = (z_score > -z_exit).shift(1).fillna(False).astype(bool)
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
        ema_fast = self.data.ewm(span=fast_period, adjust=False).mean()
        ema_slow = self.data.ewm(span=slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        entries = (macd > signal).shift(1).fillna(False).astype(bool)
        exits = (macd < signal).shift(1).fillna(False).astype(bool)
        return entries, exits
""",
}

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

if mode == "Single Strategy":
    strategy_options = list(PREDEFINED_STRATEGIES.keys()) + ["Custom"]
    choice = st.selectbox("Select or paste a strategy", options=strategy_options, index=0, key="single_choice")
    strategy_code = PREDEFINED_STRATEGIES[choice] if choice != "Custom" else st.text_area(
        "Paste your custom Strategy code (must define class Strategy with generate_signals)",
        value=DEFAULT_CUSTOM,
        height=280,
        key="single_code",
    )
    with st.expander(f"View {choice}"):
        st.code(strategy_code, language='python')
else:
    left_col, right_col = st.columns(2)
    strategy_options = list(PREDEFINED_STRATEGIES.keys()) + ["Custom"]
    with left_col:
        left_choice = st.selectbox("Left Strategy", options=strategy_options, index=0, key="left_choice")
        strategy_code_left = PREDEFINED_STRATEGIES[left_choice] if left_choice != "Custom" else st.text_area(
            "Left Custom Strategy code", value=DEFAULT_CUSTOM, height=260, key="left_code"
        )
        with st.expander(f"View Left: {left_choice}"):
            st.code(strategy_code_left, language='python')
    with right_col:
        right_choice = st.selectbox("Right Strategy", options=strategy_options, index=1 if len(strategy_options) > 1 else 0, key="right_choice")
        strategy_code_right = PREDEFINED_STRATEGIES[right_choice] if right_choice != "Custom" else st.text_area(
            "Right Custom Strategy code", value=DEFAULT_CUSTOM, height=260, key="right_code"
        )
        with st.expander(f"View Right: {right_choice}"):
            st.code(strategy_code_right, language='python')

# ---------------------------
# Asset navigation callbacks
# ---------------------------
def _asset_nav_prev(n_assets):
    def _cb():
        st.session_state['asset_idx'] = (st.session_state.get('asset_idx', 0) - 1) % max(n_assets, 1)
        st.session_state['asset_nav_clicked'] = True
    return _cb

def _asset_nav_next(n_assets):
    def _cb():
        st.session_state['asset_idx'] = (st.session_state.get('asset_idx', 0) + 1) % max(n_assets, 1)
        st.session_state['asset_nav_clicked'] = True
    return _cb

# ---------------------------
# Parameter Grid (flexible; auto-populate; no contours here)
# ---------------------------
def param_form(side_prefix: str, title: str, strategy_code_local: str):
    st.subheader(title)

    code_hash_key = f'{side_prefix}_code_hash'
    grid_df_key = f'{side_prefix}_grid_df'

    new_hash = hashlib.sha256(strategy_code_local.encode('utf-8')).hexdigest()

    # Auto-build suggested grid when code changes or no table exists yet
    if st.session_state.get(code_hash_key) != new_hash or grid_df_key not in st.session_state:
        st.session_state[code_hash_key] = new_hash
        try:
            sg = suggest_param_grid(strategy_code_local, len(data))
        except Exception:
            sg = {}
        rows = [{"name": str(n), "values": ",".join(str(v) for v in vals)} for n, vals in (sg or {}).items()]
        if not rows:
            rows = [{"name": "", "values": ""}]
        st.session_state[grid_df_key] = pd.DataFrame(rows, columns=["name", "values"])

    st.caption("Edit any row, add new parameters, or delete rows. Values are comma‑separated numbers.")
    edited_df = st.data_editor(
        st.session_state[grid_df_key],
        num_rows="dynamic",
        width='stretch',
        column_config={
            "name": st.column_config.TextColumn("Parameter name", required=True),
            "values": st.column_config.TextColumn("Values (comma‑separated)", required=True),
        },
        hide_index=True,
        key=f"{side_prefix}_editor"
    )
    st.session_state[grid_df_key] = edited_df

    cA, _ = st.columns([1, 6])
    with cA:
        reset_btn = st.button("Reset to Suggested", key=f"{side_prefix}_reset")

    if reset_btn:
        try:
            sg = suggest_param_grid(strategy_code_local, len(data))
        except Exception:
            sg = {}
        rows = [{"name": str(n), "values": ",".join(str(v) for v in vals)} for n, vals in (sg or {}).items()]
        if not rows:
            rows = [{"name": "", "values": ""}]
        st.session_state[grid_df_key] = pd.DataFrame(rows, columns=["name", "values"])
        st.rerun()

    # Parse the table into the grid dict
    def _parse_vals_list(val_str: str):
        out = []
        for x in str(val_str).split(','):
            x = x.strip()
            if not x:
                continue
            try:
                f = float(x)
                out.append(int(f) if abs(f - int(f)) < 1e-12 else f)
            except Exception:
                pass
        return out

    grid = {}
    invalid_rows = []
    for idx, row in edited_df.iterrows():
        name = str(row.get("name", "")).strip()
        vals = _parse_vals_list(row.get("values", ""))
        if name and len(vals) > 0:
            grid[name] = vals
        elif name or str(row.get("values", "")).strip():
            invalid_rows.append(idx + 1)

    if invalid_rows:
        st.warning(f"Some rows are incomplete or non‑numeric (rows {', '.join(map(str, invalid_rows))}). Only valid rows were included in the grid.")

    return grid

# ---------------------------
# Portfolio Setup
# ---------------------------
st.header("Portfolio Setup")
c1, c2, c3 = st.columns(3)
with c1:
    init_cash = st.number_input("Initial Capital ($)", value=100000, min_value=1000)
with c2:
    allocation_method = st.selectbox("Allocation Method", ["equal_weight", "custom"])
with c3:
    fee_bps = st.slider("Transaction Fee (bps)", 0, 50, 3)
fee = fee_bps / 10000.0

custom_weights = None
weights_valid = True
if allocation_method == "custom":
    st.markdown("Enter custom allocation weights (percent) for each asset. Must sum to 100.")
    cols_w = st.columns(min(4, max(1, len(symbols))))
    custom_pct = {}
    for idx, sym in enumerate(symbols):
        with cols_w[idx % len(cols_w)]:
            key = f"wt_{sym}"
            default_val = 100.0 / max(1, len(symbols))
            custom_pct[sym] = st.number_input(f"{sym} (%)", min_value=0.0, max_value=100.0, value=default_val, step=0.5, key=key)
    total_pct = float(sum(custom_pct.values()))
    st.write(f"Total: {total_pct:.2f}%")
    if abs(total_pct - 100.0) > 1e-6:
        st.error("Custom weights must sum to 100%.")
        weights_valid = False
    else:
        custom_weights = {k: (v / 100.0) for k, v in custom_pct.items()}

# ---------------------------
# Render parameter grid forms
# ---------------------------
if mode == "Single Strategy":
    single_grid = param_form("single", "Parameter Grid (Single)", strategy_code)
else:
    left_grid = param_form("left", "Left Parameter Grid", strategy_code_left)
    right_grid = param_form("right", "Right Parameter Grid", strategy_code_right)

# ---------------------------
# Run controls (no tabs above; dashboard will render below)
# ---------------------------
st.header("Run")
run_btn_col, dbg_col = st.columns([3, 1])
with run_btn_col:
    run_tests = st.button("🚀 Run Robustness Suite")
with dbg_col:
    debug_mode = st.checkbox("🐛 Debug Output", key="debug_mode")

def warn_if_no_variation(df: pd.DataFrame, label: str):
    try:
        if df is not None and not df.empty:
            s = df['sharpe'].round(6)
            if s.nunique() <= 1:
                st.warning(f"[{label}] No variation detected across grid results. Check parameter names match the strategy's params.")
    except Exception:
        pass

# ---------------------------
# Compute and cache results per label (only when Run clicked)
# ---------------------------
def run_and_cache_suite(label, strategy_code_local: str, grid_to_use: dict):
    tester = HarshTester(
        data,
        init_cash=init_cash,
        fee=fee,
        allocation_method=allocation_method,
        custom_weights=custom_weights
    )
    tester.debug = bool(debug_mode)

    # Core tests
    t1 = tester.stress_test(strategy_code_local, grid_to_use)
    warn_if_no_variation(t1, label)
    t18 = tester.test_slippage_impact(strategy_code_local, grid_to_use, slippage_bps=1, commission_bps=0.5)
    t19 = tester.test_drawdown_severity(strategy_code_local, grid_to_use)
    t20 = tester.test_walk_forward_validation(strategy_code_local, grid_to_use, train_pct=0.7, step_size=0.1)

    # Edge & Stability
    t_edge = tester.test_edge_expected_value(strategy_code_local, grid_to_use)
    t_oos_stab = tester.test_oos_distribution_stability(strategy_code_local, grid_to_use, train_pct=0.7, step_size=0.1)

    # Additional tests (cached for UI re-render)
    def safe(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            return pd.DataFrame()

    t5 = safe(tester.transaction_cost_test, strategy_code_local, grid_to_use)
    t2 = safe(tester.walk_forward_test, strategy_code_local, grid_to_use, 2, 1)
    t3 = safe(tester.monte_carlo_test, strategy_code_local, grid_to_use, 500)
    t4 = safe(tester.parameter_sensitivity_test, strategy_code_local, grid_to_use, 0.2)
    t16 = safe(tester.test_parameter_robustness, strategy_code_local, grid_to_use)
    t6 = safe(tester.rolling_metrics_test, strategy_code_local, grid_to_use, 1)
    t7 = safe(tester.crisis_stress_test, strategy_code_local, grid_to_use)
    t13 = safe(tester.test_regime_dependency, strategy_code_local, grid_to_use, 'volatility')
    t17 = safe(tester.test_market_regime_drift, strategy_code_local, grid_to_use)
    t11 = safe(tester.test_conditional_decomposition, strategy_code_local, grid_to_use)
    t14 = safe(tester.test_win_loss_distribution_stability, strategy_code_local, grid_to_use, 2)
    t10 = safe(tester.kelly_criterion_test, strategy_code_local, grid_to_use)
    t15 = safe(tester.test_sized_drawdown_simulation, strategy_code_local, grid_to_use, 0.25, init_cash)
    t21 = safe(tester.test_multi_asset_correlation_risk, strategy_code_local, grid_to_use)
    t22 = safe(tester.test_correlation_clustering, strategy_code_local, grid_to_use)
    t23 = safe(tester.test_factor_attribution, strategy_code_local, grid_to_use)
    t24 = safe(tester.test_liquidity_impact, strategy_code_local, grid_to_use)
    t25 = safe(tester.test_dynamic_position_sizing, strategy_code_local, grid_to_use)

    # Cache everything needed for lightweight re-render
    st.session_state[f"{label}_t1"] = t1
    st.session_state[f"{label}_t18"] = t18
    st.session_state[f"{label}_t19"] = t19
    st.session_state[f"{label}_t20"] = t20
    st.session_state[f"{label}_t_edge"] = t_edge
    st.session_state[f"{label}_t_oos"] = t_oos_stab
    st.session_state[f"{label}_t5"] = t5
    st.session_state[f"{label}_t2"] = t2
    st.session_state[f"{label}_t3"] = t3
    st.session_state[f"{label}_t4"] = t4
    st.session_state[f"{label}_t16"] = t16
    st.session_state[f"{label}_t6"] = t6
    st.session_state[f"{label}_t7"] = t7
    st.session_state[f"{label}_t13"] = t13
    st.session_state[f"{label}_t17"] = t17
    st.session_state[f"{label}_t11"] = t11
    st.session_state[f"{label}_t14"] = t14
    st.session_state[f"{label}_t10"] = t10
    st.session_state[f"{label}_t15"] = t15
    st.session_state[f"{label}_t21"] = t21
    st.session_state[f"{label}_t22"] = t22
    st.session_state[f"{label}_t23"] = t23
    st.session_state[f"{label}_t24"] = t24
    st.session_state[f"{label}_t25"] = t25
    st.session_state[f"{label}_best_stats"] = tester.best_stats
    st.session_state[f"{label}_strategy_code"] = strategy_code_local
    st.session_state[f"{label}_grid_used"] = grid_to_use

    st.session_state['suite_has_run'] = True

# ---------------------------
# Render suite from cache into tabs (no recompute on UI toggles)
# ---------------------------
def render_suite_from_cache(label: str, tabs):
    t1   = st.session_state.get(f"{label}_t1", pd.DataFrame())
    t18  = st.session_state.get(f"{label}_t18", pd.DataFrame())
    t19  = st.session_state.get(f"{label}_t19", pd.DataFrame())
    t20  = st.session_state.get(f"{label}_t20", pd.DataFrame())
    t_edge = st.session_state.get(f"{label}_t_edge", pd.DataFrame())
    t_oos  = st.session_state.get(f"{label}_t_oos", pd.DataFrame())
    t5   = st.session_state.get(f"{label}_t5", pd.DataFrame())
    t2   = st.session_state.get(f"{label}_t2", pd.DataFrame())
    t3   = st.session_state.get(f"{label}_t3", pd.DataFrame())
    t4   = st.session_state.get(f"{label}_t4", pd.DataFrame())
    t16  = st.session_state.get(f"{label}_t16", pd.DataFrame())
    t6   = st.session_state.get(f"{label}_t6", pd.DataFrame())
    t7   = st.session_state.get(f"{label}_t7", pd.DataFrame())
    t13  = st.session_state.get(f"{label}_t13", pd.DataFrame())
    t17  = st.session_state.get(f"{label}_t17", pd.DataFrame())
    t11  = st.session_state.get(f"{label}_t11", pd.DataFrame())
    t14  = st.session_state.get(f"{label}_t14", pd.DataFrame())
    t10  = st.session_state.get(f"{label}_t10", pd.DataFrame())
    t15  = st.session_state.get(f"{label}_t15", pd.DataFrame())
    t21  = st.session_state.get(f"{label}_t21", pd.DataFrame())
    t22  = st.session_state.get(f"{label}_t22", pd.DataFrame())
    t23  = st.session_state.get(f"{label}_t23", pd.DataFrame())
    t24  = st.session_state.get(f"{label}_t24", pd.DataFrame())
    t25  = st.session_state.get(f"{label}_t25", pd.DataFrame())
    best_stats = st.session_state.get(f"{label}_best_stats", None)
    strategy_code_local = st.session_state.get(f"{label}_strategy_code")
    grid_used = st.session_state.get(f"{label}_grid_used", {})

    # Edge & Stability
    with tabs[0]:
        st.markdown(f"### Edge & Stability — {label}")

        st.markdown("#### 1) Edge (Expected Value)")
        if t_edge is not None and not t_edge.empty:
            best_ev_row = t_edge.loc[t_edge['ev_per_trade'].idxmax()]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("EV per trade", f"{best_ev_row['ev_per_trade']*100:.2f}%")
            c2.metric("Win rate", f"{best_ev_row['win_rate']*100:.1f}%")
            c3.metric("Avg winner", f"{best_ev_row['avg_win']*100:.2f}%")
            c4.metric("Avg loser", f"{best_ev_row['avg_loss']*100:.2f}%")
            c5.metric("# Trades", f"{int(best_ev_row['num_trades'])}")
            show_df(t_edge, formatters={
                'win_rate': '{:.2%}',
                'avg_win': '{:.4%}', 'avg_loss': '{:.4%}',
                'ev_per_trade': '{:.4%}',
                'ev_from_law': '{:.4%}'
            })
        else:
            st.info("No trades found to compute expected value.")

        st.markdown("#### 2) P&L Distribution Stability (OOS)")
        if t_oos is not None and not t_oos.empty:
            ks_mean = float(pd.to_numeric(t_oos['ks_divergence'], errors='coerce').mean())
            folds = len(t_oos)
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean KS divergence", f"{ks_mean:.3f}")
            c2.metric("# Folds", f"{folds}")
            c3.metric("Mean EV (test)", f"{t_oos['test_ev'].mean()*100:.2f}%")
            show_df(t_oos, formatters={
                'train_ev': '{:.4%}', 'test_ev': '{:.4%}',
                'train_win_rate': '{:.1%}', 'test_win_rate': '{:.1%}',
                'ks_divergence': '{:.4f}'
            })
        else:
            st.info("No folds produced stable train/test distributions (insufficient trades or data).")

        st.markdown("#### 3) Cost‑Factored P&L vs Benchmark")
        if strategy_code_local is not None and t1 is not None and not t1.empty:
            bp = best_params_from_t1(t1, grid_used)
            if bp:
                tester_tmp = HarshTester(data, init_cash=init_cash, fee=fee, allocation_method=allocation_method, custom_weights=custom_weights)
                plot_cost_factored_vs_benchmark(data, tester_tmp, strategy_code_local, bp, initial_capital=init_cash,
                                                title=f"{label}: Net‑of‑costs strategy vs equal‑weight benchmark")

        st.markdown("#### 4) Asset‑Level: Strategy vs Asset Benchmark")
        if "asset_idx" not in st.session_state:
            st.session_state["asset_idx"] = 0
        n_assets = len(symbols)
        nav_cols = st.columns([1, 1, 6])
        with nav_cols[0]:
            st.button("← Prev", on_click=_asset_nav_prev(n_assets), key=f"prev_{label}")
        with nav_cols[1]:
            st.button("Next →", on_click=_asset_nav_next(n_assets), key=f"next_{label}")
        with nav_cols[2]:
            st.write(f"Viewing asset {st.session_state['asset_idx'] + 1} of {n_assets}")

        if strategy_code_local is not None and t1 is not None and not t1.empty:
            bp = best_params_from_t1(t1, grid_used)
            if bp:
                plot_asset_vs_benchmark(
                    data,
                    tester_fee=fee,
                    strategy_code_local=strategy_code_local,
                    best_params=bp,
                    asset_names=symbols,
                    asset_idx=st.session_state["asset_idx"],
                    initial_capital=init_cash,
                    chart_key=f"edge_asset_{label}_{st.session_state['asset_idx']}"
                )
                st.markdown("#### 5) Outperformance Contribution by Asset")
                plot_outperformance_contribution(
                    data,
                    tester_fee=fee,
                    strategy_code_local=strategy_code_local,
                    best_params=bp,
                    asset_names=symbols,
                    initial_capital=init_cash,
                    chart_key=f"edge_contrib_{label}"
                )

    # Risk Dashboard
    with tabs[1]:
        st.markdown(f"### Risk Dashboard — {label}")
        if t1 is None or t1.empty:
            st.info("Run the suite to populate the dashboard.")
        else:
            best_row = t1.loc[t1['sharpe'].idxmax()]
            if t18 is not None and not t18.empty:
                vals = pd.to_numeric(t18['cost_degradation_pct'], errors='coerce').replace([np.inf, -np.inf], np.nan)
                cost_deg = float(np.nanmean(vals))
                survives_pct = 100.0 * float(np.mean(t18['edge_survives_costs'].astype(float)))
            else:
                cost_deg, survives_pct = np.nan, np.nan
            if t19 is not None and not t19.empty:
                loss_streak = int(t19['max_consecutive_losses'].max())
                worst_intra_dd = float(t19['max_intra_dd'].min())
                rec_days = int(t19['recovery_time_days'].max())
            else:
                loss_streak, worst_intra_dd, rec_days = 0, 0.0, 0
            if t20 is not None and not t20.empty:
                safe_ratios = t20['overfitting_ratio'].replace([np.inf, -np.inf], np.nan)
                overfit_ratio = float(np.nanmean(safe_ratios))
                overfit_flag_pct = 100.0 * float(np.mean(t20['is_overfit'].astype(float)))
            else:
                overfit_ratio, overfit_flag_pct = np.nan, np.nan

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Best Sharpe", f"{best_row['sharpe']:.2f}")
            c2.metric("Best Sortino", f"{best_row['sortino']:.2f}")
            c3.metric("Total Return", f"{best_row['total_return']*100:.2f}%")
            c4.metric("Max Drawdown", f"{best_row['max_dd']:.2%}")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Cost Degradation (avg, %)", "N/A" if pd.isna(cost_deg) else f"{cost_deg:.1f}%")
            c6.metric("Edge Survives Costs", "N/A" if pd.isna(survives_pct) else f"{survives_pct:.0f}%")
            c7.metric("Max Loss Streak", f"{loss_streak}")
            c8.metric("Overfitting Ratio", "N/A" if pd.isna(overfit_ratio) else f"{overfit_ratio:.2f}")

            c9, c10 = st.columns(2)
            c9.metric("Worst Intra-Period DD", f"{worst_intra_dd:.2%}")
            c10.metric("% Folds Overfit", "N/A" if pd.isna(overfit_flag_pct) else f"{overfit_flag_pct:.0f}%")

            if best_stats and best_stats.get('equity_curve') and best_stats.get('dates') is not None:
                try:
                    eq = np.array(best_stats['equity_curve'], dtype=float)
                    dd_dates = best_stats['dates']
                    start_idx = max(0, len(eq) - 252)
                    eq_recent = eq[start_idx:]
                    running_max = np.maximum.accumulate(eq_recent)
                    dd_recent = (eq_recent - running_max) / np.where(running_max > 0, running_max, 1.0)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=dd_dates[start_idx:], y=dd_recent * 100.0,
                                             fill='tozeroy', name='Drawdown (%)'))
                    fig.update_layout(title="Drawdown (last ~1 year)", height=300)
                    st.plotly_chart(fig, width='stretch', key=f"risk_dd_recent_{label}")
                except Exception:
                    pass

    # Performance & Optimization (metrics & equity; live heatmap toggle below)
    with tabs[2]:
        st.markdown(f"### Performance & Optimization — {label}")
        if t1 is not None and not t1.empty:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Best Sharpe", f"{t1['sharpe'].max():.2f}")
            m2.metric("Best Sortino", f"{t1['sortino'].max():.2f}")
            m3.metric("Avg Sharpe", f"{t1['sharpe'].mean():.2f}")
            m4.metric("Best Return", f"{t1['total_return'].max()*100:.2f}%")
            m5.metric("Min Max DD", f"{t1['max_dd'].min():.2%}")
            show_df(t1.sort_values('sharpe', ascending=False))
        if best_stats is not None:
            plot_equity_curve(best_stats, "Equity Curve (Best)", key_suffix=label)

        st.markdown("#### Parameter Heatmap")
        param_cols = detect_param_columns(t1) if t1 is not None and not t1.empty else []
        if len(param_cols) >= 2:
            x_name = st.session_state.get(f"{label}_x_axis", param_cols[0])
            y_name = st.session_state.get(f"{label}_y_axis", param_cols[1 if len(param_cols) > 1 else 0])

            ax_cols = st.columns(2)
            with ax_cols[0]:
                x_name = st.selectbox(f"[{label}] X-axis param", options=param_cols,
                                      index=param_cols.index(x_name) if x_name in param_cols else 0,
                                      key=f"{label}_x_axis")
            with ax_cols[1]:
                y_name = st.selectbox(f"[{label}] Y-axis param", options=param_cols,
                                      index=param_cols.index(y_name) if y_name in param_cols else (1 if len(param_cols) > 1 else 0),
                                      key=f"{label}_y_axis")

            contour_key = f"{label}_contours_live"
            contour_flag = bool(st.session_state.get(contour_key, False))

            third_params = [p for p in param_cols if p not in [x_name, y_name]]
            if len(third_params) >= 1 and third_params[0] in t1.columns:
                slice_name = third_params[0]
                unique_vals = sorted(list(pd.Series(t1[slice_name]).dropna().unique()))
                if unique_vals:
                    s_cols = st.columns([2, 1])
                    with s_cols[0]:
                        sel_val = st.selectbox(f"[{label}] Fix '{slice_name}' at:", options=unique_vals, key=f"{label}_slice_val")
                    plot_param_heatmap_slice(t1, x_name, y_name, slice_name, sel_val, contour=contour_flag,
                                             title=f"{label}: {x_name} vs {y_name} (fixed {slice_name}={sel_val})")
                else:
                    plot_param_heatmap_quickview(t1, x_name, y_name, contour=contour_flag,
                                                 title=f"{label}: {x_name} vs {y_name}")
            else:
                plot_param_heatmap_quickview(t1, x_name, y_name, contour=contour_flag,
                                             title=f"{label}: {x_name} vs {y_name}")

            # Toggle directly below figure (re-render only; no recompute)
            st.checkbox(f"[{label}] Show contours", value=contour_flag, key=contour_key)
        else:
            st.info(f"[{label}] Need at least two parameter columns to draw a heatmap.")

    # Costs & Execution
    with tabs[3]:
        st.markdown(f"### Costs & Execution — {label}")
        st.markdown("#### Transaction Costs Impact")
        show_df(t5)
        st.markdown("#### Slippage & Commission Impact")
        show_df(t18)
        st.markdown("#### Cost Survival Curve")
        plot_cost_survival_curve(t5, t18)

    # Stability & Overfitting
    with tabs[4]:
        st.markdown(f"### Stability & Overfitting — {label}")
        st.markdown("#### Walk-Forward (2y train / 1y test)")
        show_df(t2)
        st.markdown("#### Walk-Forward Overfitting (sliding windows)")
        if t20 is not None and not t20.empty:
            t20c = t20.copy()
            t20c['overfitting_ratio'] = t20c['overfitting_ratio'].replace([np.inf, -np.inf], np.nan)
            show_df(
                t20c,
                formatters={
                    'overfitting_ratio': lambda v: 'N/A' if pd.isna(v) else f'{v:.4f}',
                    'train_ev': '{:.6f}',
                    'test_ev': '{:.6f}',
                    'train_sharpe': '{:.4f}',
                    'test_sharpe': '{:.4f}',
                }
            )
            st.markdown("##### Train vs Test Sharpe (Overfitting Funnel)")
            train_test_scatter(t20c)
        else:
            st.info("No walk-forward validation results")
        st.markdown("#### Monte Carlo (500 resamples)")
        show_df(t3)
        st.markdown("#### Monte Carlo Fan Chart (Bootstrapped Paths)")
        if best_stats is not None:
            tester_tmp = HarshTester(data, init_cash=init_cash, fee=fee, allocation_method=allocation_method, custom_weights=custom_weights)
            tester_tmp.best_stats = best_stats
            plot_monte_carlo_fan(tester_tmp, n_paths=200)

    # Parameters & Robustness
    with tabs[5]:
        st.markdown(f"### Parameters & Robustness — {label}")
        st.markdown("#### Parameter Sensitivity (±20%)")
        show_df(t4)
        st.markdown("#### Parameter Robustness")
        show_df(t16)

    # Time & Regime
    with tabs[6]:
        st.markdown(f"### Time & Regime — {label}")
        st.markdown("#### Rolling Metrics (1-year windows)")
        show_df(t6)
        st.markdown("#### Crisis Stress Tests")
        show_df(t7)
        st.markdown("#### Regime Dependency")
        show_df(t13)
        st.markdown("#### Market Regime Drift")
        show_df(t17)
        st.markdown("##### Regime Quadrant (Bull vs Bear)")
        regime_quadrant(t17)
        st.markdown("#### Rolling Sharpe Ribbon")
        if best_stats is not None:
            rolling_sharpe_ribbon(best_stats)

    # Trade Anatomy
    with tabs[7]:
        st.markdown(f"### Trade Anatomy — {label}")
        st.markdown("#### Conditional P/L Decomposition")
        show_df(t11)
        st.markdown("#### Win/Loss Distribution Stability")
        show_df(
            t14,
            formatters={
                'winner_kde_divergence': lambda v: 'N/A' if pd.isna(v) else f'{v:.4f}',
                'loser_kde_divergence': lambda v: 'N/A' if pd.isna(v) else f'{v:.4f}',
                'combined_distribution_score': lambda v: 'N/A' if pd.isna(v) else f'{v:.4f}',
            }
        )
        st.markdown("#### Loss Streaks & Recovery")
        show_df(t19)
        st.markdown("#### Kelly Criterion & Sized Drawdowns")
        show_df(t10)
        show_df(t15)

    # Extended
    with tabs[8]:
        st.markdown(f"### Extended Diagnostics — {label}")
        st.markdown("Multi-Asset Correlation Risk")
        show_df(t21)
        st.markdown("Correlation Clustering")
        show_df(t22)
        st.markdown("Factor Attribution")
        show_df(t23)
        st.markdown("Liquidity Impact")
        show_df(t24)
        st.markdown("Dynamic Position Sizing")
        show_df(t25)

# ---------------------------
# Bottom-anchored results area and run gating
# ---------------------------
results_area = st.container()
auto_nav = bool(st.session_state.pop('asset_nav_clicked', False))

if run_tests:
    if allocation_method == "custom" and not weights_valid:
        st.error("Please fix custom weights so they sum to 100% before running.")
    else:
        # Compute and cache results
        if mode == "Single Strategy":
            run_and_cache_suite("Single", strategy_code, single_grid)
        else:
            run_and_cache_suite("Left", strategy_code_left, left_grid)
            run_and_cache_suite("Right", strategy_code_right, right_grid)

# Render tabs from cache whenever a run has happened (or after nav), without recompute
if st.session_state.get('suite_has_run', False):
    with results_area:
        tabs = st.tabs([
            "Edge & Stability",
            "Risk Dashboard",
            "Performance & Optimization",
            "Costs & Execution",
            "Stability & Overfitting",
            "Parameters & Robustness",
            "Time & Regime",
            "Trade Anatomy",
            "Extended"
        ])
        if mode == "Single Strategy":
            render_suite_from_cache("Single", tabs)
        else:
            render_suite_from_cache("Left", tabs)
            render_suite_from_cache("Right", tabs)
            # Head-to-Head summary below
            cmp_tab = st.tabs(["Head-to-Head Summary"])
            with cmp_tab[0]:
                left_t1 = st.session_state.get("Left_t1", pd.DataFrame())
                right_t1 = st.session_state.get("Right_t1", pd.DataFrame())
                st.subheader("Head-to-Head (Best from Optimization)")
                try:
                    if not left_t1.empty and not right_t1.empty:
                        left_best = left_t1.loc[left_t1['sharpe'].idxmax()]
                        right_best = right_t1.loc[right_t1['sharpe'].idxmax()]
                        cA, cB, cC, cD = st.columns(4)
                        cA.metric("Sharpe: Left", f"{left_best['sharpe']:.2f}")
                        cB.metric("Sharpe: Right", f"{right_best['sharpe']:.2f}")
                        cC.metric("Max DD: Left", f"{left_best['max_dd']:.2%}")
                        cD.metric("Max DD: Right", f"{right_best['max_dd']:.2%}")
                        cE, cF = st.columns(2)
                        cE.metric("Return: Left", f"{left_best['total_return']*100:.2f}%")
                        cF.metric("Return: Right", f"{right_best['total_return']*100:.2f}%")
                        comp_df = pd.DataFrame([
                            {"side": "Left", "sharpe": left_best["sharpe"], "sortino": left_best["sortino"],
                             "max_dd": left_best["max_dd"], "total_return": left_best["total_return"]},
                            {"side": "Right", "sharpe": right_best["sharpe"], "sortino": right_best["sortino"],
                             "max_dd": right_best["max_dd"], "total_return": right_best["total_return"]},
                        ])
                        show_df(comp_df, formatters={"max_dd": "{:.2%}", "total_return": "{:.2%}"})
                    else:
                        st.info("Missing optimization results for comparison.")
                except Exception as e:
                    st.warning(f"Comparison error: {str(e)[:120]}")

# ---------------------------
# Export Results
# ---------------------------
st.subheader("Export Results")
def concat_existing_results():
    dfs = []
    # Gather cached DataFrames from session_state for export
    for k, v in st.session_state.items():
        if isinstance(v, pd.DataFrame) and not v.empty:
            dfs.append(v)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

if st.button("Export Results to CSV"):
    results_df = concat_existing_results()
    if not results_df.empty:
        results_df.to_csv("results.csv", index=False)
        st.success("Results exported to results.csv")
    else:
        st.info("No results available to export.")