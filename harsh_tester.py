import pandas as pd
import numpy as np
from itertools import product
import warnings

warnings.filterwarnings('ignore')


class BaseStrategy:
    """Base class for all strategies"""
    def __init__(self, data, params=None):
        self.data = data
        self.params = params or {}

    def generate_signals(self):
        """Override this method in subclasses"""
        raise NotImplementedError


class HarshTester:
    """Production-grade backtester with 20 test suites"""

    def __init__(self, data, init_cash=100000, fee=0.0003, allocation_method="equal_weight"):
        self.data = data.astype(float)
        self.init_cash = init_cash
        self.fee = fee
        self.allocation_method = allocation_method
        self.risk_free_rate = 0.02

        # Initialize for debug output
        self.best_stats = None
        self.best_monthly_stats = None
        self.best_dd_stats = None
        self.debug = True  # Enable debug output

    def backtest_signals(self, entries, exits, data=None):
        """Core backtesting logic with cash + positions equity"""
        if data is None:
            close = self.data.values
            dates = self.data.index
        else:
            close = data.values
            dates = data.index

        # Convert to numpy arrays
        if isinstance(entries, pd.DataFrame):
            entries = entries.values
        if isinstance(exits, pd.DataFrame):
            exits = exits.values

        entries = entries.astype(bool)
        exits = exits.astype(bool)

        n_dates = close.shape[0]
        n_assets = close.shape[1] if close.ndim == 2 else 1

        if close.ndim == 1:
            close = close.reshape(-1, 1)
        if entries.ndim == 1:
            entries = entries.reshape(-1, 1)
        else:
            entries = entries.reshape(-1, 1) if entries.shape[1] == 1 else entries
        if exits.ndim == 1:
            exits = exits.reshape(-1, 1)
        else:
            exits = exits.reshape(-1, 1) if exits.shape[1] == 1 else exits

        positions = np.zeros(n_assets, dtype=float)  # shares per asset
        cash = float(self.init_cash)                  # track cash
        equity_curve = []
        drawdowns = []
        running_max = float(self.init_cash)

        for i in range(n_dates):
            prices = close[i, :]
            entry_signals = entries[i, :] if i < len(entries) else np.zeros(n_assets, dtype=bool)
            exit_signals = exits[i, :] if i < len(exits) else np.zeros(n_assets, dtype=bool)

            for j in range(n_assets):
                if entry_signals[j] and positions[j] == 0:
                    capital_per_asset = cash / max(1, n_assets)
                    if capital_per_asset > 0 and prices[j] > 0:
                        # buy shares, apply entry fee
                        shares = (capital_per_asset / prices[j]) * (1 - self.fee)
                        positions[j] = shares
                        cash -= capital_per_asset
                elif exit_signals[j] and positions[j] > 0:
                    # sell shares, apply exit fee
                    proceeds = positions[j] * prices[j] * (1 - self.fee)
                    cash += proceeds
                    positions[j] = 0.0

            portfolio_value = cash + sum(positions[k] * prices[k] for k in range(n_assets))
            equity_curve.append(portfolio_value)

            if portfolio_value > running_max:
                running_max = portfolio_value
            dd = (portfolio_value - running_max) / running_max if running_max > 0 else 0.0
            drawdowns.append(dd * 100.0)

        # Calculate returns
        final_value = equity_curve[-1] if equity_curve else self.init_cash
        total_return = (final_value - self.init_cash) / self.init_cash

        equity_series = np.array(equity_curve, dtype=float)
        equity_series = np.where(equity_series <= 0, 1e-6, equity_series)
        daily_returns = np.diff(equity_series) / equity_series[:-1]
        daily_returns = np.where(np.isfinite(daily_returns), daily_returns, 0)

        if len(daily_returns) > 1:
            mean_return = np.mean(daily_returns)
            total_std = np.std(daily_returns)

            if total_std > 1e-8:
                sharpe = mean_return / total_std * np.sqrt(252)
            else:
                sharpe = 0.0

            negative_returns = daily_returns[daily_returns < 0]
            if len(negative_returns) > 0:
                downside_std = np.std(negative_returns)
            else:
                downside_std = total_std

            if downside_std > 1e-15:
                sortino = mean_return / downside_std * np.sqrt(252)
            else:
                sortino = 0.0
        else:
            sharpe = 0.0
            sortino = 0.0

        # drawdowns are already in %
        max_dd = min(drawdowns) / 100 if drawdowns else 0.0
        calmar = total_return / abs(max_dd) if abs(max_dd) > 1e-6 else 0.0

        return {
            'sharpe': float(sharpe),
            'sortino': float(sortino),
            'calmar': float(calmar),
            'max_dd': float(max_dd),
            'total_return': float(total_return),
            'final_value': float(final_value),
            'equity_curve': equity_curve,
            'drawdowns': drawdowns,
            'dates': dates
        }

    def stress_test(self, strategy_code: str, param_grid: dict) -> pd.DataFrame:
        """Test 1: Grid Search Backtest"""
        StrategyClass = self.load_strategy_class(strategy_code)
        keys = list(param_grid.keys())
        results = []
        best_stats = None
        best_idx = None

        for idx, combo in enumerate(product(*param_grid.values())):
            params = dict(zip(keys, combo))

            strat = StrategyClass(self.data, params)
            entries, exits = strat.generate_signals()

            if isinstance(entries, pd.Series):
                entries = entries.to_frame()
            if isinstance(exits, pd.Series):
                exits = exits.to_frame()

            stats = self.backtest_signals(entries, exits)
            stats.update(params)
            results.append(stats)

            if best_stats is None or stats['sharpe'] > best_stats['sharpe']:
                best_stats = stats
                best_idx = idx

        # Store best stats for debug output
        self.best_stats = best_stats

        # Debug output
        if self.debug:
            print(f"\n{'='*70}")
            print(f"🔍 TEST 1: STRESS TEST (GRID SEARCH) - DEBUG OUTPUT")
            print(f"{'='*70}")
            print(f"✅ Total parameter combinations: {len(results)}")
            if self.best_stats:
                print(f"✅ best_stats exists: TRUE")
                print(f"   - Sharpe: {self.best_stats.get('sharpe', 'N/A'):.4f}")
                print(f"   - equity_curve length: {len(self.best_stats.get('equity_curve', []))}")
                print(f"   - dates length: {len(self.best_stats.get('dates', []))}")
                print(f"   - max_dd: {self.best_stats.get('max_dd', 'N/A'):.4f}")
            else:
                print(f"❌ best_stats exists: FALSE (data not stored!)")
            print(f"{'='*70}\n")

        return pd.DataFrame(results).drop(columns=['equity_curve', 'drawdowns', 'dates'], errors='ignore')

    def walk_forward_test(self, strategy_code: str, param_grid: dict, train_years: int = 2, test_years: int = 1) -> pd.DataFrame:
        """Test 2: Walk-Forward Validation"""
        StrategyClass = self.load_strategy_class(strategy_code)
        keys = list(param_grid.keys())
        dates = self.data.index

        train_days = int(train_years * 252)
        test_days = int(test_years * 252)

        window_results = []

        for start_idx in range(0, max(1, len(dates) - train_days - test_days), test_days):
            train_end = start_idx + train_days
            test_end = min(train_end + test_days, len(dates))

            if test_end <= train_end or train_end >= len(dates):
                continue

            test_data = self.data.iloc[train_end:test_end]

            for combo in product(*param_grid.values()):
                params = dict(zip(keys, combo))

                strat_test = StrategyClass(test_data, params)
                entries_test, exits_test = strat_test.generate_signals()

                if isinstance(entries_test, pd.Series):
                    entries_test = entries_test.to_frame()
                if isinstance(exits_test, pd.Series):
                    exits_test = exits_test.to_frame()

                test_stats = self.backtest_signals(entries_test, exits_test, data=test_data)
                test_stats.update(params)
                test_stats['window'] = f"{start_idx}-{test_end}"
                window_results.append(test_stats)

        return pd.DataFrame(window_results).drop(columns=['equity_curve', 'drawdowns', 'dates'], errors='ignore') if window_results else pd.DataFrame()

    def monte_carlo_test(self, strategy_code: str, param_grid: dict, n_sims: int = 500) -> pd.DataFrame:
        """Test 3: Monte Carlo Stress Test"""
        StrategyClass = self.load_strategy_class(strategy_code)
        keys = list(param_grid.keys())
        base_results = []

        for combo in product(*param_grid.values()):
            params = dict(zip(keys, combo))
            strat = StrategyClass(self.data, params)
            entries, exits = strat.generate_signals()

            if isinstance(entries, pd.Series):
                entries = entries.to_frame()
            if isinstance(exits, pd.Series):
                exits = exits.to_frame()

            base_stats = self.backtest_signals(entries, exits)
            base_results.append((*params.items(), base_stats))

        mc_results = []
        for result in base_results:
            equity = np.array(result[-1]['equity_curve'], dtype=float)
            equity = np.where(equity <= 0, 1e-6, equity)
            returns = np.diff(equity) / equity[:-1]
            returns = np.where(np.isfinite(returns), returns, 0)

            if len(returns) > 10 and np.std(returns) > 1e-8:
                mc_sharpes, mc_sortinos, mc_dds = [], [], []

                for _ in range(n_sims):
                    resampled_ret = np.random.choice(returns, size=len(returns), replace=True)
                    resampled_eq = np.cumprod(1 + resampled_ret) * self.init_cash

                    ret_mean = np.mean(resampled_ret)
                    ret_std = np.std(resampled_ret)
                    mc_sharpe = ret_mean / ret_std * np.sqrt(252) if ret_std > 1e-8 else 0

                    neg_ret = resampled_ret[resampled_ret < 0]
                    if len(neg_ret) > 0:
                        down_std = np.std(neg_ret)
                    else:
                        down_std = ret_std

                    if down_std > 1e-15:
                        mc_sortino = ret_mean / down_std * np.sqrt(252)
                    else:
                        mc_sortino = 0.0

                    running_max = np.maximum.accumulate(resampled_eq)
                    dd = np.where(running_max > 0, (resampled_eq - running_max) / running_max, 0)
                    mc_dd = float(np.min(dd))

                    mc_sharpes.append(float(mc_sharpe))
                    mc_sortinos.append(float(mc_sortino))
                    mc_dds.append(mc_dd)

                result_dict = {k: v for k, v in result[:-1]}
                result_dict.update({
                    'base_sharpe': float(result[-1]['sharpe']),
                    'base_sortino': float(result[-1]['sortino']),
                    'mc_sharpe_5pct': float(np.percentile(mc_sharpes, 5)),
                    'mc_sharpe_mean': float(np.mean(mc_sharpes)),
                    'mc_sharpe_95pct': float(np.percentile(mc_sharpes, 95)),
                    'mc_sortino_5pct': float(np.percentile(mc_sortinos, 5)),
                    'mc_sortino_mean': float(np.mean(mc_sortinos)),
                    'mc_sortino_95pct': float(np.percentile(mc_sortinos, 95)),
                    'base_max_dd': float(result[-1]['max_dd']),
                    'mc_dd_5pct': float(np.percentile(mc_dds, 5)),
                    'mc_dd_mean': float(np.mean(mc_dds)),
                    'mc_dd_95pct': float(np.percentile(mc_dds, 95))
                })
            else:
                result_dict = {k: v for k, v in result[:-1]}
                result_dict.update({
                    'base_sharpe': float(result[-1]['sharpe']),
                    'base_sortino': float(result[-1]['sortino']),
                    'mc_sharpe_5pct': float(result[-1]['sharpe']),
                    'mc_sharpe_mean': float(result[-1]['sharpe']),
                    'mc_sharpe_95pct': float(result[-1]['sharpe']),
                    'mc_sortino_5pct': float(result[-1]['sortino']),
                    'mc_sortino_mean': float(result[-1]['sortino']),
                    'mc_sortino_95pct': float(result[-1]['sortino']),
                    'base_max_dd': float(result[-1]['max_dd']),
                    'mc_dd_5pct': float(result[-1]['max_dd']),
                    'mc_dd_mean': float(result[-1]['max_dd']),
                    'mc_dd_95pct': float(result[-1]['max_dd'])
                })

            mc_results.append(result_dict)

        return pd.DataFrame(mc_results) if mc_results else pd.DataFrame()

    def parameter_sensitivity_test(self, strategy_code: str, param_grid: dict, perturbation: float = 0.2, metric: str = 'sharpe') -> pd.DataFrame:
        """Test 4: Parameter Sensitivity"""
        try:
            base_results = self.stress_test(strategy_code, param_grid)
            if base_results.empty:
                return pd.DataFrame()

            # Get best row by sharpe
            best_row = base_results.loc[base_results['sharpe'].idxmax()]
            results = []

            # Only iterate over parameters that exist in param_grid
            for param_name in param_grid.keys():
                if param_name not in best_row.index:
                    continue

                try:
                    original_value = float(best_row[param_name])

                    for perturbation_type in [-perturbation, 0, perturbation]:
                        try:
                            new_val = original_value * (1 + perturbation_type)
                            if isinstance(param_grid[param_name][0], int):
                                new_val = max(1, int(new_val))

                            test_grid = {param_name: [new_val]}
                            test_results = self.stress_test(strategy_code, test_grid)

                            if not test_results.empty:
                                best_test_row = test_results.loc[test_results['sharpe'].idxmax()]
                                results.append({
                                    'parameter': param_name,
                                    'value': float(new_val),
                                    'sharpe': float(best_test_row['sharpe']),
                                    'sortino': float(best_test_row['sortino']),
                                    'return': float(best_test_row['total_return']),
                                    'max_dd': float(best_test_row['max_dd']),
                                    'change_pct': float(perturbation_type * 100)
                                })
                        except:
                            continue
                except:
                    continue

            return pd.DataFrame(results) if results else pd.DataFrame()
        except Exception as e:
            print(f"Parameter Sensitivity Error: {e}")
            return pd.DataFrame()

    def transaction_cost_test(self, strategy_code: str, param_grid: dict, cost_levels: list = None) -> pd.DataFrame:
        """Test 5: Transaction Cost Impact"""
        if cost_levels is None:
            cost_levels = [0.0, 0.0005, 0.001, 0.002]

        results = []
        original_fee = self.fee

        for cost in cost_levels:
            self.fee = cost
            grid_results = self.stress_test(strategy_code, param_grid)

            if not grid_results.empty:
                results.append({
                    'fee_level': f"{cost:.4f}",
                    'avg_sharpe': grid_results['sharpe'].mean(),
                    'best_sharpe': grid_results['sharpe'].max(),
                    'avg_sortino': grid_results['sortino'].mean(),
                    'best_sortino': grid_results['sortino'].max(),
                    'avg_return': grid_results['total_return'].mean(),
                    'best_return': grid_results['total_return'].max()
                })

        self.fee = original_fee
        return pd.DataFrame(results) if results else pd.DataFrame()

    def rolling_metrics_test(self, strategy_code: str, param_grid: dict, window_years: int = 1) -> pd.DataFrame:
        """Test 6: Rolling Metrics"""
        StrategyClass = self.load_strategy_class(strategy_code)
        keys = list(param_grid.keys())
        window_days = int(window_years * 252)
        results = []

        for start_idx in range(0, len(self.data) - window_days, window_days):
            window_data = self.data.iloc[start_idx:start_idx + window_days]

            for combo in product(*param_grid.values()):
                params = dict(zip(keys, combo))
                strat = StrategyClass(window_data, params)
                entries, exits = strat.generate_signals()

                if isinstance(entries, pd.Series):
                    entries = entries.to_frame()
                if isinstance(exits, pd.Series):
                    exits = exits.to_frame()

                stats = self.backtest_signals(entries, exits, data=window_data)

                results.append({
                    'period': window_data.index[0].strftime('%Y-%m-%d'),
                    **params,
                    'sharpe': stats['sharpe'],
                    'sortino': stats['sortino'],
                    'return': stats['total_return'],
                    'max_dd': stats['max_dd']
                })

        return pd.DataFrame(results) if results else pd.DataFrame()

    def crisis_stress_test(self, strategy_code: str, param_grid: dict, crisis_dates: dict = None) -> pd.DataFrame:
        """Test 7: Crisis Stress Tests"""
        if crisis_dates is None:
            crisis_dates = {
                '2020-covid': ('2020-01-01', '2020-06-30'),
                '2022-rates': ('2021-10-01', '2023-01-31'),
                '2023-banking': ('2022-12-01', '2023-07-31')
            }

        StrategyClass = self.load_strategy_class(strategy_code)
        keys = list(param_grid.keys())
        results = []

        for crisis_name, (start, end) in crisis_dates.items():
            try:
                crisis_data = self.data.loc[start:end]
                if crisis_data.empty or len(crisis_data) < 60:
                    continue

                for combo in product(*param_grid.values()):
                    params = dict(zip(keys, combo))
                    strat = StrategyClass(crisis_data, params)
                    entries, exits = strat.generate_signals()

                    if isinstance(entries, pd.Series):
                        entries = entries.to_frame()
                    if isinstance(exits, pd.Series):
                        exits = exits.to_frame()

                    stats = self.backtest_signals(entries, exits, data=crisis_data)

                    results.append({
                        'crisis': crisis_name,
                        **params,
                        'sharpe': stats['sharpe'],
                        'sortino': stats['sortino'],
                        'max_dd': stats['max_dd'],
                        'return': stats['total_return'],
                        'days': len(crisis_data)
                    })
            except:
                continue

        return pd.DataFrame(results) if results else pd.DataFrame()

    def monthly_consistency_test(self, strategy_code: str, param_grid: dict) -> tuple:
        """Test 8: Monthly Returns Consistency"""
        StrategyClass = self.load_strategy_class(strategy_code)
        keys = list(param_grid.keys())
        results = []
        best_stats = None

        for combo in product(*param_grid.values()):
            params = dict(zip(keys, combo))
            strat = StrategyClass(self.data, params)
            entries, exits = strat.generate_signals()

            if isinstance(entries, pd.Series):
                entries = entries.to_frame()
            if isinstance(exits, pd.Series):
                exits = exits.to_frame()

            stats = self.backtest_signals(entries, exits)

            if best_stats is None or stats['sharpe'] > best_stats['sharpe']:
                best_stats = stats

            equity_curve = np.array(stats['equity_curve'], dtype=float) if 'equity_curve' in stats else None

            if equity_curve is not None and len(equity_curve) > 21:
                monthly_returns = []
                for i in range(0, len(equity_curve) - 21, 21):
                    month_start = equity_curve[i]
                    month_end = equity_curve[i + 21]
                    if month_start > 0:
                        monthly_returns.append((month_end - month_start) / month_start)
                avg_monthly_return = np.mean(monthly_returns) if monthly_returns else 0.0
                monthly_volatility = np.std(monthly_returns) if len(monthly_returns) > 1 else 0.0
            else:
                months = max(len(self.data) // 252, 1)
                avg_monthly_return = stats['total_return'] / months
                monthly_volatility = 0.0

            results.append({
                **params,
                'avg_monthly_return': avg_monthly_return,
                'monthly_volatility': monthly_volatility,
                'total_return': stats['total_return'],
                'sharpe': stats['sharpe'],
                'sortino': stats['sortino']
            })

        self.best_monthly_stats = best_stats

        if self.debug:
            print(f"✅ TEST 8: Monthly analysis completed: {len(results)} rows")
            print(f"✅ best_monthly_stats exists: {self.best_monthly_stats is not None}")
            if self.best_monthly_stats:
                print(f"✅ best_monthly_stats equity_curve length: {len(self.best_monthly_stats.get('equity_curve', []))}\n")
            else:
                print(f"❌ best_monthly_stats is None\n")

        return pd.DataFrame(results) if results else pd.DataFrame(), best_stats

    def drawdown_analysis_test(self, strategy_code: str, param_grid: dict) -> tuple:
        """Test 9: Drawdown Analysis"""
        StrategyClass = self.load_strategy_class(strategy_code)
        keys = list(param_grid.keys())
        best_combo = None
        best_sharpe = -np.inf
        best_stats = None

        for combo in product(*param_grid.values()):
            params = dict(zip(keys, combo))
            strat = StrategyClass(self.data, params)
            entries, exits = strat.generate_signals()

            if isinstance(entries, pd.Series):
                entries = entries.to_frame()
            if isinstance(exits, pd.Series):
                exits = exits.to_frame()

            stats = self.backtest_signals(entries, exits)

            if stats['sharpe'] > best_sharpe:
                best_sharpe = stats['sharpe']
                best_combo = combo
                best_stats = stats

        if best_combo and best_stats is not None:
            # Store best stats for debug output
            self.best_dd_stats = best_stats

            if self.debug:
                print(f"✅ TEST 9: Drawdown analysis completed")
                print(f"✅ best_dd_stats exists: {self.best_dd_stats is not None}")
                if self.best_dd_stats:
                    print(f"✅ best_dd_stats drawdowns length: {len(self.best_dd_stats.get('drawdowns', []))}\n")
                else:
                    print(f"❌ best_dd_stats is None\n")

            return pd.DataFrame([{k: v for k, v in best_stats.items() if k not in ['equity_curve', 'drawdowns', 'dates']}]), best_stats

        if self.debug:
            print(f"❌ TEST 9: No best_dd_stats found\n")

        return pd.DataFrame(), None

    def kelly_criterion_test(self, strategy_code: str, param_grid: dict) -> pd.DataFrame:
        """Test 10: Kelly Criterion"""
        base_results = self.stress_test(strategy_code, param_grid)

        if base_results.empty:
            return pd.DataFrame({
                'full_kelly': [0.0],
                'half_kelly': [0.0],
                'quarter_kelly': [0.0],
                'win_rate': [0.0],
                'avg_win': [0.0],
                'avg_loss': [0.0],
                'status': ['No data']
            })

        num_wins = (base_results['total_return'] > 0).sum()
        num_losses = (base_results['total_return'] < 0).sum()
        total_combos = len(base_results)

        win_rate = num_wins / total_combos if total_combos > 0 else 0

        if num_wins > 0:
            avg_win = base_results[base_results['total_return'] > 0]['total_return'].mean()
        else:
            avg_win = 0.01

        if num_losses > 0:
            avg_loss = abs(base_results[base_results['total_return'] < 0]['total_return'].mean())
        else:
            avg_loss = 0.01

        if abs(avg_loss) > 1e-6:
            kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss
        else:
            kelly = 0.0

        kelly = max(0, min(kelly, 1))

        if kelly == 0.0:
            status = "Not tradeable: Negative edge"
        elif kelly < 0.1:
            status = "Very risky: Low edge"
        else:
            status = "Tradeable"

        return pd.DataFrame({
            'full_kelly': [kelly * 100],
            'half_kelly': [kelly * 50],
            'quarter_kelly': [kelly * 25],
            'win_rate': [win_rate * 100],
            'avg_win': [avg_win * 100],
            'avg_loss': [avg_loss * 100],
            'status': [status],
            'num_wins': [num_wins],
            'num_losses': [num_losses]
        })

    def load_strategy_class(self, strategy_code: str):
        """Load strategy from code string"""
        import importlib.util
        spec = importlib.util.spec_from_loader("strategy_module", loader=None)
        module = importlib.util.module_from_spec(spec)
        exec(strategy_code, module.__dict__)
        return module.Strategy

    # =========================
    # Helper utilities (private)
    # =========================
    def _extract_trades(self, data, entries, exits):
        """Return list of trades with per-trade returns, dates, asset"""
        if isinstance(entries, pd.Series):
            entries = entries.to_frame()
        if isinstance(exits, pd.Series):
            exits = exits.to_frame()
        entries = entries.astype(bool).values
        exits = exits.astype(bool).values

        close = data.values.astype(float)
        dates = data.index
        n_dates = close.shape[0]
        n_assets = close.shape[1] if close.ndim == 2 else 1
        if close.ndim == 1:
            close = close.reshape(-1, 1)

        trades = []
        for j in range(n_assets):
            in_trade = False
            entry_idx = None
            for i in range(n_dates):
                e = entries[i, j] if entries.ndim == 2 else bool(entries[i])
                x = exits[i, j] if exits.ndim == 2 else bool(exits[i])
                if e and not in_trade:
                    in_trade = True
                    entry_idx = i
                elif x and in_trade:
                    exit_idx = i
                    entry_price = float(close[entry_idx, j])
                    exit_price = float(close[exit_idx, j])
                    # Net return: price change minus entry fee
                    ret = (exit_price / entry_price - 1.0) - float(self.fee)
                    trades.append({
                        'asset': j,
                        'entry_idx': entry_idx,
                        'exit_idx': exit_idx,
                        'entry_date': dates[entry_idx],
                        'exit_date': dates[exit_idx],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'ret': float(ret),
                        'duration_days': int(exit_idx - entry_idx)
                    })
                    in_trade = False
                    entry_idx = None
        return trades

    def _equity_from_trades(self, trade_returns, initial_equity=None, f=1.0):
        """Compound equity using per-trade returns scaled by fraction f"""
        equity = float(initial_equity if initial_equity is not None else self.init_cash)
        curve = [equity]
        for r in trade_returns:
            equity = equity * (1.0 + f * float(r))
            curve.append(equity)
        return np.array(curve, dtype=float)

    def _max_drawdown_from_equity(self, eq):
        if len(eq) < 2:
            return 0.0
        running_max = np.maximum.accumulate(eq)
        dd = (eq - running_max) / np.where(running_max > 0, running_max, 1.0)
        return float(np.min(dd))

    def _sharpe_from_returns(self, rets):
        rets = np.array(rets, dtype=float)
        rets = rets[np.isfinite(rets)]
        if len(rets) < 2:
            return 0.0
        mu = np.mean(rets)
        sd = np.std(rets)
        return float(mu / sd * np.sqrt(252)) if sd > 1e-8 else 0.0

    def _ks_divergence(self, a, b):
        """Simple distribution divergence via KS statistic (no external deps)"""
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 5 or len(b) < 5:
            return float('nan')
        xa = np.sort(a)
        xb = np.sort(b)
        # Empirical CDF grids
        grid = np.sort(np.unique(np.concatenate([xa, xb])))
        Fa = np.searchsorted(xa, grid, side='right') / len(xa)
        Fb = np.searchsorted(xb, grid, side='right') / len(xb)
        return float(np.max(np.abs(Fa - Fb)))

    def _realized_vol(self, data, window=20):
        """Daily realized vol (rolling std of pct change) per asset"""
        returns = data.pct_change().fillna(0.0)
        return returns.rolling(window).std().fillna(method='bfill').fillna(0.0)

    # =========================
    # Test 11: Conditional P/L Decomposition (EV Breakdown)
    # =========================
    def test_conditional_decomposition(self, strategy_code, param_grid):
        StrategyClass = self.load_strategy_class(strategy_code)
        keys = list(param_grid.keys())
        results = []
        for combo in product(*param_grid.values()):
            params = dict(zip(keys, combo))
            strat = StrategyClass(self.data, params)
            entries, exits = strat.generate_signals()
            trades = self._extract_trades(self.data, entries, exits)

            if len(trades) == 0:
                results.append({
                    'param_set': params,
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'ev_per_trade': 0.0,
                    'ev_contribution_win': 0.0,
                    'ev_contribution_loss': 0.0
                })
                continue

            rets = np.array([t['ret'] for t in trades], dtype=float)
            wins = rets[rets > 0]
            losses = rets[rets <= 0]
            p_win = len(wins) / len(rets)
            p_loss = 1 - p_win
            avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
            avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
            ev_win = avg_win * p_win
            ev_loss = avg_loss * p_loss
            ev_per_trade = float(np.mean(rets)) if len(rets) > 0 else 0.0

            results.append({
                'param_set': params,
                'total_trades': int(len(rets)),
                'win_rate': float(p_win * 100),
                'avg_win': float(avg_win * 100),
                'avg_loss': float(avg_loss * 100),
                'ev_per_trade': float(ev_per_trade * 100),
                'ev_contribution_win': float(ev_win * 100),
                'ev_contribution_loss': float(ev_loss * 100)
            })
        return pd.DataFrame(results)

    # =========================
    # Test 12: Time Slice Stability
    # =========================
    def test_time_slice_stability(self, strategy_code, param_grid, n_slices=4):
        StrategyClass = self.load_strategy_class(strategy_code)
        keys = list(param_grid.keys())
        n = len(self.data)
        if n_slices < 2 or n < n_slices * 40:
            return pd.DataFrame()
        idxs = np.linspace(0, n, n_slices + 1, dtype=int)
        results = []
        for combo in product(*param_grid.values()):
            params = dict(zip(keys, combo))
            slice_evs = []
            for s in range(n_slices):
                sl = slice(idxs[s], idxs[s+1])
                dslice = self.data.iloc[sl]
                if len(dslice) < 40:
                    slice_evs.append(0.0)
                    continue
                strat = StrategyClass(dslice, params)
                entries, exits = strat.generate_signals()
                trades = self._extract_trades(dslice, entries, exits)
                rets = np.array([t['ret'] for t in trades], dtype=float)
                ev = float(np.mean(rets)) if len(rets) > 0 else 0.0
                slice_evs.append(ev)

            ev_arr = np.array(slice_evs, dtype=float)
            ev_mean = float(np.mean(ev_arr)) if len(ev_arr) > 0 else 0.0
            ev_std = float(np.std(ev_arr)) if len(ev_arr) > 1 else 0.0
            ev_cv = float(ev_std / ev_mean) if abs(ev_mean) > 1e-12 else float('inf')
            stability = float(max(0.0, 1.0 - min(1.0, ev_cv)))

            row = {'param_set': params, 'ev_cv': ev_cv, 'ev_stability_score': stability}
            for s in range(n_slices):
                row[f'slice_{s+1}_ev'] = float(ev_arr[s]) if s < len(ev_arr) else 0.0
            results.append(row)
        return pd.DataFrame(results)

    # =========================
    # Test 13: Regime Dependency (Volatility terciles)
    # =========================
    def test_regime_dependency(self, strategy_code, param_grid, regime_var='volatility'):
        StrategyClass = self.load_strategy_class(strategy_code)
        keys = list(param_grid.keys())
        results = []

        if regime_var == 'volatility':
            vol = self._realized_vol(self.data, window=20)
            vol_flat = vol.values.flatten()
            vol_flat = vol_flat[np.isfinite(vol_flat)]
            if len(vol_flat) < 50:
                return pd.DataFrame()
            terciles = np.percentile(vol_flat, [33.33, 66.67])
        else:
            # fallback: use absolute daily return magnitude as regime variable
            absret = self.data.pct_change().abs()
            abs_flat = absret.values.flatten()
            abs_flat = abs_flat[np.isfinite(abs_flat)]
            terciles = np.percentile(abs_flat, [33.33, 66.67])

        for combo in product(*param_grid.values()):
            params = dict(zip(keys, combo))
            strat = StrategyClass(self.data, params)
            entries, exits = strat.generate_signals()
            trades = self._extract_trades(self.data, entries, exits)
            if len(trades) == 0:
                continue

            # assign regime at entry by asset’s realized vol
            regime_stats = {0: [], 1: [], 2: []}
            for t in trades:
                i = t['entry_idx']
                j = t['asset']
                v = float(vol.iloc[i, j]) if regime_var == 'volatility' else float(absret.iloc[i, j])
                if v <= terciles[0]:
                    regime_stats[0].append(t['ret'])
                elif v <= terciles[1]:
                    regime_stats[1].append(t['ret'])
                else:
                    regime_stats[2].append(t['ret'])

            reg_rows = {}
            evs = []
            for r in [0, 1, 2]:
                rs = np.array(regime_stats[r], dtype=float)
                ev = float(np.mean(rs)) if len(rs) > 0 else 0.0
                evs.append(ev)
                eq = self._equity_from_trades(rs, initial_equity=self.init_cash, f=1.0)
                dd = self._max_drawdown_from_equity(eq)
                reg_rows[f'regime_{r+1}_name'] = ['low','med','high'][r]
                reg_rows[f'regime_{r+1}_ev'] = ev
                reg_rows[f'regime_{r+1}_dd'] = dd

            consistency = float(min(evs) / max(evs)) if max(evs) != 0 else 0.0
            results.append({'param_set': params, **reg_rows, 'regime_consistency_score': consistency})
        return pd.DataFrame(results)

    # =========================
    # Test 14: Win/Loss Distribution Stability (KS divergence)
    # =========================
    def test_win_loss_distribution_stability(self, strategy_code, param_grid, n_slices=2):
        StrategyClass = self.load_strategy_class(strategy_code)
        keys = list(param_grid.keys())
        n = len(self.data)
        if n_slices < 2 or n < n_slices * 40:
            return pd.DataFrame()
        idxs = np.linspace(0, n, n_slices + 1, dtype=int)
        results = []

        for combo in product(*param_grid.values()):
            params = dict(zip(keys, combo))
            slice_wins = []
            slice_losses = []
            for s in range(n_slices):
                dslice = self.data.iloc[idxs[s]:idxs[s+1]]
                strat = StrategyClass(dslice, params)
                entries, exits = strat.generate_signals()
                trades = self._extract_trades(dslice, entries, exits)
                rets = np.array([t['ret'] for t in trades], dtype=float)
                slice_wins.append(rets[rets > 0])
                slice_losses.append(rets[rets <= 0])

            # measure divergence between first and last slice
            winner_div = self._ks_divergence(slice_wins[0], slice_wins[-1])
            loser_div = self._ks_divergence(slice_losses[0], slice_losses[-1])
            # combine (lower is better)
            if np.isfinite(winner_div) and np.isfinite(loser_div):
                combined = float(1.0 - min(1.0, 0.5 * (winner_div + loser_div)))
            else:
                combined = float('nan')

            results.append({
                'param_set': params,
                'winner_kde_divergence': winner_div,
                'loser_kde_divergence': loser_div,
                'combined_distribution_score': combined
            })
        return pd.DataFrame(results)

    # =========================
    # Test 15: Drawdown Under Realistic Sizing (Path Dependence)
    # =========================
    def test_sized_drawdown_simulation(self, strategy_code, param_grid, kelly_fraction=0.25, initial_capital=100000):
        StrategyClass = self.load_strategy_class(strategy_code)
        keys = list(param_grid.keys())
        results = []

        for combo in product(*param_grid.values()):
            params = dict(zip(keys, combo))
            strat = StrategyClass(self.data, params)
            entries, exits = strat.generate_signals()
            trades = self._extract_trades(self.data, entries, exits)
            rets = np.array([t['ret'] for t in trades], dtype=float)
            mu = float(np.mean(rets)) if len(rets) > 0 else 0.0
            sig = float(np.std(rets)) if len(rets) > 1 else 0.0
            kelly = float(mu / (sig**2)) if sig > 1e-8 else 0.0
            kelly = max(0.0, min(kelly, 1.0))
            f_kelly = float(kelly_fraction * kelly)

            eq_kelly = self._equity_from_trades(rets, initial_equity=initial_capital, f=f_kelly)
            dd_kelly = self._max_drawdown_from_equity(eq_kelly)
            final_kelly = float(eq_kelly[-1]) if len(eq_kelly) > 0 else float(initial_capital)

            f_fixed = 0.02
            eq_fixed = self._equity_from_trades(rets, initial_equity=initial_capital, f=f_fixed)
            dd_fixed = self._max_drawdown_from_equity(eq_fixed)
            final_fixed = float(eq_fixed[-1]) if len(eq_fixed) > 0 else float(initial_capital)

            # simple robustness score: penalize large DD
            rob = float(max(0.0, 1.0 - max(abs(dd_kelly), abs(dd_fixed)) / 0.6))

            results.append({
                'param_set': params,
                'kelly_frac': f_kelly,
                'max_dd_kelly': dd_kelly,
                'final_equity_kelly': final_kelly,
                'max_dd_fixed': dd_fixed,
                'final_equity_fixed': final_fixed,
                'sizing_robustness_score': rob
            })
        return pd.DataFrame(results)

    # =========================
    # Test 16: Parameter Sensitivity & Robustness
    # =========================
    def test_parameter_robustness(self, strategy_code, param_grid):
        base = self.stress_test(strategy_code, param_grid)
        if base.empty:
            return pd.DataFrame()
        # baseline: best by Sharpe
        best_row = base.loc[base['sharpe'].idxmax()]
        best_params = {k: best_row[k] for k in param_grid.keys()}
        best_ev = float(best_row['total_return'])

        results = []
        for pname, pvals in param_grid.items():
            original = float(best_params[pname]) if pname in best_params else float(pvals[0])
            neigh_evs = []
            sensitivities = []
            for delta in [-0.10, 0.10]:
                new_val = original * (1.0 + delta)
                new_val = int(round(new_val)) if isinstance(pvals[0], int) else float(new_val)
                grid = {pname: [new_val]}
                test = self.stress_test(strategy_code, grid)
                if not test.empty:
                    ev = float(test['total_return'].mean())
                    neigh_evs.append(ev)
                    sensitivities.append(float(best_ev - ev))

            if len(neigh_evs) == 0:
                robustness = 0.0
                sens_drop = float('nan')
            else:
                threshold = best_ev * 0.9
                robustness = float(np.mean([1.0 if ev >= threshold else 0.0 for ev in neigh_evs]))
                sens_drop = float(np.mean(sensitivities)) if len(sensitivities) > 0 else float('nan')

            results.append({
                'param_set': best_params,
                'ev': best_ev,
                'parameter': pname,
                'robustness_score': robustness,
                'sensitivity_drop': sens_drop
            })
        return pd.DataFrame(results)

    # =========================
    # Test 17: Market Regime Drift (Beta drift)
    # =========================
    def test_market_regime_drift(self, strategy_code, param_grid):
        StrategyClass = self.load_strategy_class(strategy_code)
        keys = list(param_grid.keys())
        # proxy market: equal-weighted close across assets
        market_series = self.data.mean(axis=1)
        market_ret = market_series.pct_change().fillna(0.0).values
        window = 63
        results = []

        for combo in product(*param_grid.values()):
            params = dict(zip(keys, combo))
            strat = StrategyClass(self.data, params)
            entries, exits = strat.generate_signals()
            stats = self.backtest_signals(entries, exits)
            eq = np.array(stats['equity_curve'], dtype=float)
            eq = np.where(eq <= 0, 1e-6, eq)
            strat_ret = np.diff(eq) / eq[:-1]
            strat_ret = np.where(np.isfinite(strat_ret), strat_ret, 0.0)

            L = min(len(strat_ret), len(market_ret))
            sr = strat_ret[:L]
            mr = market_ret[:L]

            # rolling correlation (beta proxy)
            betas = []
            for i in range(0, L - window + 1):
                s = sr[i:i+window]
                m = mr[i:i+window]
                if np.std(m) > 1e-8 and np.std(s) > 1e-8:
                    corr = np.corrcoef(s, m)[0,1]
                else:
                    corr = 0.0
                betas.append(float(corr))
            beta_mean = float(np.mean(betas)) if len(betas) > 0 else 0.0
            beta_std = float(np.std(betas)) if len(betas) > 1 else 0.0

            # bull vs bear via 126-day SMA direction
            sma = pd.Series(market_series).rolling(126).mean().values
            up_mask = (market_series.values > sma)
            down_mask = ~up_mask
            bull_rets = sr[up_mask[:L]] if np.any(up_mask[:L]) else np.array([])
            bear_rets = sr[down_mask[:L]] if np.any(down_mask[:L]) else np.array([])
            bull_sharpe = self._sharpe_from_returns(bull_rets)
            bear_sharpe = self._sharpe_from_returns(bear_rets)

            regime_shift_sensitivity = float(beta_std)

            results.append({
                'param_set': params,
                'bull_market_sharpe': bull_sharpe,
                'bear_market_sharpe': bear_sharpe,
                'rolling_beta_mean': beta_mean,
                'rolling_beta_std': beta_std,
                'regime_shift_sensitivity': regime_shift_sensitivity
            })
        return pd.DataFrame(results)

    # =========================
    # Test 18: Slippage & Commissions Impact (Cost Realism)
    # =========================
    def test_slippage_impact(self, strategy_code, param_grid, slippage_bps=1, commission_bps=0.5):
        StrategyClass = self.load_strategy_class(strategy_code)
        keys = list(param_grid.keys())
        slip = slippage_bps / 10000.0
        comm = commission_bps / 10000.0
        per_trade_cost = 2.0 * (slip + comm)  # entry + exit

        results = []
        for combo in product(*param_grid.values()):
            params = dict(zip(keys, combo))
            strat = StrategyClass(self.data, params)
            entries, exits = strat.generate_signals()
            trades = self._extract_trades(self.data, entries, exits)
            rets = np.array([t['ret'] for t in trades], dtype=float)
            ev_no = float(np.mean(rets)) if len(rets) > 0 else 0.0
            sharpe_no = self._sharpe_from_returns(rets)

            rets_cost = rets - per_trade_cost
            ev_with = float(np.mean(rets_cost)) if len(rets_cost) > 0 else 0.0
            sharpe_with = self._sharpe_from_returns(rets_cost)
            degr = float((ev_no - ev_with) / abs(ev_no)) if abs(ev_no) > 1e-12 else float('inf')
            survives = bool(sharpe_with >= 0.8 * sharpe_no and ev_with > 0)

            results.append({
                'param_set': params,
                'ev_no_cost': ev_no,
                'ev_with_cost': ev_with,
                'cost_degradation_pct': degr * 100.0 if np.isfinite(degr) else float('inf'),
                'sharpe_no_cost': sharpe_no,
                'sharpe_with_cost': sharpe_with,
                'edge_survives_costs': survives
            })
        return pd.DataFrame(results)

    # =========================
    # Test 19: Consecutive Losing Streaks & Drawdown Severity
    # =========================
    def test_drawdown_severity(self, strategy_code, param_grid):
        StrategyClass = self.load_strategy_class(strategy_code)
        keys = list(param_grid.keys())
        results = []

        for combo in product(*param_grid.values()):
            params = dict(zip(keys, combo))
            strat = StrategyClass(self.data, params)
            entries, exits = strat.generate_signals()
            trades = self._extract_trades(self.data, entries, exits)
            rets = np.array([t['ret'] for t in trades], dtype=float)

            # max consecutive losses
            max_loss_streak = 0
            streak = 0
            for r in rets:
                if r <= 0:
                    streak += 1
                    max_loss_streak = max(max_loss_streak, streak)
                else:
                    streak = 0

            # daily equity drawdown and recovery time
            stats = self.backtest_signals(entries, exits)
            eq = np.array(stats.get('equity_curve', []), dtype=float)
            if len(eq) > 0:
                running_max = np.maximum.accumulate(eq)
                dd = (eq - running_max) / np.where(running_max > 0, running_max, 1.0)
                min_dd_idx = int(np.argmin(dd))
                recovery_idx = None
                peak_val = running_max[min_dd_idx]
                for k in range(min_dd_idx+1, len(eq)):
                    if eq[k] >= peak_val:
                        recovery_idx = k
                        break
                recovery_days = int(recovery_idx - min_dd_idx) if recovery_idx is not None else int(len(eq) - min_dd_idx)
                max_intra_dd = float(np.min(dd))
            else:
                recovery_days = 0
                max_intra_dd = 0.0

            p_win = float(np.mean(rets > 0)) if len(rets) > 0 else 0.0
            prob_10_loss = float((1.0 - p_win)**10)
            severity_score = float(max(0.0, 1.0 - max(0.0, abs(max_intra_dd)) / 0.6))

            results.append({
                'param_set': params,
                'max_consecutive_losses': int(max_loss_streak),
                'max_intra_dd': max_intra_dd,
                'recovery_time_days': int(recovery_days),
                'prob_10_loss_streak': prob_10_loss,
                'drawdown_severity_score': severity_score
            })
        return pd.DataFrame(results)

    # =========================
    # Test 20: Out-of-Sample Validation (Walk-Forward)
    # =========================
    def test_walk_forward_validation(self, strategy_code, param_grid, train_pct=0.7, step_size=0.1):
        StrategyClass = self.load_strategy_class(strategy_code)
        n = len(self.data)
        if n < 200:
            return pd.DataFrame()
        step = int(max(1, n * step_size))
        train_len = int(n * train_pct)
        fold = 0
        results = []
        for start in range(0, n - train_len - step + 1, step):
            fold += 1
            train_slice = slice(start, start + train_len)
            test_slice = slice(start + train_len, min(start + train_len + int(n * 0.2), n))
            train_data = self.data.iloc[train_slice]
            test_data = self.data.iloc[test_slice]
            if len(test_data) < 40:
                continue

            # choose best params on train by Sharpe (current implementation)
            base_train = self.stress_test(strategy_code, param_grid)
            if base_train.empty:
                continue
            best_row = base_train.loc[base_train['sharpe'].idxmax()]
            best_params = {k: best_row[k] for k in param_grid.keys()}

            # evaluate train metrics (EV and Sharpe via backtest on train)
            strat_train = StrategyClass(train_data, best_params)
            e_tr, x_tr = strat_train.generate_signals()
            stats_tr = self.backtest_signals(e_tr, x_tr, data=train_data)
            eq_tr = np.array(stats_tr['equity_curve'], dtype=float)
            eq_tr = np.where(eq_tr <= 0, 1e-6, eq_tr)
            r_tr = np.diff(eq_tr) / eq_tr[:-1]
            r_tr = np.where(np.isfinite(r_tr), r_tr, 0.0)
            train_ev = float(np.mean(r_tr)) if len(r_tr) > 0 else 0.0
            train_sharpe = self._sharpe_from_returns(r_tr)

            # evaluate on test
            strat_test = StrategyClass(test_data, best_params)
            e_te, x_te = strat_test.generate_signals()
            stats_te = self.backtest_signals(e_te, x_te, data=test_data)
            eq_te = np.array(stats_te['equity_curve'], dtype=float)
            eq_te = np.where(eq_te <= 0, 1e-6, eq_te)
            r_te = np.diff(eq_te) / eq_te[:-1]
            r_te = np.where(np.isfinite(r_te), r_te, 0.0)
            test_ev = float(np.mean(r_te)) if len(r_te) > 0 else 0.0
            test_sharpe = self._sharpe_from_returns(r_te)

            overfit_ratio = float(train_sharpe / test_sharpe) if test_sharpe > 1e-8 else float('inf')
            is_overfit = bool(overfit_ratio > 1.5)

            results.append({
                'fold_num': fold,
                'train_ev': train_ev,
                'test_ev': test_ev,
                'train_sharpe': train_sharpe,
                'test_sharpe': test_sharpe,
                'overfitting_ratio': overfit_ratio,
                'is_overfit': is_overfit
            })

        return pd.DataFrame(results)