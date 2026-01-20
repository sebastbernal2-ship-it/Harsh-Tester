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
    """Production-grade backtester with 10 test suites"""
    
    def __init__(self, data, init_cash=100000, fee=0.0003, allocation_method="equal_weight"):
        self.data = data.astype(float)
        self.init_cash = init_cash
        self.fee = fee
        self.allocation_method = allocation_method
        self.risk_free_rate = 0.02
        
        # ✅ INITIALIZE THESE FOR DEBUG OUTPUT
        self.best_stats = None
        self.best_monthly_stats = None
        self.best_dd_stats = None
        self.debug = True  # Enable debug output
    
    def backtest_signals(self, entries, exits, data=None):
        """Core backtesting logic"""
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
        
        positions = np.zeros(n_assets)
        equity_curve = []
        drawdowns = []
        running_max = self.init_cash

        
        for i in range(n_dates):
            prices = close[i, :]
            entry_signals = entries[i, :] if i < len(entries) else np.zeros(n_assets, dtype=bool)
            exit_signals = exits[i, :] if i < len(exits) else np.zeros(n_assets, dtype=bool)
            
            for j in range(n_assets):
                if entry_signals[j] and positions[j] == 0:
                    capital_per_asset = self.init_cash / n_assets
                    positions[j] = capital_per_asset / prices[j] * (1 - self.fee)
                elif exit_signals[j] and positions[j] > 0:
                    positions[j] = 0
            
            portfolio_value = sum(positions[j] * prices[j] for j in range(n_assets))
            equity_curve.append(portfolio_value)
            
            if portfolio_value > running_max:
                running_max = portfolio_value
            dd = (portfolio_value - running_max) / running_max if running_max > 0 else 0
            drawdowns.append(dd * 100)
        
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
        
        # ✅ FIX: Don't divide by 100 again - drawdowns are already in %
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
        
        # ✅ STORE BEST STATS FOR DEBUG OUTPUT
        self.best_stats = best_stats
        
        # ✅ DEBUG OUTPUT
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
        """Test 4: Parameter Sensitivity - FIXED"""
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
        
        # ✅ STORE BEST STATS FOR DEBUG OUTPUT
        self.best_monthly_stats = best_stats
        
        # ✅ DEBUG OUTPUT
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
            params = dict(zip(keys, best_combo))
            
            # ✅ STORE BEST STATS FOR DEBUG OUTPUT
            self.best_dd_stats = best_stats
            
            # ✅ DEBUG OUTPUT
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
