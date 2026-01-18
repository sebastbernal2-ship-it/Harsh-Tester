# harsh_tester.py
import importlib.util
import pandas as pd
import numpy as np
from itertools import product
from typing import Dict, Tuple, List

class HarshTester:
    def __init__(self, data, init_cash=100000, fee=0.0003, allocation_method="equal_weight"):
        """
        data: pd.DataFrame with shape (dates, assets)
        allocation_method: "equal_weight" or "signal_weight"
        """
        self.data = data.astype(float)
        self.init_cash = init_cash
        self.fee = fee
        self.allocation_method = allocation_method
        self.n_assets = data.shape[1]

    def _load_strategy_class(self, strategy_code: str):
        spec = importlib.util.spec_from_loader("strategy_module", loader=None)
        module = importlib.util.module_from_spec(spec)
        exec(strategy_code, module.__dict__)
        return module.Strategy

    def _backtest_signals(self, entries, exits, data=None):
        """
        Backtest multi-asset portfolio.
        entries/exits: DataFrames with shape (dates, assets)
        data: optional custom data slice (for walk-forward)
        """
        if data is None:
            close = self.data.values
        else:
            close = data.values
            
        entries = entries.values.astype(bool)
        exits = exits.values.astype(bool)
        
        n_dates = close.shape[0]
        n_assets = close.shape[1]
        positions = np.zeros(n_assets)
        equity_curve = []
        
        for i in range(n_dates):
            prices = close[i, :]
            entry_signals = entries[i, :] if i < len(entries) else np.zeros(n_assets, dtype=bool)
            exit_signals = exits[i, :] if i < len(exits) else np.zeros(n_assets, dtype=bool)
            
            # Entry logic
            for j in range(n_assets):
                if entry_signals[j] and positions[j] == 0:
                    capital_per_asset = self.init_cash / n_assets
                    positions[j] = (capital_per_asset / prices[j]) * (1 - self.fee)
            
            # Exit logic
            for j in range(n_assets):
                if exit_signals[j] and positions[j] > 0:
                    positions[j] = 0
            
            portfolio_value = sum(positions[j] * prices[j] for j in range(n_assets))
            equity_curve.append(portfolio_value)
        
        final_value = equity_curve[-1] if equity_curve else self.init_cash
        total_return = (final_value - self.init_cash) / self.init_cash
        
        # Sharpe ratio with safety checks
        equity_series = np.array(equity_curve)
        equity_series = np.where(equity_series <= 0, 1e-6, equity_series)
        
        daily_returns = np.diff(equity_series) / equity_series[:-1]
        daily_returns = np.where(np.isfinite(daily_returns), daily_returns, 0)
        
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Max drawdown with safety checks
        running_max = np.maximum.accumulate(equity_series)
        drawdown = np.where(running_max > 0, (equity_series - running_max) / running_max, 0)
        max_dd = np.min(drawdown) if len(drawdown) > 0 else 0.0
        
        return {
            "sharpe": float(sharpe),
            "max_dd": float(max_dd),
            "total_return": float(total_return),
            "final_value": float(final_value),
            "equity_curve": equity_curve
        }

    def stress_test(self, strategy_code: str, param_grid: dict) -> pd.DataFrame:
        """Grid search backtest"""
        StrategyClass = self._load_strategy_class(strategy_code)
        keys = list(param_grid.keys())
        results = []

        for combo in product(*param_grid.values()):
            params = dict(zip(keys, combo))
            strat = StrategyClass(self.data, params)
            entries, exits = strat.generate_signals()
            
            if isinstance(entries, pd.Series):
                entries = entries.to_frame()
            if isinstance(exits, pd.Series):
                exits = exits.to_frame()
            
            stats = self._backtest_signals(entries, exits)
            stats.update(params)
            results.append(stats)

        df = pd.DataFrame(results)
        return df.drop(columns=['equity_curve'], errors='ignore')

    def walk_forward_test(self, strategy_code: str, param_grid: dict, train_years: int = 3, test_years: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Walk-forward analysis: rolling train/test windows
        Returns: (results per window, overall stats)
        """
        StrategyClass = self._load_strategy_class(strategy_code)
        keys = list(param_grid.keys())
        
        dates = self.data.index
        total_days = len(dates)
        train_days = int(train_years * 252)
        test_days = int(test_years * 252)
        
        window_results = []
        
        for start_idx in range(0, total_days - train_days - test_days, test_days):
            train_end = start_idx + train_days
            test_end = min(train_end + test_days, total_days)
            
            train_data = self.data.iloc[start_idx:train_end]
            test_data = self.data.iloc[train_end:test_end]
            
            window_label = f"{dates[start_idx].date()} to {dates[test_end-1].date()}"
            
            best_sharpe = -np.inf
            best_params = None
            
            for combo in product(*param_grid.values()):
                params = dict(zip(keys, combo))
                strat = StrategyClass(train_data, params)
                entries, exits = strat.generate_signals()
                
                if isinstance(entries, pd.Series):
                    entries = entries.to_frame()
                if isinstance(exits, pd.Series):
                    exits = exits.to_frame()
                
                stats = self._backtest_signals(entries, exits, train_data)
                if stats['sharpe'] > best_sharpe:
                    best_sharpe = stats['sharpe']
                    best_params = params
            
            if best_params:
                strat = StrategyClass(test_data, best_params)
                entries, exits = strat.generate_signals()
                
                if isinstance(entries, pd.Series):
                    entries = entries.to_frame()
                if isinstance(exits, pd.Series):
                    exits = exits.to_frame()
                
                test_stats = self._backtest_signals(entries, exits, test_data)
                test_stats['window'] = window_label
                test_stats['train_sharpe'] = best_sharpe
                test_stats['params'] = str(best_params)
                window_results.append(test_stats)
        
        wf_df = pd.DataFrame(window_results)
        
        summary = {
            'avg_test_sharpe': wf_df['sharpe'].mean() if len(wf_df) > 0 else 0,
            'std_test_sharpe': wf_df['sharpe'].std() if len(wf_df) > 0 else 0,
            'avg_max_dd': wf_df['max_dd'].mean() if len(wf_df) > 0 else 0,
            'sharpe_degradation': (wf_df['train_sharpe'].mean() - wf_df['sharpe'].mean()) / (wf_df['train_sharpe'].mean() + 1e-6) if len(wf_df) > 0 else 0
        }
        
        return wf_df.drop(columns=['equity_curve'], errors='ignore'), pd.DataFrame([summary])

    def monte_carlo_test(self, strategy_code: str, param_grid: dict, n_sims: int = 1000) -> pd.DataFrame:
        """
        Monte Carlo: resample daily returns, compute tail risk
        Returns: distribution of Sharpe/Max DD across resamples
        """
        StrategyClass = self._load_strategy_class(strategy_code)
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
            
            base_stats = self._backtest_signals(entries, exits)
            base_results.append({
                **params,
                'base_sharpe': base_stats['sharpe'],
                'base_max_dd': base_stats['max_dd'],
                'equity_curve': base_stats['equity_curve']
            })
        
        mc_results = []
        for result in base_results:
            equity = np.array(result['equity_curve'], dtype=np.float64)
            
            # Calculate returns
            returns = np.diff(equity) / np.maximum(equity[:-1], 1e-6)
            returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
            
            mc_sharpes = []
            mc_dds = []
            
            for _ in range(n_sims):
                resampled_idx = np.random.choice(len(returns), size=len(returns), replace=True)
                resampled_returns = returns[resampled_idx]
                resampled_equity = np.cumprod(1 + resampled_returns) * self.init_cash
                
                ret_std = np.std(resampled_returns)
                if ret_std > 1e-8:
                    mc_sharpe = np.mean(resampled_returns) / ret_std * np.sqrt(252)
                else:
                    mc_sharpe = 0.0
                
                running_max = np.maximum.accumulate(resampled_equity)
                drawdown = (resampled_equity - running_max) / np.maximum(running_max, 1e-6)
                mc_dd = float(np.min(drawdown))
                
                mc_sharpes.append(float(mc_sharpe))
                mc_dds.append(mc_dd)
            
            result_dict = {
                **{k: result[k] for k in keys},
                'base_sharpe': float(result['base_sharpe']),
                'mc_sharpe_5pct': float(np.percentile(mc_sharpes, 5)),
                'mc_sharpe_mean': float(np.mean(mc_sharpes)),
                'mc_sharpe_95pct': float(np.percentile(mc_sharpes, 95)),
                'base_max_dd': float(result['base_max_dd']),
                'mc_dd_5pct': float(np.percentile(mc_dds, 5)),
                'mc_dd_mean': float(np.mean(mc_dds)),
                'mc_dd_95pct': float(np.percentile(mc_dds, 95)),
            }
            mc_results.append(result_dict)
        
        return pd.DataFrame(mc_results)

    def parameter_sensitivity_test(self, strategy_code: str, param_grid: dict, perturbation: float = 0.2) -> pd.DataFrame:
        """
        Parameter sensitivity: perturb each param ±20%, check stability
        Returns: metrics for base vs perturbed params
        """
        StrategyClass = self._load_strategy_class(strategy_code)
        keys = list(param_grid.keys())
        
        sensitivity_results = []
        
        for combo in product(*param_grid.values()):
            params = dict(zip(keys, combo))
            
            strat = StrategyClass(self.data, params)
            entries, exits = strat.generate_signals()
            
            if isinstance(entries, pd.Series):
                entries = entries.to_frame()
            if isinstance(exits, pd.Series):
                exits = exits.to_frame()
            
            base_stats = self._backtest_signals(entries, exits)
            
            for key in keys:
                if isinstance(param_grid[key][0], (int, float)):
                    for direction in [-1, 1]:
                        perturbed_params = params.copy()
                        perturbed_val = params[key] * (1 + direction * perturbation)
                        
                        if isinstance(param_grid[key][0], int):
                            perturbed_val = max(1, int(perturbed_val))
                        
                        perturbed_params[key] = perturbed_val
                        
                        strat_p = StrategyClass(self.data, perturbed_params)
                        entries_p, exits_p = strat_p.generate_signals()
                        
                        if isinstance(entries_p, pd.Series):
                            entries_p = entries_p.to_frame()
                        if isinstance(exits_p, pd.Series):
                            exits_p = exits_p.to_frame()
                        
                        perturbed_stats = self._backtest_signals(entries_p, exits_p)
                        
                        sensitivity_results.append({
                            **params,
                            'perturbed_param': key,
                            'perturbation': f"{direction*perturbation*100:+.0f}%",
                            'base_sharpe': base_stats['sharpe'],
                            'perturbed_sharpe': perturbed_stats['sharpe'],
                            'sharpe_change': (perturbed_stats['sharpe'] - base_stats['sharpe']) / (abs(base_stats['sharpe']) + 1e-6),
                            'base_max_dd': base_stats['max_dd'],
                            'perturbed_max_dd': perturbed_stats['max_dd'],
                            'dd_change': (perturbed_stats['max_dd'] - base_stats['max_dd']) / (abs(base_stats['max_dd']) + 1e-6)
                        })
        
        return pd.DataFrame(sensitivity_results)

    def stress_test_historical(self, strategy_code: str, param_grid: dict) -> pd.DataFrame:
        """
        Historical stress tests: replay known crises
        Crisis windows (approx):
        - 2008 crash: 2008-09 to 2009-03
        - 2020 COVID: 2020-02 to 2020-04
        - 2022 rates: 2022-01 to 2022-10
        """
        StrategyClass = self._load_strategy_class(strategy_code)
        keys = list(param_grid.keys())
        
        crises = {
            '2008_financial_crisis': ('2008-09-01', '2009-03-31'),
            '2020_covid_crash': ('2020-02-01', '2020-04-30'),
            '2022_rate_shock': ('2022-01-01', '2022-10-31'),
        }
        
        stress_results = []
        
        for crisis_name, (start, end) in crises.items():
            crisis_data = self.data.loc[start:end]
            
            if len(crisis_data) < 10:
                continue
            
            for combo in product(*param_grid.values()):
                params = dict(zip(keys, combo))
                strat = StrategyClass(crisis_data, params)
                entries, exits = strat.generate_signals()
                
                if isinstance(entries, pd.Series):
                    entries = entries.to_frame()
                if isinstance(exits, pd.Series):
                    exits = exits.to_frame()
                
                crisis_stats = self._backtest_signals(entries, exits, crisis_data)
                crisis_stats['crisis'] = crisis_name
                crisis_stats.update(params)
                stress_results.append(crisis_stats)
        
        return pd.DataFrame(stress_results).drop(columns=['equity_curve'], errors='ignore')
