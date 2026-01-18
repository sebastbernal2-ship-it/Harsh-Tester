import importlib.util
import pandas as pd
import numpy as np
from itertools import product

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

    def _backtest_signals(self, entries, exits):
        """
        Backtest multi-asset portfolio.
        entries/exits: DataFrames with shape (dates, assets)
        """
        close = self.data.values  # (dates, assets)
        entries = entries.values.astype(bool)
        exits = exits.values.astype(bool)
        
        n_dates = close.shape[0]
        positions = np.zeros(self.n_assets)  # shares held per asset
        equity_curve = []
        
        for i in range(n_dates):
            prices = close[i, :]  # (assets,)
            entry_signals = entries[i, :]
            exit_signals = exits[i, :]
            
            # Entry logic
            for j in range(self.n_assets):
                if entry_signals[j] and positions[j] == 0:
                    # Allocate capital per asset
                    if self.allocation_method == "equal_weight":
                        capital_per_asset = self.init_cash / self.n_assets
                    else:  # signal_weight
                        capital_per_asset = self.init_cash / self.n_assets
                    
                    positions[j] = (capital_per_asset / prices[j]) * (1 - self.fee)
            
            # Exit logic
            for j in range(self.n_assets):
                if exit_signals[j] and positions[j] > 0:
                    positions[j] = 0
            
            # Calculate portfolio value (mark-to-market)
            portfolio_value = sum(positions[j] * prices[j] for j in range(self.n_assets))
            equity_curve.append(portfolio_value)
        
        final_value = equity_curve[-1] if equity_curve else self.init_cash
        total_return = (final_value - self.init_cash) / self.init_cash
        
        # Sharpe ratio
        equity_series = np.array(equity_curve)
        equity_series = np.where(equity_series <= 0, 1e-6, equity_series)  # Avoid divide by zero
        daily_returns = np.diff(equity_series) / equity_series[:-1]
        daily_returns = np.where(np.isfinite(daily_returns), daily_returns, 0)  # Replace NaN/inf with 0

        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        running_max = np.maximum.accumulate(equity_series)
        drawdown = np.where(running_max > 0, (equity_series - running_max) / running_max, 0)
        max_dd = np.min(drawdown) if len(drawdown) > 0 else 0.0

        
        return {
            "sharpe": float(sharpe),
            "max_dd": float(max_dd),
            "total_return": float(total_return),
            "final_value": float(final_value)
        }

    def stress_test(self, strategy_code: str, param_grid: dict) -> pd.DataFrame:
        StrategyClass = self._load_strategy_class(strategy_code)
        keys = list(param_grid.keys())
        results = []

        for combo in product(*param_grid.values()):
            params = dict(zip(keys, combo))
            strat = StrategyClass(self.data, params)
            entries, exits = strat.generate_signals()
            
            # Ensure proper shape
            if isinstance(entries, pd.Series):
                entries = entries.to_frame()
            if isinstance(exits, pd.Series):
                exits = exits.to_frame()
            
            stats = self._backtest_signals(entries, exits)
            stats.update(params)
            results.append(stats)

        return pd.DataFrame(results)
