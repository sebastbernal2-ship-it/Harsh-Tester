# harsh_tester.py
import importlib.util
import pandas as pd
import numpy as np
from itertools import product

class HarshTester:
    def __init__(self, data, init_cash=100000, fee=0.0003):
        self.data = data.astype(float)  # Convert to float immediately
        self.init_cash = init_cash
        self.fee = fee

    def _load_strategy_class(self, strategy_code: str):
        spec = importlib.util.spec_from_loader("strategy_module", loader=None)
        module = importlib.util.module_from_spec(spec)
        exec(strategy_code, module.__dict__)
        return module.Strategy

    def _backtest_signals(self, entries, exits):
        close = self.data.values  # Convert to numpy array
        entries = entries.values.astype(bool)  # numpy array
        exits = exits.values.astype(bool)  # numpy array
        
        position = 0.0
        cash = float(self.init_cash)
        equity_curve = []
        
        for i in range(len(close)):
            price = float(close[i])
            entry_signal = bool(entries[i])
            exit_signal = bool(exits[i])
            
            if entry_signal and position == 0:
                position = (cash / price) * (1 - self.fee)
                cash = 0.0
            elif exit_signal and position > 0:
                cash = position * price * (1 - self.fee)
                position = 0.0
            
            current_value = position * price if position > 0 else cash
            equity_curve.append(current_value)
        
        final_value = equity_curve[-1] if equity_curve else self.init_cash
        total_return = (final_value - self.init_cash) / self.init_cash
        
        equity_series = np.array(equity_curve)
        daily_returns = np.diff(equity_series) / equity_series[:-1]
        
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        running_max = np.maximum.accumulate(equity_series)
        drawdown = (equity_series - running_max) / running_max
        max_dd = np.min(drawdown)
        
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
            
            stats = self._backtest_signals(entries, exits)
            stats.update(params)
            results.append(stats)

        return pd.DataFrame(results)
