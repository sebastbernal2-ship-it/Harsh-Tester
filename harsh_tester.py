import importlib.util
import pandas as pd
import vectorbt as vbt
from itertools import product

class HarshTester:
    def __init__(self, data, init_cash=100000, fee=0.0003):
        self.data = data
        self.init_cash = init_cash
        self.fee = fee

    def _load_strategy_class(self, strategy_code: str):
        """Load strategy from code string"""
        spec = importlib.util.spec_from_loader("strategy_module", loader=None)
        module = importlib.util.module_from_spec(spec)
        exec(strategy_code, module.__dict__)
        return module.Strategy

    def _run_from_signals(self, entries, exits):
        pf = vbt.Portfolio.from_signals(
            self.data, entries, exits,
            init_cash=self.init_cash, fees=self.fee, freq="D"
        )
        return pf

    def stress_test(self, strategy_code: str, param_grid: dict) -> pd.DataFrame:
        StrategyClass = self._load_strategy_class(strategy_code)
        keys = list(param_grid.keys())
        results = []

        for combo in product(*param_grid.values()):
            params = dict(zip(keys, combo))
            strat = StrategyClass(self.data, params)
            entries, exits = strat.generate_signals()
            pf = self._run_from_signals(entries, exits)

            stats = {
                **params,
                "sharpe": float(pf.sharpe_ratio()),
                "max_dd": float(pf.max_drawdown()),
                "total_return": float(pf.total_return())
            }
            results.append(stats)

        return pd.DataFrame(results)
