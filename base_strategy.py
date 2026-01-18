import pandas as pd
from typing import Tuple, Dict, Any

class BaseStrategy:
    def __init__(self, data: pd.DataFrame, params: Dict[str, Any]):
        self.data = data
        self.params = params

    def generate_signals(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return (entries, exits) aligned with self.data"""
        raise NotImplementedError
