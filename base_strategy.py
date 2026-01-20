import pandas as pd
import numpy as np
from typing import Tuple


class BaseStrategy:
    """
    Base class for all trading strategies.
    
    All strategies must:
    1. Inherit from this class
    2. Implement the generate_signals() method
    3. Return (entries, exits) as boolean Series/DataFrames
    
    Example:
        from harsh_tester import BaseStrategy
        
        class MyStrategy(BaseStrategy):
            def generate_signals(self):
                entries = ...  # Boolean Series
                exits = ...    # Boolean Series
                return entries, exits
    """
    
    def __init__(self, data: pd.DataFrame, params: dict = None):
        """
        Initialize strategy with price data and parameters.
        
        Args:
            data: DataFrame or Series of prices, shape (dates, assets)
            params: Dictionary of parameters from optimization grid
            
        Example:
            strategy = MyStrategy(data, {"lookback": 20, "threshold": 1.5})
        """
        self.data = data
        self.params = params or {}
    
    def generate_signals(self) -> Tuple[pd.Series, pd.Series]:
        """
        Generate trading signals.
        
        Returns:
            entries: Boolean Series/DataFrame, True = go long
            exits: Boolean Series/DataFrame, True = exit position
            
        Must be implemented by subclasses.
        
        Example:
            def generate_signals(self):
                lookback = self.params.get("lookback", 20)
                entries = (self.data > self.data.rolling(lookback).mean()).shift(1)
                exits = (self.data < self.data.rolling(lookback).mean()).shift(1)
                return entries, exits
        """
        raise NotImplementedError(
            "Subclass must implement generate_signals() method"
        )


# ===== EXAMPLE STRATEGY: MOMENTUM =====
class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy using z-score of price changes.
    
    Parameters:
        lookback: Lookback period for z-score calculation
        z_entry: Z-score threshold for entry
        z_exit: Z-score threshold for exit
    """
    
    def generate_signals(self) -> Tuple[pd.Series, pd.Series]:
        lookback = self.params.get("lookback", 20)
        z_entry = self.params.get("z_entry", 1.0)
        z_exit = self.params.get("z_exit", 0.5)
        
        rolling_mean = self.data.rolling(lookback).mean()
        rolling_std = self.data.rolling(lookback).std()
        z_score = (self.data - rolling_mean) / (rolling_std + 1e-6)
        
        entries = (z_score < -z_entry).shift(1).fillna(False).astype(bool)
        exits = (z_score > -z_exit).shift(1).fillna(False).astype(bool)
        
        return entries, exits


# ===== EXAMPLE STRATEGY: MEAN REVERSION =====
class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy using simple moving average crossover.
    
    Parameters:
        fast: Fast MA period
        slow: Slow MA period
    """
    
    def generate_signals(self) -> Tuple[pd.Series, pd.Series]:
        fast = self.params.get("fast", 10)
        slow = self.params.get("slow", 30)
        
        fast_ma = self.data.rolling(fast).mean()
        slow_ma = self.data.rolling(slow).mean()
        
        entries = (fast_ma > slow_ma).shift(1).fillna(False).astype(bool)
        exits = (fast_ma < slow_ma).shift(1).fillna(False).astype(bool)
        
        return entries, exits


# ===== EXAMPLE STRATEGY: VOLATILITY =====
class VolatilityStrategy(BaseStrategy):
    """
    Volatility-based strategy.
    Trade when volatility is below threshold.
    
    Parameters:
        vol_lookback: Lookback for volatility calculation
        vol_threshold: Volatility threshold (as multiplier of mean)
    """
    
    def generate_signals(self) -> Tuple[pd.Series, pd.Series]:
        vol_lookback = self.params.get("vol_lookback", 20)
        vol_multiplier = self.params.get("vol_multiplier", 0.8)
        
        returns = self.data.pct_change()
        rolling_vol = returns.rolling(vol_lookback).std()
        mean_vol = rolling_vol.mean()
        
        vol_threshold = mean_vol * vol_multiplier
        
        entries = (rolling_vol < vol_threshold).shift(1).fillna(False).astype(bool)
        exits = (rolling_vol > mean_vol).shift(1).fillna(False).astype(bool)
        
        return entries, exits


# ===== TIPS FOR WRITING YOUR OWN STRATEGY =====
"""
1. ACCESS DATA:
   self.data                           → All prices
   self.data.iloc[:, 0]               → First asset
   self.data['SPY']                   → Named column (if available)
   
2. ACCESS PARAMETERS:
   self.params.get("lookback", 20)   → Get with default
   self.params["lookback"]            → Get without default
   
3. GENERATE SIGNALS:
   entries = boolean_series           → True = enter long
   exits = boolean_series             → True = exit position
   
4. RETURN FORMAT:
   Must return exactly (entries, exits)
   Both must be boolean Series/DataFrames aligned with self.data
   
5. BEST PRACTICES:
   • Use .shift(1) to avoid look-ahead bias
   • Use .fillna(False) to handle NaN at start
   • Use .astype(bool) to ensure boolean type
   • Use .rolling() for moving averages/volatility
   • Add 1e-6 to divisors to prevent divide-by-zero
   
6. PARAMETER NAMES:
   Keep simple: "lookback", "threshold", "fast", "slow", etc.
   Avoid special characters: use underscores, not hyphens

7. TESTING YOUR STRATEGY:
   Copy-paste this template into the app's "Strategy Code" section
   Define your parameters in the "Parameter Grid" section
   Click "Run Complete Harsh Test Suite"
   Check results immediately!
"""

# Example of accessing multi-asset data:
class PairsStrategy(BaseStrategy):
    """
    Pairs trading strategy using spread z-score.
    Works with 2+ assets.
    """
    
    def generate_signals(self) -> Tuple[pd.Series, pd.Series]:
        lookback = self.params.get("lookback", 60)
        z_threshold = self.params.get("z_threshold", 2.0)
        
        # Assume 2 assets: asset1 and asset2
        if isinstance(self.data, pd.DataFrame) and self.data.shape[1] >= 2:
            asset1 = self.data.iloc[:, 0]
            asset2 = self.data.iloc[:, 1]
        else:
            # Fallback to single asset
            asset1 = self.data
            asset2 = self.data
        
        # Calculate spread and z-score
        spread = asset1 - asset2
        spread_mean = spread.rolling(lookback).mean()
        spread_std = spread.rolling(lookback).std()
        z_score = (spread - spread_mean) / (spread_std + 1e-6)
        
        # Entry when spread is extended
        entries = (z_score > z_threshold).shift(1).fillna(False).astype(bool)
        
        # Exit when spread reverts to mean
        exits = (z_score < 0).shift(1).fillna(False).astype(bool)
        
        return entries, exits