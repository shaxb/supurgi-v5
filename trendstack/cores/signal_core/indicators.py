"""
Technical Indicators Module

Comprehensive collection of technical analysis indicators
for signal generation and market analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from loguru import logger
import talib
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class TechnicalIndicators:
    """
    Core technical indicators for market analysis.
    
    Features:
    - Trend indicators (SMA, EMA, MACD, etc.)
    - Momentum indicators (RSI, Stochastic, etc.)
    - Volatility indicators (Bollinger Bands, ATR, etc.)
    - Volume indicators (OBV, VWAP, etc.)
    - Custom composite indicators
    """
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int, alpha: Optional[float] = None) -> pd.Series:
        """Exponential Moving Average"""
        if alpha is None:
            alpha = 2 / (period + 1)
        return data.ewm(alpha=alpha).mean()
    
    @staticmethod
    def wma(data: pd.Series, period: int) -> pd.Series:
        """Weighted Moving Average"""
        weights = np.arange(1, period + 1)
        return data.rolling(window=period).apply(
            lambda x: np.average(x, weights=weights), raw=True
        )
    
    @staticmethod
    def hull_ma(data: pd.Series, period: int) -> pd.Series:
        """Hull Moving Average"""
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        wma_half = TechnicalIndicators.wma(data, half_period)
        wma_full = TechnicalIndicators.wma(data, period)
        
        hull_raw = 2 * wma_half - wma_full
        return TechnicalIndicators.wma(hull_raw, sqrt_period)
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def stochastic(
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    @staticmethod
    def macd(
        data: pd.Series, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Moving Average Convergence Divergence"""
        ema_fast = TechnicalIndicators.ema(data, fast_period)
        ema_slow = TechnicalIndicators.ema(data, slow_period)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(
        data: pd.Series, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def atr(
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        period: int = 14
    ) -> pd.Series:
        """Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    @staticmethod
    def adx(
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        period: int = 14
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Average Directional Index"""
        tr = TechnicalIndicators.atr(high, low, close, 1)
        
        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # Smoothed values
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
        
        # ADX calculation
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def williams_r(
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        period: int = 14
    ) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        return -100 * (highest_high - close) / (highest_high - lowest_low)
    
    @staticmethod
    def cci(
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        period: int = 20
    ) -> pd.Series:
        """Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        
        return (typical_price - sma_tp) / (0.015 * mean_deviation)


class VolumeIndicators:
    """Volume-based technical indicators."""
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume"""
        price_change = close.diff()
        obv_values = []
        obv = 0
        
        for i, change in enumerate(price_change):
            if pd.isna(change):
                obv_values.append(obv)
            elif change > 0:
                obv += volume.iloc[i]
                obv_values.append(obv)
            elif change < 0:
                obv -= volume.iloc[i]
                obv_values.append(obv)
            else:
                obv_values.append(obv)
        
        return pd.Series(obv_values, index=close.index)
    
    @staticmethod
    def vwap(
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        volume: pd.Series
    ) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def ad_line(
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        volume: pd.Series
    ) -> pd.Series:
        """Accumulation/Distribution Line"""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)  # Handle division by zero
        ad_values = (clv * volume).cumsum()
        return ad_values
    
    @staticmethod
    def chaikin_mf(
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        volume: pd.Series, 
        period: int = 20
    ) -> pd.Series:
        """Chaikin Money Flow"""
        ad = VolumeIndicators.ad_line(high, low, close, volume)
        return ad.diff().rolling(window=period).sum() / volume.rolling(window=period).sum()


class VolatilityIndicators:
    """Volatility-based indicators."""
    
    @staticmethod
    def historical_volatility(
        data: pd.Series, 
        period: int = 20, 
        annualize: bool = True
    ) -> pd.Series:
        """Historical Volatility"""
        returns = data.pct_change()
        vol = returns.rolling(window=period).std()
        
        if annualize:
            vol *= np.sqrt(252)  # Annualize assuming 252 trading days
        
        return vol
    
    @staticmethod
    def keltner_channels(
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        period: int = 20, 
        multiplier: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Keltner Channels"""
        ema = TechnicalIndicators.ema(close, period)
        atr = TechnicalIndicators.atr(high, low, close, period)
        
        upper_channel = ema + (multiplier * atr)
        lower_channel = ema - (multiplier * atr)
        
        return upper_channel, ema, lower_channel
    
    @staticmethod
    def donchian_channels(
        high: pd.Series, 
        low: pd.Series, 
        period: int = 20
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Donchian Channels"""
        upper_channel = high.rolling(window=period).max()
        lower_channel = low.rolling(window=period).min()
        middle_channel = (upper_channel + lower_channel) / 2
        
        return upper_channel, middle_channel, lower_channel


class CustomIndicators:
    """Custom and composite indicators."""
    
    @staticmethod
    def trend_strength(
        data: pd.Series, 
        short_period: int = 10, 
        long_period: int = 50
    ) -> pd.Series:
        """Custom trend strength indicator"""
        short_ma = TechnicalIndicators.ema(data, short_period)
        long_ma = TechnicalIndicators.ema(data, long_period)
        
        # Trend strength based on MA separation and slope
        ma_separation = (short_ma - long_ma) / long_ma * 100
        
        # Add slope component
        short_slope = short_ma.diff(5) / short_ma * 100
        long_slope = long_ma.diff(5) / long_ma * 100
        
        # Combine components
        trend_strength = ma_separation + (short_slope + long_slope) / 2
        
        return trend_strength
    
    @staticmethod
    def momentum_composite(
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        volume: pd.Series
    ) -> pd.Series:
        """Composite momentum indicator"""
        # Combine multiple momentum indicators
        rsi = TechnicalIndicators.rsi(close, 14)
        rsi_normalized = (rsi - 50) / 50  # Normalize RSI around 0
        
        williams = TechnicalIndicators.williams_r(high, low, close, 14)
        williams_normalized = (williams + 50) / 50  # Normalize Williams %R
        
        # Price momentum
        price_momentum = close.pct_change(10) * 100
        
        # Volume momentum
        volume_ma = volume.rolling(window=20).mean()
        volume_momentum = (volume / volume_ma - 1) * 100
        
        # Weight and combine
        composite = (
            0.3 * rsi_normalized + 
            0.3 * williams_normalized + 
            0.3 * price_momentum + 
            0.1 * volume_momentum
        )
        
        return composite
    
    @staticmethod
    def volatility_regime(
        data: pd.Series, 
        short_period: int = 10, 
        long_period: int = 50
    ) -> pd.Series:
        """Volatility regime indicator"""
        short_vol = VolatilityIndicators.historical_volatility(data, short_period, False)
        long_vol = VolatilityIndicators.historical_volatility(data, long_period, False)
        
        # Volatility ratio
        vol_ratio = short_vol / long_vol
        
        # Regime classification
        # > 1.5: High volatility regime
        # 0.7-1.5: Normal volatility regime  
        # < 0.7: Low volatility regime
        
        return vol_ratio
    
    @staticmethod
    def market_regime(
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        volume: pd.Series
    ) -> pd.Series:
        """Market regime indicator combining trend and volatility"""
        # Trend component
        trend = CustomIndicators.trend_strength(close, 20, 50)
        
        # Volatility component
        vol_regime = CustomIndicators.volatility_regime(close, 10, 30)
        
        # Volume component
        volume_ma = volume.rolling(window=20).mean()
        volume_ratio = volume / volume_ma
        
        # Combine into regime score
        # Positive: Trending market
        # Negative: Ranging/choppy market
        regime_score = trend * vol_regime * np.log(volume_ratio)
        
        return regime_score
