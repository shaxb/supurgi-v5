"""
Signal Filters - EMA, Volatility Regime, etc.

Filters for processing and validating signals before execution.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
from loguru import logger

from .datatypes import SignalIntent, SignalType


class BaseFilter(ABC):
    """Base class for all signal filters."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize filter with configuration.
        
        Args:
            config: Filter configuration parameters
        """
        self.config = config or {}
        self.name = self.__class__.__name__
    
    @abstractmethod
    def apply(self, signals: List[SignalIntent], data: pd.DataFrame) -> List[SignalIntent]:
        """
        Apply filter to list of signals.
        
        Args:
            signals: List of signals to filter
            data: Market data for context
            
        Returns:
            Filtered list of signals
        """
        pass
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration parameter."""
        return self.config.get(key, default)


class EMAFilter(BaseFilter):
    """
    Exponential Moving Average Filter.
    
    Filters signals based on EMA trend direction and position.
    Only allows signals that align with the EMA trend.
    """
    
    def apply(self, signals: List[SignalIntent], data: pd.DataFrame) -> List[SignalIntent]:
        """
        Filter signals based on EMA trend.
        
        Args:
            signals: Input signals
            data: OHLCV data
            
        Returns:
            Filtered signals that align with EMA trend
        """
        if not signals or data.empty:
            return signals
        
        period = self.get_config('period', 21)
        trend_strength_threshold = self.get_config('trend_strength', 0.01)
        
        try:
            # Calculate EMA
            ema = data['close'].ewm(span=period).mean()
            
            # Calculate trend strength (EMA slope)
            ema_slope = ema.pct_change(periods=5)
            
            filtered_signals = []
            
            for signal in signals:
                # Get EMA values at signal timestamp
                try:
                    signal_ema = ema.loc[signal.timestamp]
                    signal_slope = ema_slope.loc[signal.timestamp]
                    
                    # Check if signal aligns with EMA trend
                    if signal.signal_type == SignalType.BUY:
                        # For buy signals, require upward EMA trend and price above EMA
                        if (signal_slope > trend_strength_threshold and 
                            signal.price > signal_ema):
                            filtered_signals.append(signal)
                    
                    elif signal.signal_type == SignalType.SELL:
                        # For sell signals, require downward EMA trend and price below EMA
                        if (signal_slope < -trend_strength_threshold and 
                            signal.price < signal_ema):
                            filtered_signals.append(signal)
                    
                    else:
                        # Keep exit signals unchanged
                        filtered_signals.append(signal)
                
                except (KeyError, IndexError):
                    # If timestamp not found, skip signal
                    logger.debug(f"EMAFilter: Timestamp {signal.timestamp} not found in data")
                    continue
            
            filtered_count = len(signals) - len(filtered_signals)
            logger.debug(f"EMAFilter: Filtered out {filtered_count} signals")
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"EMAFilter error: {e}")
            return signals  # Return original signals on error


class VolatilityRegimeFilter(BaseFilter):
    """
    Volatility Regime Filter.
    
    Adjusts signal confidence based on current volatility regime.
    High volatility periods may require stronger signals.
    """
    
    def apply(self, signals: List[SignalIntent], data: pd.DataFrame) -> List[SignalIntent]:
        """
        Filter signals based on volatility regime.
        
        Args:
            signals: Input signals
            data: OHLCV data
            
        Returns:
            Signals with adjusted confidence based on volatility
        """
        if not signals or data.empty:
            return signals
        
        lookback_period = self.get_config('lookback_period', 20)
        high_vol_threshold = self.get_config('high_vol_threshold', 2.0)
        low_vol_threshold = self.get_config('low_vol_threshold', 0.5)
        high_vol_penalty = self.get_config('high_vol_penalty', 0.2)
        low_vol_penalty = self.get_config('low_vol_penalty', 0.1)
        
        try:
            # Calculate volatility (rolling standard deviation of returns)
            returns = data['close'].pct_change()
            volatility = returns.rolling(window=lookback_period).std()
            vol_mean = volatility.mean()
            
            filtered_signals = []
            
            for signal in signals:
                try:
                    # Get volatility at signal timestamp
                    signal_vol = volatility.loc[signal.timestamp]
                    vol_ratio = signal_vol / vol_mean
                    
                    # Create modified signal
                    modified_signal = signal
                    
                    # Adjust confidence based on volatility regime
                    if vol_ratio > high_vol_threshold:
                        # High volatility regime - reduce confidence
                        new_confidence = max(0.0, signal.confidence - high_vol_penalty)
                        modified_signal.confidence = new_confidence
                        modified_signal.notes += f" | High vol regime (ratio: {vol_ratio:.2f})"
                        
                    elif vol_ratio < low_vol_threshold:
                        # Low volatility regime - slightly reduce confidence (low volume breakouts)
                        new_confidence = max(0.0, signal.confidence - low_vol_penalty)
                        modified_signal.confidence = new_confidence
                        modified_signal.notes += f" | Low vol regime (ratio: {vol_ratio:.2f})"
                    
                    # Only keep signals above minimum confidence after adjustment
                    min_confidence = self.get_config('min_confidence_after_filter', 0.3)
                    if modified_signal.confidence >= min_confidence:
                        filtered_signals.append(modified_signal)
                
                except (KeyError, IndexError):
                    # If timestamp not found, keep original signal
                    filtered_signals.append(signal)
                    continue
            
            filtered_count = len(signals) - len(filtered_signals)
            logger.debug(f"VolatilityRegimeFilter: Filtered out {filtered_count} signals")
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"VolatilityRegimeFilter error: {e}")
            return signals


class VolumeFilter(BaseFilter):
    """
    Volume Filter.
    
    Filters signals based on volume confirmation.
    Requires above-average volume for signal validation.
    """
    
    def apply(self, signals: List[SignalIntent], data: pd.DataFrame) -> List[SignalIntent]:
        """
        Filter signals based on volume confirmation.
        
        Args:
            signals: Input signals
            data: OHLCV data
            
        Returns:
            Volume-confirmed signals
        """
        if not signals or data.empty or 'volume' not in data.columns:
            return signals
        
        volume_period = self.get_config('volume_period', 20)
        volume_threshold = self.get_config('volume_threshold', 1.5)
        
        try:
            # Calculate average volume
            avg_volume = data['volume'].rolling(window=volume_period).mean()
            
            filtered_signals = []
            
            for signal in signals:
                try:
                    # Get volume at signal timestamp
                    signal_volume = data.loc[signal.timestamp, 'volume']
                    signal_avg_volume = avg_volume.loc[signal.timestamp]
                    
                    volume_ratio = signal_volume / signal_avg_volume
                    
                    # Require above-average volume for confirmation
                    if volume_ratio >= volume_threshold:
                        # Add volume information to signal
                        signal.indicators['volume_ratio'] = volume_ratio
                        signal.notes += f" | Volume confirmed ({volume_ratio:.2f}x avg)"
                        filtered_signals.append(signal)
                
                except (KeyError, IndexError):
                    # If timestamp not found or no volume data, skip
                    continue
            
            filtered_count = len(signals) - len(filtered_signals)
            logger.debug(f"VolumeFilter: Filtered out {filtered_count} signals")
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"VolumeFilter error: {e}")
            return signals


class FilterChain:
    """
    Chain multiple filters together for sequential processing.
    """
    
    def __init__(self, filters: List[BaseFilter]):
        """
        Initialize filter chain.
        
        Args:
            filters: List of filters to apply in order
        """
        self.filters = filters
    
    def apply(self, signals: List[SignalIntent], data: pd.DataFrame) -> List[SignalIntent]:
        """
        Apply all filters in sequence.
        
        Args:
            signals: Input signals
            data: Market data
            
        Returns:
            Signals after all filters applied
        """
        current_signals = signals
        
        for filter_instance in self.filters:
            try:
                current_signals = filter_instance.apply(current_signals, data)
                logger.debug(f"After {filter_instance.name}: {len(current_signals)} signals remaining")
            except Exception as e:
                logger.error(f"Filter {filter_instance.name} failed: {e}")
                continue
        
        return current_signals
    
    def add_filter(self, filter_instance: BaseFilter) -> None:
        """Add a filter to the chain."""
        self.filters.append(filter_instance)
    
    def remove_filter(self, filter_class: type) -> bool:
        """Remove first filter of specified type."""
        for i, filter_instance in enumerate(self.filters):
            if isinstance(filter_instance, filter_class):
                del self.filters[i]
                return True
        return False


# Convenience function to create common filter chains
def create_default_filter_chain() -> FilterChain:
    """Create a default filter chain with common filters."""
    return FilterChain([
        EMAFilter({'period': 21, 'trend_strength': 0.01}),
        VolatilityRegimeFilter({
            'lookback_period': 20,
            'high_vol_threshold': 2.0,
            'high_vol_penalty': 0.2,
            'min_confidence_after_filter': 0.4
        }),
        VolumeFilter({'volume_period': 20, 'volume_threshold': 1.2})
    ])


def create_conservative_filter_chain() -> FilterChain:
    """Create a conservative filter chain with stricter requirements."""
    return FilterChain([
        EMAFilter({'period': 50, 'trend_strength': 0.02}),
        VolatilityRegimeFilter({
            'lookback_period': 30,
            'high_vol_threshold': 1.5,
            'high_vol_penalty': 0.3,
            'min_confidence_after_filter': 0.6
        }),
        VolumeFilter({'volume_period': 30, 'volume_threshold': 2.0})
    ])


def create_aggressive_filter_chain() -> FilterChain:
    """Create an aggressive filter chain with looser requirements."""
    return FilterChain([
        EMAFilter({'period': 10, 'trend_strength': 0.005}),
        VolatilityRegimeFilter({
            'lookback_period': 10,
            'high_vol_threshold': 3.0,
            'high_vol_penalty': 0.1,
            'min_confidence_after_filter': 0.2
        })
    ])
