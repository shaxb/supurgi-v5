"""
Signal Processing Module

Advanced signal processing, filtering, and validation for trading signals.
Handles signal aggregation, filtering, and quality assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from loguru import logger
from datetime import datetime, timedelta
import warnings
from scipy import stats
from collections import deque

warnings.filterwarnings('ignore')


class SignalProcessor:
    """
    Core signal processing functionality.
    
    Features:
    - Signal smoothing and filtering
    - Noise reduction
    - Signal transformation
    - Quality metrics calculation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.default_params = {
            'smoothing_window': 3,
            'noise_threshold': 0.1,
            'min_signal_duration': 2,
            'max_signal_gap': 5
        }
        
        # Merge configurations
        for key, value in self.default_params.items():
            if key not in self.config:
                self.config[key] = value
        
        logger.info("SignalProcessor initialized")
    
    def smooth_signals(
        self, 
        signals: pd.Series, 
        method: str = 'moving_average',
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Smooth signals to reduce noise.
        
        Args:
            signals: Signal series to smooth
            method: Smoothing method ('moving_average', 'exponential', 'gaussian')
            window: Smoothing window size
            
        Returns:
            Smoothed signal series
        """
        if window is None:
            window = self.config['smoothing_window']
        
        if method == 'moving_average':
            smoothed = signals.rolling(window=window, center=True).mean()
        elif method == 'exponential':
            alpha = 2.0 / (window + 1)
            smoothed = signals.ewm(alpha=alpha).mean()
        elif method == 'gaussian':
            # Simple Gaussian-like smoothing using weighted moving average
            weights = np.exp(-0.5 * ((np.arange(window) - window//2) / (window/4))**2)
            weights /= weights.sum()
            smoothed = signals.rolling(window=window, center=True).apply(
                lambda x: np.average(x, weights=weights), raw=True
            )
        else:
            logger.warning(f"Unknown smoothing method: {method}, using moving average")
            smoothed = signals.rolling(window=window, center=True).mean()
        
        # Fill NaN values
        smoothed = smoothed.fillna(method='bfill').fillna(method='ffill')
        
        return smoothed
    
    def remove_noise(
        self, 
        signals: pd.Series, 
        threshold: Optional[float] = None
    ) -> pd.Series:
        """
        Remove noise from signals based on amplitude threshold.
        
        Args:
            signals: Input signals
            threshold: Noise threshold (signals below this are set to 0)
            
        Returns:
            Denoised signals
        """
        if threshold is None:
            threshold = self.config['noise_threshold']
        
        # Remove signals below threshold
        denoised = signals.copy()
        denoised[np.abs(denoised) < threshold] = 0
        
        return denoised
    
    def consolidate_signals(
        self, 
        signals: pd.Series, 
        min_duration: Optional[int] = None,
        max_gap: Optional[int] = None
    ) -> pd.Series:
        """
        Consolidate signals by removing short-duration signals and filling small gaps.
        
        Args:
            signals: Input signal series
            min_duration: Minimum signal duration to keep
            max_gap: Maximum gap to fill between signals
            
        Returns:
            Consolidated signals
        """
        if min_duration is None:
            min_duration = self.config['min_signal_duration']
        if max_gap is None:
            max_gap = self.config['max_signal_gap']
        
        consolidated = signals.copy()
        
        # Remove short-duration signals
        signal_changes = (consolidated != consolidated.shift(1)).astype(int)
        signal_groups = signal_changes.cumsum()
        
        for group_id in signal_groups.unique():
            group_mask = signal_groups == group_id
            group_length = group_mask.sum()
            group_value = consolidated[group_mask].iloc[0]
            
            # Remove short signals
            if group_value != 0 and group_length < min_duration:
                consolidated[group_mask] = 0
        
        # Fill small gaps
        for signal_value in [-1, 1]:
            signal_mask = consolidated == signal_value
            
            # Find gaps in signals
            signal_diff = signal_mask.astype(int).diff()
            signal_ends = consolidated.index[signal_diff == -1]
            signal_starts = consolidated.index[signal_diff == 1]
            
            # Fill gaps between signals of the same type
            for i, end_time in enumerate(signal_ends):
                if i < len(signal_starts):
                    start_time = signal_starts[i]
                    gap_size = (consolidated.index.get_loc(start_time) - 
                              consolidated.index.get_loc(end_time) - 1)
                    
                    if 0 < gap_size <= max_gap:
                        gap_mask = ((consolidated.index > end_time) & 
                                  (consolidated.index < start_time))
                        consolidated[gap_mask] = signal_value
        
        return consolidated
    
    def calculate_signal_quality(self, signals: pd.Series) -> Dict[str, float]:
        """
        Calculate signal quality metrics.
        
        Args:
            signals: Signal series to analyze
            
        Returns:
            Dictionary of quality metrics
        """
        non_zero_signals = signals[signals != 0]
        
        metrics = {
            'signal_ratio': len(non_zero_signals) / len(signals) if len(signals) > 0 else 0,
            'mean_signal_strength': non_zero_signals.abs().mean() if len(non_zero_signals) > 0 else 0,
            'signal_stability': 1 - (signals.diff().abs().mean() / 2) if len(signals) > 1 else 0,
            'signal_consistency': non_zero_signals.std() if len(non_zero_signals) > 1 else 0,
            'positive_ratio': (signals > 0).sum() / len(non_zero_signals) if len(non_zero_signals) > 0 else 0
        }
        
        # Signal change frequency
        signal_changes = (signals != signals.shift(1)).sum()
        metrics['change_frequency'] = signal_changes / len(signals) if len(signals) > 0 else 0
        
        return metrics


class SignalFilter:
    """
    Advanced signal filtering with multiple filter types.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.filters = []
        
        logger.info("SignalFilter initialized")
    
    def add_filter(self, filter_func: Callable, **kwargs):
        """Add a custom filter function."""
        self.filters.append((filter_func, kwargs))
    
    def market_hours_filter(
        self, 
        signals: pd.DataFrame, 
        market_open: str = "09:30", 
        market_close: str = "16:00"
    ) -> pd.DataFrame:
        """
        Filter signals to only generate during market hours.
        
        Args:
            signals: Signal DataFrame with datetime index
            market_open: Market opening time (HH:MM format)
            market_close: Market closing time (HH:MM format)
            
        Returns:
            Filtered signals
        """
        filtered_signals = signals.copy()
        
        # Extract time from index
        if hasattr(signals.index, 'time'):
            signal_times = signals.index.time
            
            open_time = pd.to_datetime(market_open).time()
            close_time = pd.to_datetime(market_close).time()
            
            # Create mask for market hours
            market_hours_mask = (signal_times >= open_time) & (signal_times <= close_time)
            
            # Set signals outside market hours to 0
            filtered_signals.loc[~market_hours_mask, 'signal'] = 0
        
        return filtered_signals
    
    def volatility_filter(
        self, 
        signals: pd.DataFrame, 
        price_data: pd.Series, 
        min_volatility: float = 0.01,
        max_volatility: float = 0.1,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Filter signals based on market volatility conditions.
        
        Args:
            signals: Signal DataFrame
            price_data: Price series for volatility calculation
            min_volatility: Minimum volatility threshold
            max_volatility: Maximum volatility threshold
            window: Volatility calculation window
            
        Returns:
            Volatility-filtered signals
        """
        filtered_signals = signals.copy()
        
        # Calculate rolling volatility
        returns = price_data.pct_change()
        volatility = returns.rolling(window=window).std()
        
        # Create volatility mask
        vol_mask = (volatility >= min_volatility) & (volatility <= max_volatility)
        
        # Apply filter
        filtered_signals.loc[~vol_mask, 'signal'] = 0
        
        return filtered_signals
    
    def volume_filter(
        self, 
        signals: pd.DataFrame, 
        volume_data: pd.Series, 
        min_volume_ratio: float = 0.5,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Filter signals based on volume conditions.
        
        Args:
            signals: Signal DataFrame
            volume_data: Volume series
            min_volume_ratio: Minimum volume ratio to average
            window: Volume average calculation window
            
        Returns:
            Volume-filtered signals
        """
        filtered_signals = signals.copy()
        
        # Calculate average volume
        avg_volume = volume_data.rolling(window=window).mean()
        volume_ratio = volume_data / avg_volume
        
        # Create volume mask
        volume_mask = volume_ratio >= min_volume_ratio
        
        # Apply filter
        filtered_signals.loc[~volume_mask, 'signal'] = 0
        
        return filtered_signals
    
    def apply_all_filters(
        self, 
        signals: pd.DataFrame, 
        **filter_data
    ) -> pd.DataFrame:
        """
        Apply all registered filters to signals.
        
        Args:
            signals: Input signals
            **filter_data: Additional data needed for filters
            
        Returns:
            Filtered signals
        """
        filtered_signals = signals.copy()
        
        for filter_func, kwargs in self.filters:
            try:
                filtered_signals = filter_func(filtered_signals, **kwargs, **filter_data)
                logger.debug(f"Applied filter: {filter_func.__name__}")
            except Exception as e:
                logger.error(f"Error applying filter {filter_func.__name__}: {e}")
        
        return filtered_signals


class SignalAggregator:
    """
    Aggregate signals from multiple sources or timeframes.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.signal_sources = {}
        
        logger.info("SignalAggregator initialized")
    
    def add_signal_source(
        self, 
        name: str, 
        signals: pd.DataFrame, 
        weight: float = 1.0
    ):
        """Add a signal source with given weight."""
        self.signal_sources[name] = {
            'signals': signals,
            'weight': weight
        }
        
        logger.debug(f"Added signal source: {name} with weight {weight}")
    
    def weighted_aggregation(
        self, 
        method: str = 'average'
    ) -> pd.DataFrame:
        """
        Aggregate signals using weighted method.
        
        Args:
            method: Aggregation method ('average', 'majority', 'consensus')
            
        Returns:
            Aggregated signals
        """
        if not self.signal_sources:
            logger.warning("No signal sources available for aggregation")
            return pd.DataFrame()
        
        # Get common index
        all_indices = [source['signals'].index for source in self.signal_sources.values()]
        common_index = all_indices[0]
        for idx in all_indices[1:]:
            common_index = common_index.intersection(idx)
        
        if len(common_index) == 0:
            logger.warning("No common timestamps between signal sources")
            return pd.DataFrame()
        
        # Initialize result DataFrame
        result = pd.DataFrame(index=common_index)
        result['signal'] = 0
        result['strength'] = 0.0
        result['confidence'] = 0.0
        
        if method == 'average':
            total_weight = sum(source['weight'] for source in self.signal_sources.values())
            
            for name, source in self.signal_sources.items():
                signals = source['signals'].reindex(common_index).fillna(0)
                weight = source['weight'] / total_weight
                
                result['signal'] += signals['signal'] * weight
                result['strength'] += signals.get('strength', 0) * weight
                result['confidence'] += signals.get('confidence', 0) * weight
            
            # Convert to discrete signals
            result['signal'] = np.sign(result['signal'])
            
        elif method == 'majority':
            # Majority vote
            for timestamp in common_index:
                votes = []
                weights = []
                
                for name, source in self.signal_sources.items():
                    if timestamp in source['signals'].index:
                        signal_val = source['signals'].loc[timestamp, 'signal']
                        votes.append(signal_val)
                        weights.append(source['weight'])
                
                if votes:
                    # Weighted majority vote
                    vote_counts = {-1: 0, 0: 0, 1: 0}
                    for vote, weight in zip(votes, weights):
                        vote_counts[vote] += weight
                    
                    result.loc[timestamp, 'signal'] = max(vote_counts, key=vote_counts.get)
        
        elif method == 'consensus':
            # Require consensus (all non-zero signals agree)
            for timestamp in common_index:
                signals_at_time = []
                
                for name, source in self.signal_sources.items():
                    if timestamp in source['signals'].index:
                        signal_val = source['signals'].loc[timestamp, 'signal']
                        if signal_val != 0:
                            signals_at_time.append(signal_val)
                
                if signals_at_time:
                    # Check if all agree
                    if len(set(signals_at_time)) == 1:
                        result.loc[timestamp, 'signal'] = signals_at_time[0]
                    else:
                        result.loc[timestamp, 'signal'] = 0  # No consensus
        
        return result
    
    def time_series_aggregation(
        self, 
        timeframe_map: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Aggregate signals across different timeframes.
        
        Args:
            timeframe_map: Mapping of source names to timeframes
            
        Returns:
            Time-aggregated signals
        """
        # This would implement logic to aggregate signals from different timeframes
        # For now, return simple weighted aggregation
        return self.weighted_aggregation('average')


class SignalValidator:
    """
    Validate signal quality and consistency.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.validation_rules = []
        
        logger.info("SignalValidator initialized")
    
    def add_validation_rule(self, rule_func: Callable, **kwargs):
        """Add a validation rule."""
        self.validation_rules.append((rule_func, kwargs))
    
    def validate_signal_consistency(
        self, 
        signals: pd.DataFrame, 
        max_changes_per_period: int = 10,
        period_minutes: int = 60
    ) -> Dict[str, bool]:
        """
        Validate that signals don't change too frequently.
        
        Args:
            signals: Signal DataFrame
            max_changes_per_period: Maximum allowed signal changes per period
            period_minutes: Period length in minutes
            
        Returns:
            Validation results
        """
        results = {'consistency_check': True, 'details': {}}
        
        if len(signals) == 0:
            return results
        
        # Calculate signal changes
        signal_changes = (signals['signal'] != signals['signal'].shift(1)).sum()
        
        # Calculate time period
        if hasattr(signals.index, 'total_seconds'):
            total_hours = (signals.index[-1] - signals.index[0]).total_seconds() / 3600
            periods = total_hours * (60 / period_minutes)
        else:
            periods = len(signals) / (period_minutes / 5)  # Assume 5-minute bars
        
        if periods > 0:
            changes_per_period = signal_changes / periods
            results['details']['changes_per_period'] = changes_per_period
            results['consistency_check'] = changes_per_period <= max_changes_per_period
        
        return results
    
    def validate_signal_strength(
        self, 
        signals: pd.DataFrame, 
        min_strength: float = 0.3,
        min_confidence: float = 0.5
    ) -> Dict[str, bool]:
        """
        Validate signal strength and confidence levels.
        
        Args:
            signals: Signal DataFrame
            min_strength: Minimum required signal strength
            min_confidence: Minimum required confidence
            
        Returns:
            Validation results
        """
        results = {'strength_check': True, 'confidence_check': True, 'details': {}}
        
        if 'strength' in signals.columns:
            non_zero_signals = signals[signals['signal'] != 0]
            if len(non_zero_signals) > 0:
                avg_strength = non_zero_signals['strength'].mean()
                results['details']['average_strength'] = avg_strength
                results['strength_check'] = avg_strength >= min_strength
        
        if 'confidence' in signals.columns:
            non_zero_signals = signals[signals['signal'] != 0]
            if len(non_zero_signals) > 0:
                avg_confidence = non_zero_signals['confidence'].mean()
                results['details']['average_confidence'] = avg_confidence
                results['confidence_check'] = avg_confidence >= min_confidence
        
        return results
    
    def validate_all(self, signals: pd.DataFrame) -> Dict[str, any]:
        """
        Run all validation rules on signals.
        
        Args:
            signals: Signal DataFrame to validate
            
        Returns:
            Complete validation results
        """
        validation_results = {
            'overall_valid': True,
            'checks': {}
        }
        
        # Run built-in validations
        consistency_results = self.validate_signal_consistency(signals)
        strength_results = self.validate_signal_strength(signals)
        
        validation_results['checks']['consistency'] = consistency_results
        validation_results['checks']['strength'] = strength_results
        
        # Run custom validation rules
        for rule_func, kwargs in self.validation_rules:
            try:
                rule_name = rule_func.__name__
                rule_results = rule_func(signals, **kwargs)
                validation_results['checks'][rule_name] = rule_results
                
                # Update overall validity
                if isinstance(rule_results, dict):
                    for key, value in rule_results.items():
                        if isinstance(value, bool) and not value:
                            validation_results['overall_valid'] = False
                
                logger.debug(f"Applied validation rule: {rule_name}")
            except Exception as e:
                logger.error(f"Error in validation rule {rule_func.__name__}: {e}")
        
        # Check overall validity
        for check_name, check_results in validation_results['checks'].items():
            if isinstance(check_results, dict):
                for key, value in check_results.items():
                    if isinstance(value, bool) and not value:
                        validation_results['overall_valid'] = False
                        break
        
        return validation_results
