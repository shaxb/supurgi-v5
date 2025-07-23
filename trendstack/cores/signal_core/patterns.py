"""
Pattern Detection Module

Advanced pattern recognition for candlestick patterns,
chart patterns, and volume patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from loguru import logger
from scipy import signal
from scipy.stats import linregress
import warnings

warnings.filterwarnings('ignore')


class CandlestickPatterns:
    """
    Candlestick pattern detection and analysis.
    
    Features:
    - Single candlestick patterns
    - Multi-candlestick patterns
    - Pattern strength scoring
    - Bullish/Bearish classification
    """
    
    @staticmethod
    def doji(
        open_price: pd.Series, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series,
        threshold: float = 0.1
    ) -> pd.Series:
        """
        Detect Doji patterns.
        threshold: Maximum body size as percentage of range
        """
        body_size = np.abs(close - open_price)
        range_size = high - low
        
        # Avoid division by zero
        range_size = range_size.replace(0, np.nan)
        
        body_pct = body_size / range_size
        is_doji = body_pct <= threshold
        
        return is_doji.astype(int)
    
    @staticmethod
    def hammer(
        open_price: pd.Series, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series,
        body_ratio: float = 0.3,
        shadow_ratio: float = 2.0
    ) -> pd.Series:
        """
        Detect Hammer patterns.
        Bullish reversal pattern with small body and long lower shadow.
        """
        body_size = np.abs(close - open_price)
        total_range = high - low
        lower_shadow = np.minimum(open_price, close) - low
        upper_shadow = high - np.maximum(open_price, close)
        
        # Avoid division by zero
        total_range = total_range.replace(0, np.nan)
        body_size = body_size.replace(0, 0.001)  # Small value to avoid issues
        
        conditions = [
            body_size / total_range <= body_ratio,  # Small body
            lower_shadow / body_size >= shadow_ratio,  # Long lower shadow
            upper_shadow / body_size <= 0.5,  # Short upper shadow
            lower_shadow > 0  # Must have lower shadow
        ]
        
        is_hammer = np.all(conditions, axis=0)
        return pd.Series(is_hammer.astype(int), index=open_price.index)
    
    @staticmethod
    def shooting_star(
        open_price: pd.Series, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series,
        body_ratio: float = 0.3,
        shadow_ratio: float = 2.0
    ) -> pd.Series:
        """
        Detect Shooting Star patterns.
        Bearish reversal pattern with small body and long upper shadow.
        """
        body_size = np.abs(close - open_price)
        total_range = high - low
        lower_shadow = np.minimum(open_price, close) - low
        upper_shadow = high - np.maximum(open_price, close)
        
        # Avoid division by zero
        total_range = total_range.replace(0, np.nan)
        body_size = body_size.replace(0, 0.001)
        
        conditions = [
            body_size / total_range <= body_ratio,  # Small body
            upper_shadow / body_size >= shadow_ratio,  # Long upper shadow
            lower_shadow / body_size <= 0.5,  # Short lower shadow
            upper_shadow > 0  # Must have upper shadow
        ]
        
        is_shooting_star = np.all(conditions, axis=0)
        return pd.Series(is_shooting_star.astype(int), index=open_price.index)
    
    @staticmethod
    def engulfing_pattern(
        open_price: pd.Series, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Detect Bullish and Bearish Engulfing patterns.
        Returns: (bullish_engulfing, bearish_engulfing)
        """
        # Current and previous candle info
        prev_open = open_price.shift(1)
        prev_close = close.shift(1)
        
        # Body definitions
        curr_body_top = np.maximum(open_price, close)
        curr_body_bottom = np.minimum(open_price, close)
        prev_body_top = np.maximum(prev_open, prev_close)
        prev_body_bottom = np.minimum(prev_open, prev_close)
        
        # Bullish engulfing: white body engulfs previous black body
        bullish_conditions = [
            prev_close < prev_open,  # Previous candle was bearish
            close > open_price,  # Current candle is bullish
            curr_body_bottom < prev_body_bottom,  # Engulfs from below
            curr_body_top > prev_body_top  # Engulfs from above
        ]
        
        # Bearish engulfing: black body engulfs previous white body
        bearish_conditions = [
            prev_close > prev_open,  # Previous candle was bullish
            close < open_price,  # Current candle is bearish
            curr_body_top > prev_body_top,  # Engulfs from above
            curr_body_bottom < prev_body_bottom  # Engulfs from below
        ]
        
        bullish_engulfing = np.all(bullish_conditions, axis=0)
        bearish_engulfing = np.all(bearish_conditions, axis=0)
        
        return (
            pd.Series(bullish_engulfing.astype(int), index=open_price.index),
            pd.Series(bearish_engulfing.astype(int), index=open_price.index)
        )
    
    @staticmethod
    def morning_star(
        open_price: pd.Series, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series,
        gap_threshold: float = 0.5
    ) -> pd.Series:
        """
        Detect Morning Star pattern (3-candle bullish reversal).
        """
        # Need at least 3 candles
        if len(open_price) < 3:
            return pd.Series(0, index=open_price.index)
        
        # Define candle positions (t-2, t-1, t)
        open_2 = open_price.shift(2)
        high_2 = high.shift(2)
        low_2 = low.shift(2)
        close_2 = close.shift(2)
        
        open_1 = open_price.shift(1)
        high_1 = high.shift(1)
        low_1 = low.shift(1)
        close_1 = close.shift(1)
        
        # Current candle (t=0)
        open_0 = open_price
        high_0 = high
        low_0 = low
        close_0 = close
        
        # Pattern conditions
        conditions = [
            # First candle: Large bearish candle
            close_2 < open_2,
            (open_2 - close_2) / (high_2 - low_2) > 0.6,
            
            # Second candle: Small body (star), gaps down
            np.abs(close_1 - open_1) / (high_1 - low_1) < 0.3,
            high_1 < close_2,  # Gaps down
            
            # Third candle: Bullish, closes above midpoint of first candle
            close_0 > open_0,
            close_0 > (open_2 + close_2) / 2,
            open_0 > close_1  # Gaps up from star
        ]
        
        is_morning_star = np.all(conditions, axis=0)
        return pd.Series(is_morning_star.astype(int), index=open_price.index)
    
    @staticmethod
    def evening_star(
        open_price: pd.Series, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series,
        gap_threshold: float = 0.5
    ) -> pd.Series:
        """
        Detect Evening Star pattern (3-candle bearish reversal).
        """
        # Need at least 3 candles
        if len(open_price) < 3:
            return pd.Series(0, index=open_price.index)
        
        # Define candle positions
        open_2 = open_price.shift(2)
        high_2 = high.shift(2)
        low_2 = low.shift(2)
        close_2 = close.shift(2)
        
        open_1 = open_price.shift(1)
        high_1 = high.shift(1)
        low_1 = low.shift(1)
        close_1 = close.shift(1)
        
        open_0 = open_price
        high_0 = high
        low_0 = low
        close_0 = close
        
        # Pattern conditions
        conditions = [
            # First candle: Large bullish candle
            close_2 > open_2,
            (close_2 - open_2) / (high_2 - low_2) > 0.6,
            
            # Second candle: Small body (star), gaps up
            np.abs(close_1 - open_1) / (high_1 - low_1) < 0.3,
            low_1 > close_2,  # Gaps up
            
            # Third candle: Bearish, closes below midpoint of first candle
            close_0 < open_0,
            close_0 < (open_2 + close_2) / 2,
            open_0 < close_1  # Gaps down from star
        ]
        
        is_evening_star = np.all(conditions, axis=0)
        return pd.Series(is_evening_star.astype(int), index=open_price.index)


class ChartPatterns:
    """
    Chart pattern detection using price action analysis.
    """
    
    @staticmethod
    def find_peaks_valleys(
        data: pd.Series, 
        window: int = 5, 
        min_distance: int = 3
    ) -> Tuple[List[int], List[int]]:
        """
        Find peaks and valleys in price data.
        """
        # Use scipy.signal to find peaks
        peaks, _ = signal.find_peaks(data.values, distance=min_distance)
        valleys, _ = signal.find_peaks(-data.values, distance=min_distance)
        
        return peaks.tolist(), valleys.tolist()
    
    @staticmethod
    def support_resistance_levels(
        data: pd.Series, 
        window: int = 20, 
        min_touches: int = 3,
        tolerance: float = 0.02
    ) -> Dict[str, List[float]]:
        """
        Identify support and resistance levels.
        """
        peaks, valleys = ChartPatterns.find_peaks_valleys(data, window)
        
        # Get peak and valley values
        peak_values = [data.iloc[i] for i in peaks]
        valley_values = [data.iloc[i] for i in valleys]
        
        # Find clustered levels
        def find_clusters(values, tolerance):
            if not values:
                return []
            
            values = sorted(values)
            clusters = []
            current_cluster = [values[0]]
            
            for value in values[1:]:
                if abs(value - current_cluster[-1]) / current_cluster[-1] <= tolerance:
                    current_cluster.append(value)
                else:
                    if len(current_cluster) >= min_touches:
                        clusters.append(np.mean(current_cluster))
                    current_cluster = [value]
            
            # Don't forget the last cluster
            if len(current_cluster) >= min_touches:
                clusters.append(np.mean(current_cluster))
            
            return clusters
        
        resistance_levels = find_clusters(peak_values, tolerance)
        support_levels = find_clusters(valley_values, tolerance)
        
        return {
            'support': support_levels,
            'resistance': resistance_levels
        }
    
    @staticmethod
    def trend_line_detection(
        data: pd.Series, 
        window: int = 20, 
        min_points: int = 3
    ) -> Dict[str, List[Dict]]:
        """
        Detect trend lines (support and resistance).
        """
        peaks, valleys = ChartPatterns.find_peaks_valleys(data, window // 2)
        
        def find_trend_lines(indices, is_resistance=True):
            trend_lines = []
            
            if len(indices) < min_points:
                return trend_lines
            
            # Try different combinations of points
            for i in range(len(indices) - min_points + 1):
                for j in range(i + min_points - 1, len(indices)):
                    # Get points for trend line
                    x_points = indices[i:j+1]
                    y_points = [data.iloc[idx] for idx in x_points]
                    
                    # Fit trend line
                    if len(x_points) >= 2:
                        slope, intercept, r_value, _, _ = linregress(x_points, y_points)
                        
                        # Check if trend line is valid
                        if abs(r_value) > 0.8:  # Good correlation
                            trend_lines.append({
                                'slope': slope,
                                'intercept': intercept,
                                'r_squared': r_value ** 2,
                                'start_idx': x_points[0],
                                'end_idx': x_points[-1],
                                'points': len(x_points)
                            })
            
            return trend_lines
        
        resistance_lines = find_trend_lines(peaks, is_resistance=True)
        support_lines = find_trend_lines(valleys, is_resistance=False)
        
        return {
            'resistance_lines': resistance_lines,
            'support_lines': support_lines
        }
    
    @staticmethod
    def triangle_pattern(
        data: pd.Series, 
        window: int = 50, 
        convergence_threshold: float = 0.1
    ) -> Dict[str, any]:
        """
        Detect triangle patterns (ascending, descending, symmetrical).
        """
        if len(data) < window:
            return {'pattern': None}
        
        # Get recent data
        recent_data = data.tail(window)
        
        # Find trend lines
        trend_lines = ChartPatterns.trend_line_detection(recent_data, window // 3)
        
        resistance_lines = trend_lines['resistance_lines']
        support_lines = trend_lines['support_lines']
        
        if not resistance_lines or not support_lines:
            return {'pattern': None}
        
        # Get the best trend lines (highest R²)
        best_resistance = max(resistance_lines, key=lambda x: x['r_squared'])
        best_support = max(support_lines, key=lambda x: x['r_squared'])
        
        res_slope = best_resistance['slope']
        sup_slope = best_support['slope']
        
        # Classify triangle pattern
        if abs(res_slope - sup_slope) < convergence_threshold:
            if res_slope < 0 and sup_slope > 0:
                pattern_type = 'symmetrical'
            elif res_slope < 0 and sup_slope >= 0:
                pattern_type = 'descending'
            elif res_slope >= 0 and sup_slope > 0:
                pattern_type = 'ascending'
            else:
                pattern_type = 'unknown'
        else:
            pattern_type = None
        
        return {
            'pattern': pattern_type,
            'resistance_line': best_resistance,
            'support_line': best_support,
            'convergence': abs(res_slope - sup_slope)
        }
    
    @staticmethod
    def double_top_bottom(
        data: pd.Series, 
        window: int = 50, 
        tolerance: float = 0.02
    ) -> Dict[str, any]:
        """
        Detect double top and double bottom patterns.
        """
        peaks, valleys = ChartPatterns.find_peaks_valleys(data, window // 5)
        
        def find_double_pattern(indices, values, is_top=True):
            if len(indices) < 2:
                return None
            
            # Look for two similar peaks/valleys
            for i in range(len(indices) - 1):
                for j in range(i + 1, len(indices)):
                    val1 = data.iloc[indices[i]]
                    val2 = data.iloc[indices[j]]
                    
                    # Check if values are similar
                    if abs(val1 - val2) / max(val1, val2) <= tolerance:
                        # Check if there's a significant valley/peak between them
                        between_indices = [idx for idx in (valleys if is_top else peaks) 
                                         if indices[i] < idx < indices[j]]
                        
                        if between_indices:
                            between_val = data.iloc[between_indices[0]]
                            if is_top:
                                # For double top, valley should be significantly lower
                                if (min(val1, val2) - between_val) / min(val1, val2) > 0.05:
                                    return {
                                        'first_peak': indices[i],
                                        'second_peak': indices[j],
                                        'valley': between_indices[0],
                                        'peak_values': [val1, val2],
                                        'valley_value': between_val
                                    }
                            else:
                                # For double bottom, peak should be significantly higher
                                if (between_val - max(val1, val2)) / max(val1, val2) > 0.05:
                                    return {
                                        'first_bottom': indices[i],
                                        'second_bottom': indices[j],
                                        'peak': between_indices[0],
                                        'bottom_values': [val1, val2],
                                        'peak_value': between_val
                                    }
            return None
        
        double_top = find_double_pattern(peaks, [data.iloc[i] for i in peaks], is_top=True)
        double_bottom = find_double_pattern(valleys, [data.iloc[i] for i in valleys], is_top=False)
        
        return {
            'double_top': double_top,
            'double_bottom': double_bottom
        }


class VolumePatterns:
    """
    Volume-based pattern detection.
    """
    
    @staticmethod
    def volume_spike(
        volume: pd.Series, 
        threshold: float = 2.0, 
        window: int = 20
    ) -> pd.Series:
        """
        Detect volume spikes above average.
        """
        volume_ma = volume.rolling(window=window).mean()
        volume_std = volume.rolling(window=window).std()
        
        # Volume spike when volume > mean + threshold * std
        spike_threshold = volume_ma + threshold * volume_std
        
        return (volume > spike_threshold).astype(int)
    
    @staticmethod
    def volume_breakout(
        close: pd.Series, 
        volume: pd.Series, 
        price_threshold: float = 0.02, 
        volume_threshold: float = 1.5,
        window: int = 20
    ) -> pd.Series:
        """
        Detect breakouts confirmed by volume.
        """
        # Price breakout
        high_20 = close.rolling(window=window).max()
        price_breakout = (close > high_20.shift(1) * (1 + price_threshold))
        
        # Volume confirmation
        volume_ma = volume.rolling(window=window).mean()
        volume_confirmation = (volume > volume_ma * volume_threshold)
        
        # Confirmed breakout
        confirmed_breakout = price_breakout & volume_confirmation
        
        return confirmed_breakout.astype(int)
    
    @staticmethod
    def accumulation_distribution(
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        volume: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Detect accumulation/distribution patterns.
        """
        # Calculate A/D line
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        ad_line = (clv * volume).cumsum()
        
        # Detect accumulation/distribution
        ad_ma = ad_line.rolling(window=window).mean()
        ad_slope = ad_line.diff(window)
        
        # Positive slope = accumulation, negative = distribution
        return np.sign(ad_slope)


class PatternDetector:
    """
    Main pattern detection class combining all pattern types.
    """
    
    def __init__(self):
        self.candlestick = CandlestickPatterns()
        self.chart = ChartPatterns()
        self.volume = VolumePatterns()
    
    def detect_all_patterns(
        self, 
        ohlcv_data: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """
        Detect all patterns in OHLCV data.
        
        Args:
            ohlcv_data: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
            
        Returns:
            Dictionary of pattern detection results
        """
        results = {}
        
        # Extract OHLCV
        open_price = ohlcv_data['Open']
        high = ohlcv_data['High']
        low = ohlcv_data['Low']
        close = ohlcv_data['Close']
        volume = ohlcv_data['Volume'] if 'Volume' in ohlcv_data.columns else None
        
        # Candlestick patterns
        results['doji'] = self.candlestick.doji(open_price, high, low, close)
        results['hammer'] = self.candlestick.hammer(open_price, high, low, close)
        results['shooting_star'] = self.candlestick.shooting_star(open_price, high, low, close)
        
        bullish_eng, bearish_eng = self.candlestick.engulfing_pattern(open_price, high, low, close)
        results['bullish_engulfing'] = bullish_eng
        results['bearish_engulfing'] = bearish_eng
        
        results['morning_star'] = self.candlestick.morning_star(open_price, high, low, close)
        results['evening_star'] = self.candlestick.evening_star(open_price, high, low, close)
        
        # Volume patterns (if volume data available)
        if volume is not None:
            results['volume_spike'] = self.volume.volume_spike(volume)
            results['volume_breakout'] = self.volume.volume_breakout(close, volume)
            results['accumulation_distribution'] = self.volume.accumulation_distribution(
                high, low, close, volume
            )
        
        logger.info(f"Detected {len(results)} pattern types")
        return results
    
    def pattern_strength_score(
        self, 
        patterns: Dict[str, pd.Series], 
        weights: Optional[Dict[str, float]] = None
    ) -> pd.Series:
        """
        Calculate composite pattern strength score.
        """
        if weights is None:
            # Default weights
            weights = {
                'doji': 0.1,
                'hammer': 0.15,
                'shooting_star': 0.15,
                'bullish_engulfing': 0.2,
                'bearish_engulfing': 0.2,
                'morning_star': 0.25,
                'evening_star': 0.25,
                'volume_spike': 0.1,
                'volume_breakout': 0.2
            }
        
        composite_score = pd.Series(0, index=patterns[list(patterns.keys())[0]].index)
        
        for pattern_name, pattern_series in patterns.items():
            if pattern_name in weights:
                weight = weights[pattern_name]
                composite_score += pattern_series * weight
        
        return composite_score
