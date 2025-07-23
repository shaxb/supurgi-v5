"""
Signal Generation Module

Core signal generation classes for different types of trading signals.
Supports technical analysis, pattern-based, and ML-based signal generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
from loguru import logger
from datetime import datetime, timedelta
import yaml

from .indicators import TechnicalIndicators, VolumeIndicators, VolatilityIndicators, CustomIndicators
from .patterns import PatternDetector, CandlestickPatterns, ChartPatterns


class SignalGenerator(ABC):
    """
    Abstract base class for all signal generators.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.name = self.__class__.__name__
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from input data."""
        pass
    
    @abstractmethod
    def get_signal_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate signal strength/confidence."""
        pass


class TechnicalSignalGenerator(SignalGenerator):
    """
    Generate signals based on technical indicators.
    
    Features:
    - Multiple indicator combinations
    - Configurable parameters
    - Signal strength calculation
    - Trend and momentum analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        # Default configuration
        self.default_config = {
            'indicators': {
                'sma': {'periods': [20, 50], 'weight': 0.2},
                'ema': {'periods': [12, 26], 'weight': 0.2},
                'rsi': {'period': 14, 'overbought': 70, 'oversold': 30, 'weight': 0.15},
                'macd': {'fast': 12, 'slow': 26, 'signal': 9, 'weight': 0.2},
                'bollinger': {'period': 20, 'std': 2, 'weight': 0.15},
                'stochastic': {'k_period': 14, 'd_period': 3, 'weight': 0.1}
            },
            'signal_threshold': 0.6,
            'min_strength': 0.3
        }
        
        # Merge with provided config
        if self.config:
            self._merge_config(self.default_config, self.config)
        else:
            self.config = self.default_config
            
        logger.info(f"TechnicalSignalGenerator initialized with {len(self.config['indicators'])} indicators")
    
    def _merge_config(self, default: Dict, custom: Dict):
        """Recursively merge configuration dictionaries."""
        for key, value in custom.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate technical analysis signals.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with signal columns
        """
        signals = pd.DataFrame(index=data.index)
        signals['timestamp'] = data.index
        signals['signal'] = 0  # -1: sell, 0: hold, 1: buy
        signals['strength'] = 0.0
        signals['confidence'] = 0.0
        
        # Generate individual indicator signals
        indicator_signals = {}
        
        # Moving Average signals
        if 'sma' in self.config['indicators']:
            sma_config = self.config['indicators']['sma']
            sma_signal = self._generate_ma_signals(data['Close'], 'sma', sma_config)
            indicator_signals['sma'] = sma_signal
        
        if 'ema' in self.config['indicators']:
            ema_config = self.config['indicators']['ema']
            ema_signal = self._generate_ma_signals(data['Close'], 'ema', ema_config)
            indicator_signals['ema'] = ema_signal
        
        # RSI signals
        if 'rsi' in self.config['indicators']:
            rsi_config = self.config['indicators']['rsi']
            rsi_signal = self._generate_rsi_signals(data['Close'], rsi_config)
            indicator_signals['rsi'] = rsi_signal
        
        # MACD signals
        if 'macd' in self.config['indicators']:
            macd_config = self.config['indicators']['macd']
            macd_signal = self._generate_macd_signals(data['Close'], macd_config)
            indicator_signals['macd'] = macd_signal
        
        # Bollinger Bands signals
        if 'bollinger' in self.config['indicators']:
            bb_config = self.config['indicators']['bollinger']
            bb_signal = self._generate_bollinger_signals(data['Close'], bb_config)
            indicator_signals['bollinger'] = bb_signal
        
        # Stochastic signals
        if 'stochastic' in self.config['indicators']:
            stoch_config = self.config['indicators']['stochastic']
            stoch_signal = self._generate_stochastic_signals(
                data['High'], data['Low'], data['Close'], stoch_config
            )
            indicator_signals['stochastic'] = stoch_signal
        
        # Combine signals using weighted average
        signals = self._combine_signals(signals, indicator_signals)
        
        return signals
    
    def _generate_ma_signals(self, price: pd.Series, ma_type: str, config: Dict) -> pd.Series:
        """Generate moving average crossover signals."""
        periods = config['periods']
        
        if ma_type == 'sma':
            ma_short = TechnicalIndicators.sma(price, periods[0])
            ma_long = TechnicalIndicators.sma(price, periods[1])
        elif ma_type == 'ema':
            ma_short = TechnicalIndicators.ema(price, periods[0])
            ma_long = TechnicalIndicators.ema(price, periods[1])
        else:
            raise ValueError(f"Unknown MA type: {ma_type}")
        
        # Generate signals based on crossovers
        signal = pd.Series(0, index=price.index)
        
        # Bullish: short MA crosses above long MA
        bullish_cross = (ma_short > ma_long) & (ma_short.shift(1) <= ma_long.shift(1))
        # Bearish: short MA crosses below long MA
        bearish_cross = (ma_short < ma_long) & (ma_short.shift(1) >= ma_long.shift(1))
        
        signal[bullish_cross] = 1
        signal[bearish_cross] = -1
        
        return signal
    
    def _generate_rsi_signals(self, price: pd.Series, config: Dict) -> pd.Series:
        """Generate RSI-based signals."""
        period = config['period']
        overbought = config['overbought']
        oversold = config['oversold']
        
        rsi = TechnicalIndicators.rsi(price, period)
        signal = pd.Series(0, index=price.index)
        
        # RSI signals
        signal[rsi < oversold] = 1  # Oversold -> Buy
        signal[rsi > overbought] = -1  # Overbought -> Sell
        
        return signal
    
    def _generate_macd_signals(self, price: pd.Series, config: Dict) -> pd.Series:
        """Generate MACD signals."""
        fast = config['fast']
        slow = config['slow']
        signal_period = config['signal']
        
        macd_line, signal_line, histogram = TechnicalIndicators.macd(
            price, fast, slow, signal_period
        )
        
        signal = pd.Series(0, index=price.index)
        
        # MACD line crosses above signal line
        bullish_cross = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        # MACD line crosses below signal line
        bearish_cross = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        
        signal[bullish_cross] = 1
        signal[bearish_cross] = -1
        
        return signal
    
    def _generate_bollinger_signals(self, price: pd.Series, config: Dict) -> pd.Series:
        """Generate Bollinger Bands signals."""
        period = config['period']
        std_dev = config['std']
        
        upper, middle, lower = TechnicalIndicators.bollinger_bands(price, period, std_dev)
        signal = pd.Series(0, index=price.index)
        
        # Price touches lower band -> Buy
        signal[price <= lower] = 1
        # Price touches upper band -> Sell
        signal[price >= upper] = -1
        
        return signal
    
    def _generate_stochastic_signals(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        config: Dict
    ) -> pd.Series:
        """Generate Stochastic oscillator signals."""
        k_period = config['k_period']
        d_period = config['d_period']
        
        k_percent, d_percent = TechnicalIndicators.stochastic(
            high, low, close, k_period, d_period
        )
        
        signal = pd.Series(0, index=close.index)
        
        # Stochastic signals
        signal[(k_percent < 20) & (k_percent > d_percent)] = 1  # Oversold and %K crosses above %D
        signal[(k_percent > 80) & (k_percent < d_percent)] = -1  # Overbought and %K crosses below %D
        
        return signal
    
    def _combine_signals(self, base_signals: pd.DataFrame, indicator_signals: Dict) -> pd.DataFrame:
        """Combine individual indicator signals using weighted average."""
        total_weight = 0
        weighted_sum = pd.Series(0, index=base_signals.index)
        
        for indicator, signal_series in indicator_signals.items():
            if indicator in self.config['indicators']:
                weight = self.config['indicators'][indicator]['weight']
                weighted_sum += signal_series * weight
                total_weight += weight
        
        if total_weight > 0:
            composite_signal = weighted_sum / total_weight
            
            # Convert to discrete signals
            signal_threshold = self.config['signal_threshold']
            base_signals['signal'] = 0
            base_signals['signal'][composite_signal > signal_threshold] = 1
            base_signals['signal'][composite_signal < -signal_threshold] = -1
            
            # Calculate strength and confidence
            base_signals['strength'] = np.abs(composite_signal)
            base_signals['confidence'] = np.minimum(base_signals['strength'] / signal_threshold, 1.0)
        
        return base_signals
    
    def get_signal_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate signal strength for the latest data point."""
        signals_df = self.generate_signals(data)
        return signals_df['strength']


class PatternSignalGenerator(SignalGenerator):
    """
    Generate signals based on candlestick and chart patterns.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        self.default_config = {
            'patterns': {
                'candlestick_weight': 0.4,
                'chart_weight': 0.6,
                'min_pattern_strength': 0.5
            },
            'signal_threshold': 0.6
        }
        
        if self.config:
            self._merge_config(self.default_config, self.config)
        else:
            self.config = self.default_config
        
        self.pattern_detector = PatternDetector()
        
        logger.info("PatternSignalGenerator initialized")
    
    def _merge_config(self, default: Dict, custom: Dict):
        """Recursively merge configuration dictionaries."""
        for key, value in custom.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate pattern-based signals."""
        signals = pd.DataFrame(index=data.index)
        signals['timestamp'] = data.index
        signals['signal'] = 0
        signals['strength'] = 0.0
        signals['confidence'] = 0.0
        
        # Detect all patterns
        patterns = self.pattern_detector.detect_all_patterns(data)
        
        # Calculate pattern strength
        pattern_strength = self.pattern_detector.pattern_strength_score(patterns)
        
        # Generate signals based on patterns
        signal_threshold = self.config['signal_threshold']
        min_strength = self.config['patterns']['min_pattern_strength']
        
        # Bullish patterns
        bullish_mask = (pattern_strength > min_strength) & (pattern_strength > 0)
        signals.loc[bullish_mask, 'signal'] = 1
        
        # Bearish patterns  
        bearish_mask = (pattern_strength < -min_strength) & (pattern_strength < 0)
        signals.loc[bearish_mask, 'signal'] = -1
        
        # Set strength and confidence
        signals['strength'] = np.abs(pattern_strength)
        signals['confidence'] = np.minimum(signals['strength'] / signal_threshold, 1.0)
        
        return signals
    
    def get_signal_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate pattern signal strength."""
        signals_df = self.generate_signals(data)
        return signals_df['strength']


class MLSignalGenerator(SignalGenerator):
    """
    Machine Learning-based signal generation.
    
    Note: This is a placeholder for ML implementation.
    Would typically use scikit-learn, tensorflow, or similar.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        self.default_config = {
            'model_type': 'random_forest',
            'features': ['rsi', 'macd', 'bollinger_position', 'volume_ratio'],
            'lookback_period': 20,
            'signal_threshold': 0.6
        }
        
        if self.config:
            self.config = {**self.default_config, **self.config}
        else:
            self.config = self.default_config
        
        self.model = None  # Placeholder for ML model
        
        logger.info("MLSignalGenerator initialized (placeholder)")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ML-based signals.
        
        Note: This is a simplified implementation.
        In production, this would use trained ML models.
        """
        signals = pd.DataFrame(index=data.index)
        signals['timestamp'] = data.index
        signals['signal'] = 0
        signals['strength'] = 0.0
        signals['confidence'] = 0.0
        
        # For demonstration, generate random signals
        # In practice, this would use trained models
        np.random.seed(42)  # For reproducibility
        
        # Generate features (simplified)
        features = self._generate_features(data)
        
        # Placeholder ML prediction (random with some logic)
        predictions = np.random.randn(len(data)) * 0.3
        
        # Add some trend-following bias
        sma_20 = TechnicalIndicators.sma(data['Close'], 20)
        sma_50 = TechnicalIndicators.sma(data['Close'], 50)
        trend_bias = np.where(sma_20 > sma_50, 0.2, -0.2)
        
        predictions += trend_bias
        
        # Convert to signals
        threshold = self.config['signal_threshold']
        signals['signal'] = 0
        signals['signal'][predictions > threshold] = 1
        signals['signal'][predictions < -threshold] = -1
        
        signals['strength'] = np.abs(predictions)
        signals['confidence'] = np.minimum(signals['strength'] / threshold, 1.0)
        
        return signals
    
    def _generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features for ML model."""
        features = pd.DataFrame(index=data.index)
        
        # Technical indicators as features
        features['rsi'] = TechnicalIndicators.rsi(data['Close'], 14)
        
        macd_line, signal_line, _ = TechnicalIndicators.macd(data['Close'])
        features['macd'] = macd_line - signal_line
        
        upper, middle, lower = TechnicalIndicators.bollinger_bands(data['Close'])
        features['bollinger_position'] = (data['Close'] - lower) / (upper - lower)
        
        if 'Volume' in data.columns:
            volume_ma = data['Volume'].rolling(window=20).mean()
            features['volume_ratio'] = data['Volume'] / volume_ma
        else:
            features['volume_ratio'] = 1.0
        
        return features.fillna(0)
    
    def get_signal_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate ML signal strength."""
        signals_df = self.generate_signals(data)
        return signals_df['strength']


class CompositeSignalGenerator(SignalGenerator):
    """
    Combines multiple signal generators for robust signal generation.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        self.default_config = {
            'generators': {
                'technical': {'weight': 0.4, 'enabled': True},
                'pattern': {'weight': 0.3, 'enabled': True},
                'ml': {'weight': 0.3, 'enabled': False}  # Disabled by default
            },
            'consensus_threshold': 0.6,
            'min_generators': 2
        }
        
        if self.config:
            self.config = {**self.default_config, **self.config}
        else:
            self.config = self.default_config
        
        # Initialize sub-generators
        self.generators = {}
        
        if self.config['generators']['technical']['enabled']:
            self.generators['technical'] = TechnicalSignalGenerator()
        
        if self.config['generators']['pattern']['enabled']:
            self.generators['pattern'] = PatternSignalGenerator()
        
        if self.config['generators']['ml']['enabled']:
            self.generators['ml'] = MLSignalGenerator()
        
        logger.info(f"CompositeSignalGenerator initialized with {len(self.generators)} generators")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate composite signals from multiple generators."""
        base_signals = pd.DataFrame(index=data.index)
        base_signals['timestamp'] = data.index
        base_signals['signal'] = 0
        base_signals['strength'] = 0.0
        base_signals['confidence'] = 0.0
        
        # Collect signals from all generators
        generator_signals = {}
        
        for name, generator in self.generators.items():
            try:
                signals = generator.generate_signals(data)
                generator_signals[name] = signals
                logger.debug(f"Generated signals from {name} generator")
            except Exception as e:
                logger.error(f"Error generating signals from {name}: {e}")
                continue
        
        if len(generator_signals) < self.config['min_generators']:
            logger.warning(f"Only {len(generator_signals)} generators available, minimum is {self.config['min_generators']}")
            return base_signals
        
        # Combine signals using weighted consensus
        total_weight = 0
        weighted_signal = pd.Series(0.0, index=data.index)
        weighted_strength = pd.Series(0.0, index=data.index)
        
        for name, signals_df in generator_signals.items():
            if name in self.config['generators']:
                weight = self.config['generators'][name]['weight']
                
                # Weight the signals
                weighted_signal += signals_df['signal'] * weight
                weighted_strength += signals_df['strength'] * weight
                
                total_weight += weight
        
        if total_weight > 0:
            # Normalize
            consensus_signal = weighted_signal / total_weight
            consensus_strength = weighted_strength / total_weight
            
            # Apply consensus threshold
            threshold = self.config['consensus_threshold']
            
            base_signals['signal'] = 0
            base_signals['signal'][consensus_signal > threshold] = 1
            base_signals['signal'][consensus_signal < -threshold] = -1
            
            base_signals['strength'] = consensus_strength
            base_signals['confidence'] = np.minimum(consensus_strength / threshold, 1.0)
        
        return base_signals
    
    def get_signal_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate composite signal strength."""
        signals_df = self.generate_signals(data)
        return signals_df['strength']
    
    def get_generator_breakdown(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Get signals from individual generators for analysis."""
        breakdown = {}
        
        for name, generator in self.generators.items():
            try:
                signals = generator.generate_signals(data)
                breakdown[name] = signals
            except Exception as e:
                logger.error(f"Error getting breakdown from {name}: {e}")
        
        return breakdown
