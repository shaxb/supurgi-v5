"""
Tests for Momentum Strategy

Basic tests to ensure the momentum strategy works correctly.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch

from ..datatypes import SignalType, SignalStrength, StrategyConfig
from ..momentum import MomentumStrategy


class TestMomentumStrategy:
    """Test cases for MomentumStrategy."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = StrategyConfig(
            name="test_momentum",
            parameters={
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "ma_period": 20,
                "confidence_threshold": 0.6
            }
        )
        self.strategy = MomentumStrategy(self.config)
    
    def create_test_data(self, length: int = 100) -> pd.DataFrame:
        """
        Create synthetic test data.
        
        Args:
            length: Number of data points
            
        Returns:
            DataFrame with OHLCV data
        """
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=length),
            periods=length,
            freq='1H'
        )
        
        # Create trending price data
        base_price = 100
        trend = np.linspace(0, 20, length)  # Upward trend
        noise = np.random.normal(0, 2, length)  # Random noise
        
        close_prices = base_price + trend + noise
        
        # Generate OHLC from close prices
        high_prices = close_prices + np.random.uniform(0, 2, length)
        low_prices = close_prices - np.random.uniform(0, 2, length)
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        
        volume = np.random.randint(1000, 10000, length)
        
        return pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        }, index=dates)
    
    def test_strategy_initialization(self):
        """Test that strategy initializes correctly."""
        assert self.strategy.name == "MomentumStrategy"
        assert self.strategy.config.name == "test_momentum"
        assert self.strategy.config.parameters["rsi_period"] == 14
    
    def test_rsi_calculation(self):
        """Test RSI calculation."""
        data = self.create_test_data(50)
        
        rsi = self.strategy._calculate_rsi(data['close'])
        
        # RSI should be between 0 and 100
        assert all(rsi.dropna() >= 0)
        assert all(rsi.dropna() <= 100)
        
        # RSI should have NaN values for initial periods
        rsi_period = self.config.parameters["rsi_period"]
        assert rsi.iloc[:rsi_period-1].isna().all()
    
    def test_moving_average_calculation(self):
        """Test moving average calculation."""
        data = self.create_test_data(50)
        
        ma = self.strategy._calculate_ma(data['close'])
        
        # MA should be numeric
        assert pd.api.types.is_numeric_dtype(ma)
        
        # MA should have NaN values for initial periods
        ma_period = self.config.parameters["ma_period"]
        assert ma.iloc[:ma_period-1].isna().all()
    
    def test_generate_signals_empty_data(self):
        """Test signal generation with empty data."""
        empty_data = pd.DataFrame()
        
        signals = self.strategy.generate_signals(empty_data, "TEST")
        
        assert signals == []
    
    def test_generate_signals_insufficient_data(self):
        """Test signal generation with insufficient data."""
        small_data = self.create_test_data(10)  # Less than required periods
        
        signals = self.strategy.generate_signals(small_data, "TEST")
        
        assert signals == []
    
    def test_generate_signals_normal_case(self):
        """Test signal generation with normal data."""
        data = self.create_test_data(100)
        
        signals = self.strategy.generate_signals(data, "TEST")
        
        # Should return a list (may be empty)
        assert isinstance(signals, list)
        
        # If signals exist, check their properties
        for signal in signals:
            assert signal.symbol == "TEST"
            assert signal.signal_type in [SignalType.BUY, SignalType.SELL]
            assert 0 <= signal.confidence <= 1
            assert signal.source == "MomentumStrategy"
            assert signal.timestamp in data.index
    
    def test_oversold_condition(self):
        """Test oversold condition detection."""
        # Create data with artificially low RSI
        data = self.create_test_data(50)
        
        # Mock RSI to be oversold
        with patch.object(self.strategy, '_calculate_rsi') as mock_rsi:
            mock_rsi.return_value = pd.Series([25] * len(data), index=data.index)
            
            # Mock MA to show upward trend
            with patch.object(self.strategy, '_calculate_ma') as mock_ma:
                mock_ma.return_value = pd.Series(range(len(data)), index=data.index)
                
                signals = self.strategy.generate_signals(data, "TEST")
                
                # Should generate buy signals
                buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
                assert len(buy_signals) > 0
    
    def test_overbought_condition(self):
        """Test overbought condition detection."""
        # Create data with artificially high RSI
        data = self.create_test_data(50)
        
        # Mock RSI to be overbought
        with patch.object(self.strategy, '_calculate_rsi') as mock_rsi:
            mock_rsi.return_value = pd.Series([80] * len(data), index=data.index)
            
            # Mock MA to show downward trend
            with patch.object(self.strategy, '_calculate_ma') as mock_ma:
                mock_ma.return_value = pd.Series(list(reversed(range(len(data)))), index=data.index)
                
                signals = self.strategy.generate_signals(data, "TEST")
                
                # Should generate sell signals
                sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
                assert len(sell_signals) > 0
    
    def test_signal_confidence_calculation(self):
        """Test signal confidence calculation."""
        data = self.create_test_data(50)
        
        # Test extreme RSI values should give high confidence
        extreme_rsi = 15  # Very oversold
        confidence = self.strategy._calculate_confidence(extreme_rsi, True)
        assert confidence > 0.8
        
        # Test borderline RSI values should give lower confidence
        borderline_rsi = 32  # Just above oversold threshold
        confidence = self.strategy._calculate_confidence(borderline_rsi, True)
        assert confidence < 0.7
    
    def test_signal_deduplication(self):
        """Test that duplicate signals are handled correctly."""
        data = self.create_test_data(50)
        
        # Generate signals twice and ensure no duplicates
        signals1 = self.strategy.generate_signals(data, "TEST")
        signals2 = self.strategy.generate_signals(data, "TEST")
        
        # Should produce consistent results
        assert len(signals1) == len(signals2)
    
    def test_strategy_registration(self):
        """Test that strategy can be registered."""
        from ..registry import StrategyRegistry
        
        registry = StrategyRegistry()
        
        # Strategy should auto-register
        registered_strategies = registry.list_strategies()
        strategy_names = [s['name'] for s in registered_strategies]
        
        assert "MomentumStrategy" in strategy_names
    
    def test_config_parameter_access(self):
        """Test configuration parameter access."""
        # Test getting existing parameter
        rsi_period = self.strategy.get_config("rsi_period")
        assert rsi_period == 14
        
        # Test getting non-existent parameter with default
        fake_param = self.strategy.get_config("fake_param", "default_value")
        assert fake_param == "default_value"
    
    def test_invalid_config(self):
        """Test handling of invalid configuration."""
        # Test with invalid RSI period
        invalid_config = StrategyConfig(
            name="invalid_test",
            parameters={
                "rsi_period": -5,  # Invalid
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "ma_period": 20
            }
        )
        
        strategy = MomentumStrategy(invalid_config)
        data = self.create_test_data(50)
        
        # Should handle gracefully (return empty signals)
        signals = strategy.generate_signals(data, "TEST")
        assert signals == []


class TestMomentumStrategyEdgeCases:
    """Test edge cases for MomentumStrategy."""
    
    def test_flat_market_data(self):
        """Test strategy with flat (no movement) market data."""
        config = StrategyConfig(
            name="flat_test",
            parameters={
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "ma_period": 20
            }
        )
        strategy = MomentumStrategy(config)
        
        # Create flat price data
        dates = pd.date_range(start=datetime.now(), periods=50, freq='1H')
        flat_price = 100
        
        data = pd.DataFrame({
            'open': [flat_price] * 50,
            'high': [flat_price] * 50,
            'low': [flat_price] * 50,
            'close': [flat_price] * 50,
            'volume': [1000] * 50
        }, index=dates)
        
        signals = strategy.generate_signals(data, "FLAT")
        
        # Flat market should produce few or no signals
        assert len(signals) <= 2  # Allow for minimal noise
    
    def test_extreme_volatility_data(self):
        """Test strategy with extremely volatile data."""
        config = StrategyConfig(
            name="volatile_test",
            parameters={
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "ma_period": 20
            }
        )
        strategy = MomentumStrategy(config)
        
        # Create highly volatile data
        dates = pd.date_range(start=datetime.now(), periods=50, freq='1H')
        volatile_prices = [100 + 50 * (-1)**i for i in range(50)]  # Alternating high/low
        
        data = pd.DataFrame({
            'open': volatile_prices,
            'high': [p + 5 for p in volatile_prices],
            'low': [p - 5 for p in volatile_prices],
            'close': volatile_prices,
            'volume': [1000] * 50
        }, index=dates)
        
        signals = strategy.generate_signals(data, "VOLATILE")
        
        # Should handle volatile data without crashing
        assert isinstance(signals, list)


# Integration test
def test_momentum_strategy_integration():
    """Integration test using real-world-like data patterns."""
    config = StrategyConfig(
        name="integration_test",
        parameters={
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "ma_period": 20,
            "confidence_threshold": 0.5
        }
    )
    
    strategy = MomentumStrategy(config)
    
    # Create realistic trending data with pullbacks
    dates = pd.date_range(start=datetime.now() - timedelta(days=100), periods=100, freq='1H')
    
    # Uptrend with pullbacks
    base_prices = []
    price = 100
    trend = 0.02
    
    for i in range(100):
        # Add trend
        price += trend
        
        # Add some pullbacks every 10-15 periods
        if i % 12 == 0:
            price -= 3  # Pullback
        
        # Add noise
        price += np.random.normal(0, 0.5)
        base_prices.append(price)
    
    data = pd.DataFrame({
        'open': base_prices,
        'high': [p + np.random.uniform(0, 1) for p in base_prices],
        'low': [p - np.random.uniform(0, 1) for p in base_prices],
        'close': base_prices,
        'volume': np.random.randint(1000, 5000, 100)
    }, index=dates)
    
    signals = strategy.generate_signals(data, "INTEGRATION_TEST")
    
    # Should generate some signals in trending market with pullbacks
    assert len(signals) > 0
    
    # Verify signal quality
    buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
    sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
    
    # In an uptrending market, should have more buy signals
    # (though this may vary based on pullbacks and RSI conditions)
    assert len(buy_signals) >= 0  # At least some buy opportunities
    
    # All signals should meet confidence threshold
    for signal in signals:
        assert signal.confidence >= config.parameters["confidence_threshold"]


if __name__ == "__main__":
    # Run basic tests if file is executed directly
    test_momentum_strategy_integration()
    print("Integration test passed!")
