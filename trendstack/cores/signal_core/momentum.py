"""
Momentum Strategy - First Strategy Plugin

Simple momentum-based signal generation strategy.
Generates signals based on price momentum and moving average crossovers.
"""

import pandas as pd
import numpy as np
from typing import List
from datetime import datetime
from loguru import logger

from .registry import BaseStrategy
from .datatypes import SignalIntent, SignalType, SignalStrength


class MomentumStrategy(BaseStrategy):
    """
    Momentum-based signal generation strategy.
    
    Generates BUY signals when:
    - Price is above short-term moving average
    - Short MA is above long MA (golden cross)
    - RSI is in favorable range
    
    Generates SELL signals when opposite conditions are met.
    """
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> SignalIntent:
        """
        Generate a simple test signal from OHLCV data.
        
        Args:
            data: OHLCV DataFrame
            symbol: Symbol identifier
            
        Returns:
            Single SignalIntent object or None
        """
        if not self.validate_data(data):
            return None

        logger.error(self.config)

        try:
            # Simple test: generate BUY signal if last close > previous close
            if len(data) < 2:
                return None
                
            last_row = data.iloc[-1]
            prev_row = data.iloc[-2]
            
            # Simple momentum test
            if last_row['close'] > prev_row['close']:
                signal = SignalIntent(
                    symbol=symbol,
                    timestamp=last_row.name if hasattr(last_row, 'name') else datetime.now(),
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.MODERATE,
                    price=last_row['close'],
                    stop_loss=last_row['close'] * 0.98,  # 2% stop loss
                    take_profit=last_row['close'] * 1.04,  # 4% take profit
                    strategy_name=self.name,
                    confidence=0.7,
                    risk_reward_ratio=2.0,
                    timeframe=self.get_parameter('timeframe', 'D'),
                    indicators={
                        'last_close': last_row['close'],
                        'prev_close': prev_row['close'],
                        'momentum': (last_row['close'] - prev_row['close']) / prev_row['close']
                    },
                    notes=f"Simple test signal: price up from {prev_row['close']:.2f} to {last_row['close']:.2f}"
                )
                
                logger.info(f"MomentumStrategy generated BUY signal for {symbol} at {last_row['close']:.2f}")
                return signal
            else:
                logger.info(f"MomentumStrategy: No signal for {symbol} (price down)")
                return None
            
        except Exception as e:
            logger.error(f"MomentumStrategy error for {symbol}: {e}")
            return None


# Register the strategy automatically when module is imported
def register_momentum_strategy():
    """Register momentum strategy with default registry."""
    try:
        from .api import _strategy_registry
        _strategy_registry.register('momentum', MomentumStrategy)
        logger.debug("Momentum strategy registered")
    except ImportError:
        # Registry not available yet, will be registered later
        pass

# Auto-register when imported
register_momentum_strategy()
