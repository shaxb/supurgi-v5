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
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[SignalIntent]:
        """
        Generate momentum signals from OHLCV data.
        
        Args:
            data: OHLCV DataFrame
            symbol: Symbol identifier
            
        Returns:
            List of SignalIntent objects
        """
        if not self.validate_data(data):
            return []
        
        # Get parameters
        short_period = self.get_parameter('short_period', 10)
        long_period = self.get_parameter('long_period', 20)
        rsi_period = self.get_parameter('rsi_period', 14)
        rsi_oversold = self.get_parameter('rsi_oversold', 30)
        rsi_overbought = self.get_parameter('rsi_overbought', 70)
        min_confidence = self.get_parameter('min_confidence', 0.6)
        
        try:
            # Calculate indicators
            signals = []
            
            # Moving averages
            data['ma_short'] = data['close'].rolling(window=short_period).mean()
            data['ma_long'] = data['close'].rolling(window=long_period).mean()
            
            # RSI
            data['rsi'] = self._calculate_rsi(data['close'], rsi_period)
            
            # Price momentum
            data['momentum'] = data['close'].pct_change(periods=5)
            
            # Generate signals for each bar
            for i in range(long_period, len(data)):
                current_row = data.iloc[i]
                prev_row = data.iloc[i-1]
                
                # Check for golden cross (bullish signal)
                if (current_row['ma_short'] > current_row['ma_long'] and
                    prev_row['ma_short'] <= prev_row['ma_long'] and
                    current_row['close'] > current_row['ma_short'] and
                    current_row['rsi'] > rsi_oversold and
                    current_row['momentum'] > 0):
                    
                    signal = self._create_signal(
                        symbol=symbol,
                        timestamp=current_row.name,
                        signal_type=SignalType.BUY,
                        price=current_row['close'],
                        data_row=current_row,
                        confidence_factors=[
                            current_row['momentum'],
                            (current_row['rsi'] - 50) / 50,  # Normalized RSI
                            (current_row['ma_short'] - current_row['ma_long']) / current_row['ma_long']
                        ]
                    )
                    
                    if signal and signal.confidence >= min_confidence:
                        signals.append(signal)
                
                # Check for death cross (bearish signal)
                elif (current_row['ma_short'] < current_row['ma_long'] and
                      prev_row['ma_short'] >= prev_row['ma_long'] and
                      current_row['close'] < current_row['ma_short'] and
                      current_row['rsi'] < rsi_overbought and
                      current_row['momentum'] < 0):
                    
                    signal = self._create_signal(
                        symbol=symbol,
                        timestamp=current_row.name,
                        signal_type=SignalType.SELL,
                        price=current_row['close'],
                        data_row=current_row,
                        confidence_factors=[
                            -current_row['momentum'],  # Negative momentum for bearish
                            -(current_row['rsi'] - 50) / 50,  # Inverted RSI
                            -(current_row['ma_short'] - current_row['ma_long']) / current_row['ma_long']
                        ]
                    )
                    
                    if signal and signal.confidence >= min_confidence:
                        signals.append(signal)
            
            logger.info(f"MomentumStrategy generated {len(signals)} signals for {symbol}")
            return signals
            
        except Exception as e:
            logger.error(f"MomentumStrategy error for {symbol}: {e}")
            return []
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _create_signal(
        self, 
        symbol: str, 
        timestamp: datetime, 
        signal_type: SignalType, 
        price: float,
        data_row: pd.Series,
        confidence_factors: List[float]
    ) -> SignalIntent:
        """Create a signal intent with calculated confidence and levels."""
        
        # Calculate confidence from factors
        confidence = np.mean([abs(factor) for factor in confidence_factors if not np.isnan(factor)])
        confidence = min(confidence, 1.0)  # Cap at 1.0
        
        # Determine signal strength
        if confidence >= 0.8:
            strength = SignalStrength.VERY_STRONG
        elif confidence >= 0.7:
            strength = SignalStrength.STRONG
        elif confidence >= 0.6:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK
        
        # Calculate stop loss and take profit
        atr_period = self.get_parameter('atr_period', 14)
        risk_reward = self.get_parameter('risk_reward', 2.0)
        
        # Simple ATR-based levels (simplified calculation)
        high_low_range = data_row.get('high', price) - data_row.get('low', price)
        atr_estimate = high_low_range * 1.5  # Simplified ATR
        
        if signal_type == SignalType.BUY:
            stop_loss = price - atr_estimate
            take_profit = price + (atr_estimate * risk_reward)
        else:  # SELL
            stop_loss = price + atr_estimate
            take_profit = price - (atr_estimate * risk_reward)
        
        # Create signal
        signal = SignalIntent(
            symbol=symbol,
            timestamp=timestamp,
            signal_type=signal_type,
            strength=strength,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=self.name,
            confidence=confidence,
            risk_reward_ratio=risk_reward,
            timeframe=self.get_parameter('timeframe', 'D'),
            indicators={
                'ma_short': data_row.get('ma_short'),
                'ma_long': data_row.get('ma_long'),
                'rsi': data_row.get('rsi'),
                'momentum': data_row.get('momentum'),
                'atr_estimate': atr_estimate
            },
            notes=f"Momentum crossover signal with {confidence:.2f} confidence"
        )
        
        return signal


# Register the strategy automatically when module is imported
def register_momentum_strategy():
    """Register momentum strategy with default registry."""
    try:
        from .api import register_strategy
        register_strategy('momentum', MomentumStrategy)
        logger.debug("Momentum strategy registered")
    except ImportError:
        # API not available yet, will be registered later
        pass

# Auto-register when imported
register_momentum_strategy()
