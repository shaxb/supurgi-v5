"""
Simple position management
"""

from datetime import datetime
from loguru import logger

from ..signal_core.datatypes import SignalIntent, SignalType


class Position:
    """Simple position with SL/TP."""
    
    def __init__(self, signal: SignalIntent, size: float, margin: float):
        self.symbol = signal.symbol
        self.signal_type = signal.signal_type
        self.entry_price = signal.price
        self.size = size
        self.margin = margin
        self.open_time = signal.timestamp
        
        # SL/TP from signal
        self.stop_loss = signal.stop_loss
        self.take_profit = signal.take_profit
        
        # P&L
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
        # Status
        self.is_closed = False
        self.close_reason = None
        self.close_time = None
        self.close_price = None
        
        logger.info(f"{self.symbol} {signal.signal_type.value} opened at {self.entry_price}")
    
    def update_pnl(self, current_price):
        """Update unrealized P&L."""
        if self.is_closed:
            return
            
        price_diff = current_price - self.entry_price
        if self.signal_type == SignalType.SELL:
            price_diff = -price_diff
            
        self.unrealized_pnl = price_diff * abs(self.size)
    
    def should_close(self, current_price):
        """Check if position should close due to SL/TP."""
        if self.is_closed:
            return False, None
            
        if self.signal_type == SignalType.BUY:
            if self.stop_loss and current_price <= self.stop_loss:
                return True, "SL"
            if self.take_profit and current_price >= self.take_profit:
                return True, "TP"
        else:  # SELL
            if self.stop_loss and current_price >= self.stop_loss:
                return True, "SL"
            if self.take_profit and current_price <= self.take_profit:
                return True, "TP"
        
        return False, None
    
    def close(self, price, timestamp, reason="Manual"):
        """Close the position."""
        if self.is_closed:
            return
            
        self.close_price = price
        self.close_time = timestamp
        self.close_reason = reason
        self.is_closed = True
        
        # Calculate realized P&L
        price_diff = price - self.entry_price
        if self.signal_type == SignalType.SELL:
            price_diff = -price_diff
            
        self.realized_pnl = price_diff * abs(self.size)
        
        logger.info(f"{self.symbol} closed at {price} ({reason}): P&L=${self.realized_pnl:.2f}")
