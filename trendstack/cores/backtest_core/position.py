"""
Position management for backtesting
"""

from datetime import datetime
from typing import Optional
from loguru import logger

from ..signal_core.datatypes import SignalIntent, SignalType


class Position:
    """Simple position with SL/TP management."""
    
    def __init__(self, signal: SignalIntent, position_size: float, margin_required: float):
        # Basic position info
        self.symbol = signal.symbol
        self.signal_type = signal.signal_type
        self.entry_price = signal.price
        self.position_size = position_size
        self.margin_required = margin_required
        self.open_time = signal.timestamp
        
        # SL/TP from signal
        self.stop_loss = signal.stop_loss
        self.take_profit = signal.take_profit
        
        # P&L tracking
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
        # Status
        self.is_open = True
        self.close_reason = None
        self.close_time = None
        self.close_price = None
        
        logger.info(f"Position opened: {self.symbol} {signal.signal_type.value} at {self.entry_price:.2f}")
    
    def update_pnl(self, current_price: float):
        """Update unrealized P&L based on current price."""
        if not self.is_open:
            return
            
        if self.signal_type == SignalType.BUY:
            self.unrealized_pnl = (current_price - self.entry_price) * self.position_size
        elif self.signal_type == SignalType.SELL:
            self.unrealized_pnl = (self.entry_price - current_price) * self.position_size
    
    def check_exit_conditions(self, bar_high: float, bar_low: float, bar_close: float) -> Optional[tuple]:
        """
        Check if SL or TP should be hit during this bar.
        
        Returns:
            (exit_price, exit_reason) if position should close, None otherwise
        """
        if not self.is_open:
            return None
        
        if self.signal_type == SignalType.BUY:
            # Long position: check if low hit SL or high hit TP
            if self.stop_loss and bar_low <= self.stop_loss:
                return (self.stop_loss, "Stop Loss")
            elif self.take_profit and bar_high >= self.take_profit:
                return (self.take_profit, "Take Profit")
                
        elif self.signal_type == SignalType.SELL:
            # Short position: check if high hit SL or low hit TP  
            if self.stop_loss and bar_high >= self.stop_loss:
                return (self.stop_loss, "Stop Loss")
            elif self.take_profit and bar_low <= self.take_profit:
                return (self.take_profit, "Take Profit")
        
        return None
    
    def close(self, close_price: float, close_reason: str, close_time: datetime):
        """Close the position and calculate realized P&L."""
        if not self.is_open:
            return
            
        self.is_open = False
        self.close_price = close_price
        self.close_reason = close_reason
        self.close_time = close_time
        
        # Calculate final realized P&L
        if self.signal_type == SignalType.BUY:
            self.realized_pnl = (close_price - self.entry_price) * self.position_size
        elif self.signal_type == SignalType.SELL:
            self.realized_pnl = (self.entry_price - close_price) * self.position_size
            
        self.unrealized_pnl = 0.0  # No longer unrealized
        
        logger.info(f"Position closed: {self.symbol} at {close_price:.2f} ({close_reason}) - P&L: {self.realized_pnl:.2f}")
    
    def __str__(self):
        status = "OPEN" if self.is_open else "CLOSED"
        pnl = self.unrealized_pnl if self.is_open else self.realized_pnl
        return f"Position({self.symbol} {self.signal_type.value} @ {self.entry_price:.2f} - {status} P&L: {pnl:.2f})"
