"""
Broker account simulation - handles margin, equity, balance like real broker
"""

from typing import List
from loguru import logger


class BrokerAccount:
    """Simple broker account simulation with margin management."""
    
    def __init__(self, initial_deposit: float, leverage: int = 100):
        # Core account values
        self.initial_deposit = initial_deposit
        self.balance = initial_deposit          # Realized P&L + deposit  
        self.equity = initial_deposit           # Balance + floating P&L
        self.used_margin = 0.0                 # Margin used by positions
        self.free_margin = initial_deposit      # Available for new trades
        self.leverage = leverage
        
        # Risk levels (like MT5)
        self.margin_call_level = 100.0         # % - margin call threshold
        self.stop_out_level = 50.0             # % - forced liquidation
        
        # Track history
        self.equity_history = [initial_deposit]
        self.balance_history = [initial_deposit]
        
    def update_equity(self, total_floating_pnl: float):
        """Update equity with current floating P&L."""
        self.equity = self.balance + total_floating_pnl
        self.equity_history.append(self.equity)
        
        # Update free margin
        self.free_margin = max(0, self.equity - self.used_margin)
        
        logger.debug(f"Account update: Balance={self.balance:.2f}, Equity={self.equity:.2f}, Free Margin={self.free_margin:.2f}")
    
    def can_open_position(self, required_margin: float) -> bool:
        """Check if account has enough free margin."""
        return self.free_margin >= required_margin
    
    def open_position(self, required_margin: float):
        """Reserve margin for new position."""
        if not self.can_open_position(required_margin):
            raise ValueError(f"Insufficient margin. Required: {required_margin}, Available: {self.free_margin}")
            
        self.used_margin += required_margin
        self.free_margin -= required_margin
        logger.debug(f"Position opened: Used margin now {self.used_margin:.2f}")
    
    def close_position(self, released_margin: float, realized_pnl: float):
        """Release margin and realize P&L."""
        self.used_margin -= released_margin
        self.balance += realized_pnl
        self.balance_history.append(self.balance)
        
        logger.info(f"Position closed: P&L={realized_pnl:.2f}, Balance={self.balance:.2f}")
    
    @property
    def margin_level(self) -> float:
        """Calculate margin level (Equity/Used Margin * 100)."""
        if self.used_margin == 0:
            return float('inf')
        return (self.equity / self.used_margin) * 100
    
    def is_margin_call(self) -> bool:
        """Check if account is in margin call."""
        return self.margin_level <= self.margin_call_level
    
    def is_stop_out(self) -> bool:
        """Check if account should be stopped out (liquidated)."""
        return self.margin_level <= self.stop_out_level
    
    def get_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity history."""
        if len(self.equity_history) < 2:
            return 0.0
            
        peak = self.equity_history[0]
        max_dd = 0.0
        
        for equity in self.equity_history:
            if equity > peak:
                peak = equity
            else:
                drawdown = (peak - equity) / peak * 100
                max_dd = max(max_dd, drawdown)
                
        return max_dd
    
    def __str__(self):
        return f"Account(Balance: ${self.balance:.2f}, Equity: ${self.equity:.2f}, Margin Level: {self.margin_level:.1f}%)"
