"""
Self-initializing broker account with config loading
"""

import yaml
import os
from loguru import logger


class BrokerAccount:
    """Simple broker account that loads its own config."""
    
    def __init__(self):
        # Load config
        config = self._load_config()
        
        # Initialize from config
        initial_deposit = config['account']['initial_deposit']
        leverage = config['account']['leverage']
        self.stop_out_level = config['risk']['stop_out_level']
        self.risk_per_trade_pct = config['risk']['risk_per_trade_pct']
        
        # Account state
        self.balance = initial_deposit
        self.equity = initial_deposit
        self.used_margin = 0.0
        self.leverage = leverage
        self.is_stopped_out = False
        
        logger.info(f"Account: ${initial_deposit:,.0f} balance, {leverage}x leverage")
    
    def _load_config(self):
        """Load config from YAML file."""
        config_path = os.path.join(os.path.dirname(__file__), 'backtest_config.yaml')
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)['backtest']
    
    @property
    def free_margin(self):
        """Available margin for new positions."""
        return max(0, self.equity - self.used_margin)
    
    @property
    def margin_level(self):
        """Margin level percentage."""
        if self.used_margin == 0:
            return float('inf')
        return (self.equity / self.used_margin) * 100
    
    def update_equity(self, total_floating_pnl):
        """Update equity with current floating P&L."""
        self.equity = self.balance + total_floating_pnl
        
        # Check stop out
        if self.margin_level <= self.stop_out_level and not self.is_stopped_out:
            self.is_stopped_out = True
            logger.error(f"STOP OUT: Margin level {self.margin_level:.1f}%")
    
    def can_open_position(self, required_margin):
        """Check if can open position."""
        return not self.is_stopped_out and self.free_margin >= required_margin
    
    def open_position(self, required_margin):
        """Reserve margin for position."""
        if not self.can_open_position(required_margin):
            return False
        self.used_margin += required_margin
        return True
    
    def close_position(self, released_margin, realized_pnl):
        """Release margin and realize P&L."""
        self.used_margin -= released_margin
        self.balance += realized_pnl
        logger.debug(f"Closed: PnL=${realized_pnl:.2f}, Balance=${self.balance:.2f}")
    
    def calculate_position_size(self, price, signal_direction):
        """Calculate position size based on risk percentage."""
        risk_amount = self.equity * (self.risk_per_trade_pct / 100)
        position_value = risk_amount * self.leverage
        size = position_value / price
        
        if signal_direction == "SELL":
            size = -size
            
        return round(size, 4)
