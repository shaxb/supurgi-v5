"""
Risk Management Module

Core risk management functionality including position sizing,
risk limits, and drawdown control.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from loguru import logger
from datetime import datetime, timedelta
import yaml
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Position:
    """Position data structure."""
    symbol: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    entry_time: datetime
    side: str  # 'long' or 'short'


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    max_position_size: float = 0.05  # 5% of portfolio
    max_daily_loss: float = 0.02  # 2% daily loss limit
    max_drawdown: float = 0.10  # 10% maximum drawdown
    max_portfolio_risk: float = 0.15  # 15% portfolio risk
    max_single_asset_weight: float = 0.20  # 20% max single asset
    max_correlation_exposure: float = 0.50  # 50% max correlated exposure
    
    # Position limits
    max_positions: int = 10
    max_leverage: float = 3.0
    
    # Risk concentration limits
    max_sector_exposure: float = 0.30  # 30% max sector exposure
    max_currency_exposure: float = 0.40  # 40% max currency exposure


class PositionSizer:
    """
    Position sizing based on risk management principles.
    
    Features:
    - Kelly Criterion sizing
    - Fixed fractional sizing
    - Volatility-based sizing
    - Risk parity sizing
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.default_config = {
            'sizing_method': 'fixed_fractional',
            'risk_per_trade': 0.02,  # 2% risk per trade
            'kelly_fraction': 0.25,  # Use 25% of Kelly optimal
            'min_position_size': 0.001,  # 0.1% minimum
            'max_position_size': 0.05,   # 5% maximum
            'volatility_lookback': 20
        }
        
        # Merge configurations
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        logger.info(f"PositionSizer initialized with method: {self.config['sizing_method']}")
    
    def calculate_position_size(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss_price: float,
        signal_strength: float = 1.0,
        volatility: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win_loss_ratio: Optional[float] = None
    ) -> float:
        """
        Calculate optimal position size based on configured method.
        
        Args:
            portfolio_value: Total portfolio value
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            signal_strength: Signal confidence (0-1)
            volatility: Asset volatility (for volatility-based sizing)
            win_rate: Historical win rate (for Kelly sizing)
            avg_win_loss_ratio: Average win/loss ratio (for Kelly sizing)
            
        Returns:
            Position size as fraction of portfolio
        """
        method = self.config['sizing_method']
        
        if method == 'fixed_fractional':
            size = self._fixed_fractional_sizing(
                portfolio_value, entry_price, stop_loss_price, signal_strength
            )
        elif method == 'kelly':
            size = self._kelly_sizing(
                portfolio_value, entry_price, stop_loss_price, 
                win_rate, avg_win_loss_ratio, signal_strength
            )
        elif method == 'volatility_based':
            size = self._volatility_based_sizing(
                portfolio_value, entry_price, volatility, signal_strength
            )
        elif method == 'risk_parity':
            size = self._risk_parity_sizing(
                portfolio_value, entry_price, volatility, signal_strength
            )
        else:
            logger.warning(f"Unknown sizing method: {method}, using fixed fractional")
            size = self._fixed_fractional_sizing(
                portfolio_value, entry_price, stop_loss_price, signal_strength
            )
        
        # Apply min/max constraints
        size = max(size, self.config['min_position_size'])
        size = min(size, self.config['max_position_size'])
        
        logger.debug(f"Calculated position size: {size:.4f} ({method})")
        return size
    
    def _fixed_fractional_sizing(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss_price: float,
        signal_strength: float
    ) -> float:
        """Fixed fractional position sizing."""
        risk_per_trade = self.config['risk_per_trade'] * signal_strength
        
        if stop_loss_price == 0 or entry_price == stop_loss_price:
            # If no stop loss, use default position size
            return risk_per_trade
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        
        # Calculate position value that would risk the desired amount
        risk_amount = portfolio_value * risk_per_trade
        position_value = risk_amount / (risk_per_share / entry_price)
        
        return position_value / portfolio_value
    
    def _kelly_sizing(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss_price: float,
        win_rate: Optional[float],
        avg_win_loss_ratio: Optional[float],
        signal_strength: float
    ) -> float:
        """Kelly Criterion position sizing."""
        if win_rate is None or avg_win_loss_ratio is None:
            logger.warning("Kelly sizing requires win rate and win/loss ratio, using fixed fractional")
            return self._fixed_fractional_sizing(
                portfolio_value, entry_price, stop_loss_price, signal_strength
            )
        
        # Kelly formula: f = (bp - q) / b
        # where f = fraction to bet, b = odds, p = win probability, q = loss probability
        b = avg_win_loss_ratio  # odds
        p = win_rate  # win probability
        q = 1 - p  # loss probability
        
        kelly_fraction = (b * p - q) / b
        
        # Apply Kelly fraction limit and signal strength
        kelly_fraction = min(kelly_fraction, 1.0)  # Cap at 100%
        kelly_fraction *= self.config['kelly_fraction']  # Apply conservative factor
        kelly_fraction *= signal_strength  # Adjust for signal strength
        
        return max(kelly_fraction, 0)  # Ensure non-negative
    
    def _volatility_based_sizing(
        self,
        portfolio_value: float,
        entry_price: float,
        volatility: Optional[float],
        signal_strength: float
    ) -> float:
        """Volatility-based position sizing (inverse volatility)."""
        if volatility is None or volatility <= 0:
            # Use default sizing if volatility not available
            return self.config['risk_per_trade'] * signal_strength
        
        # Target volatility for position
        target_volatility = 0.02  # 2% target volatility
        
        # Inverse volatility scaling
        size = (target_volatility / volatility) * signal_strength
        
        return size
    
    def _risk_parity_sizing(
        self,
        portfolio_value: float,
        entry_price: float,
        volatility: Optional[float],
        signal_strength: float
    ) -> float:
        """Risk parity position sizing."""
        # This would typically require correlation matrix and portfolio context
        # For now, implement as volatility-adjusted sizing
        if volatility is None:
            return self.config['risk_per_trade'] * signal_strength
        
        # Equal risk contribution approach
        base_risk = 0.02  # 2% base risk
        volatility_adjustment = base_risk / max(volatility, 0.01)
        
        return volatility_adjustment * signal_strength


class DrawdownController:
    """
    Manages drawdown and implements protective measures.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.default_config = {
            'max_drawdown': 0.10,  # 10% max drawdown
            'recovery_threshold': 0.05,  # 5% recovery threshold
            'position_reduction_levels': [0.05, 0.08, 0.10],  # Reduction trigger levels
            'position_reduction_factors': [0.8, 0.5, 0.2],  # Reduction factors
            'stop_trading_threshold': 0.15,  # Stop trading at 15% drawdown
            'lookback_period': 252  # 1 year lookback
        }
        
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        self.peak_value = 0.0
        self.current_drawdown = 0.0
        self.trading_enabled = True
        
        logger.info("DrawdownController initialized")
    
    def update_drawdown(self, current_portfolio_value: float) -> Dict[str, any]:
        """
        Update drawdown calculations and return control actions.
        
        Args:
            current_portfolio_value: Current portfolio value
            
        Returns:
            Dictionary with drawdown info and recommended actions
        """
        # Update peak value
        if current_portfolio_value > self.peak_value:
            self.peak_value = current_portfolio_value
        
        # Calculate current drawdown
        if self.peak_value > 0:
            self.current_drawdown = (self.peak_value - current_portfolio_value) / self.peak_value
        else:
            self.current_drawdown = 0.0
        
        # Determine actions based on drawdown level
        actions = {
            'current_drawdown': self.current_drawdown,
            'peak_value': self.peak_value,
            'position_reduction_factor': 1.0,
            'stop_trading': False,
            'alert_level': RiskLevel.LOW
        }
        
        # Check drawdown levels
        reduction_levels = self.config['position_reduction_levels']
        reduction_factors = self.config['position_reduction_factors']
        
        for i, level in enumerate(reduction_levels):
            if self.current_drawdown >= level:
                actions['position_reduction_factor'] = reduction_factors[i]
                actions['alert_level'] = RiskLevel.MEDIUM if i < 2 else RiskLevel.HIGH
        
        # Check if trading should be stopped
        if self.current_drawdown >= self.config['stop_trading_threshold']:
            actions['stop_trading'] = True
            actions['alert_level'] = RiskLevel.CRITICAL
            self.trading_enabled = False
        
        # Check for recovery
        recovery_threshold = self.config['recovery_threshold']
        if not self.trading_enabled and self.current_drawdown < recovery_threshold:
            self.trading_enabled = True
            actions['stop_trading'] = False
            logger.info(f"Trading re-enabled after recovery to {self.current_drawdown:.2%} drawdown")
        
        return actions
    
    def get_position_scaling_factor(self) -> float:
        """Get current position scaling factor based on drawdown."""
        reduction_levels = self.config['position_reduction_levels']
        reduction_factors = self.config['position_reduction_factors']
        
        for i, level in enumerate(reduction_levels):
            if self.current_drawdown >= level:
                return reduction_factors[i]
        
        return 1.0  # No reduction


class RiskManager:
    """
    Main risk management coordinator.
    
    Features:
    - Portfolio-level risk management
    - Position-level risk control
    - Real-time risk monitoring
    - Risk limit enforcement
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize RiskManager with configuration."""
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.risk_limits = RiskLimits(**self.config.get('risk_limits', {}))
        self.position_sizer = PositionSizer(self.config.get('position_sizing', {}))
        self.drawdown_controller = DrawdownController(self.config.get('drawdown_control', {}))
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.portfolio_value = 0.0
        self.daily_pnl = 0.0
        self.risk_metrics = {}
        
        logger.info("RiskManager initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}
    
    def check_position_limits(
        self,
        symbol: str,
        proposed_size: float,
        entry_price: float
    ) -> Tuple[bool, str, float]:
        """
        Check if proposed position meets risk limits.
        
        Args:
            symbol: Trading symbol
            proposed_size: Proposed position size (as fraction of portfolio)
            entry_price: Entry price
            
        Returns:
            Tuple of (allowed, reason, adjusted_size)
        """
        # Check maximum position size
        if proposed_size > self.risk_limits.max_position_size:
            return False, f"Position size {proposed_size:.2%} exceeds limit {self.risk_limits.max_position_size:.2%}", self.risk_limits.max_position_size
        
        # Check maximum number of positions
        if len(self.positions) >= self.risk_limits.max_positions and symbol not in self.positions:
            return False, f"Maximum positions ({self.risk_limits.max_positions}) reached", 0.0
        
        # Check portfolio concentration
        if symbol in self.positions:
            current_weight = abs(self.positions[symbol].size * self.positions[symbol].current_price) / self.portfolio_value
            new_weight = abs(proposed_size * entry_price) / self.portfolio_value
            total_weight = current_weight + new_weight
            
            if total_weight > self.risk_limits.max_single_asset_weight:
                max_additional = self.risk_limits.max_single_asset_weight - current_weight
                adjusted_size = max_additional * self.portfolio_value / entry_price
                return False, f"Total asset weight {total_weight:.2%} exceeds limit", max(adjusted_size, 0)
        
        # Check drawdown-based scaling
        scaling_factor = self.drawdown_controller.get_position_scaling_factor()
        if scaling_factor < 1.0:
            adjusted_size = proposed_size * scaling_factor
            return True, f"Position scaled by {scaling_factor:.2f} due to drawdown", adjusted_size
        
        return True, "Position approved", proposed_size
    
    def calculate_portfolio_risk(self) -> Dict[str, float]:
        """Calculate portfolio-level risk metrics."""
        if not self.positions or self.portfolio_value <= 0:
            return {'total_risk': 0.0, 'var_1d': 0.0, 'concentration_risk': 0.0}
        
        total_exposure = 0.0
        max_single_exposure = 0.0
        
        for position in self.positions.values():
            exposure = abs(position.size * position.current_price) / self.portfolio_value
            total_exposure += exposure
            max_single_exposure = max(max_single_exposure, exposure)
        
        # Simple risk metrics (would be more sophisticated in production)
        risk_metrics = {
            'total_exposure': total_exposure,
            'max_single_exposure': max_single_exposure,
            'concentration_risk': max_single_exposure / max(total_exposure, 0.01),
            'leverage': total_exposure,
            'num_positions': len(self.positions)
        }
        
        return risk_metrics
    
    def update_positions(self, positions_data: List[Dict]):
        """Update position data."""
        self.positions.clear()
        
        for pos_data in positions_data:
            position = Position(
                symbol=pos_data['symbol'],
                size=pos_data['size'],
                entry_price=pos_data['entry_price'],
                current_price=pos_data['current_price'],
                unrealized_pnl=pos_data['unrealized_pnl'],
                entry_time=pos_data.get('entry_time', datetime.now()),
                side=pos_data.get('side', 'long')
            )
            self.positions[position.symbol] = position
    
    def update_portfolio_value(self, portfolio_value: float, daily_pnl: float):
        """Update portfolio value and daily P&L."""
        self.portfolio_value = portfolio_value
        self.daily_pnl = daily_pnl
        
        # Update drawdown controller
        drawdown_actions = self.drawdown_controller.update_drawdown(portfolio_value)
        
        # Update risk metrics
        self.risk_metrics = self.calculate_portfolio_risk()
        self.risk_metrics.update(drawdown_actions)
        
        logger.debug(f"Portfolio updated: ${portfolio_value:,.2f}, Daily P&L: ${daily_pnl:,.2f}")
    
    def get_risk_summary(self) -> Dict[str, any]:
        """Get comprehensive risk summary."""
        summary = {
            'portfolio_value': self.portfolio_value,
            'daily_pnl': self.daily_pnl,
            'positions_count': len(self.positions),
            'risk_limits': {
                'max_position_size': self.risk_limits.max_position_size,
                'max_daily_loss': self.risk_limits.max_daily_loss,
                'max_drawdown': self.risk_limits.max_drawdown,
                'max_positions': self.risk_limits.max_positions
            },
            'current_metrics': self.risk_metrics,
            'alerts': []
        }
        
        # Check for risk limit violations
        if self.risk_metrics.get('current_drawdown', 0) > self.risk_limits.max_drawdown:
            summary['alerts'].append({
                'level': 'HIGH',
                'message': f"Drawdown {self.risk_metrics['current_drawdown']:.2%} exceeds limit {self.risk_limits.max_drawdown:.2%}"
            })
        
        if abs(self.daily_pnl) > self.portfolio_value * self.risk_limits.max_daily_loss:
            summary['alerts'].append({
                'level': 'MEDIUM',
                'message': f"Daily loss ${self.daily_pnl:,.2f} approaches limit"
            })
        
        if len(self.positions) >= self.risk_limits.max_positions:
            summary['alerts'].append({
                'level': 'LOW',
                'message': f"Position count at maximum ({self.risk_limits.max_positions})"
            })
        
        return summary
    
    def should_allow_trade(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float
    ) -> Tuple[bool, str]:
        """
        Determine if a trade should be allowed based on risk rules.
        
        Args:
            symbol: Trading symbol
            side: 'long' or 'short'
            size: Position size
            price: Entry price
            
        Returns:
            Tuple of (allowed, reason)
        """
        # Check if trading is enabled (drawdown control)
        if not self.drawdown_controller.trading_enabled:
            return False, "Trading disabled due to excessive drawdown"
        
        # Check daily loss limit
        if self.daily_pnl < -self.portfolio_value * self.risk_limits.max_daily_loss:
            return False, f"Daily loss limit exceeded: ${self.daily_pnl:,.2f}"
        
        # Check position limits
        position_size_fraction = size * price / self.portfolio_value
        allowed, reason, adjusted_size = self.check_position_limits(symbol, position_size_fraction, price)
        
        if not allowed and adjusted_size <= 0:
            return False, reason
        
        return True, "Trade approved"
