"""
Trading Costs Module

Handles calculation of trading costs including spreads, commissions,
and swap rates for different instruments and brokers.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple
from loguru import logger
import yaml


class CostCalculator:
    """
    Trading cost calculation and management class.
    
    Features:
    - Spread calculation and estimation
    - Commission structure handling
    - Swap rate calculations
    - Cost impact analysis
    - Broker-specific cost models
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize CostCalculator with configuration."""
        self.config = self._load_config(config_path)
        self.broker = self.config['trading']['broker']
        self.commission_pct = self.config.get('backtesting', {}).get('commission_pct', 0.01)
        self.slippage_bps = self.config['trading']['execution']['slippage_bps']
        
        # Load cost tables
        self._load_cost_tables()
        
        logger.info(f"CostCalculator initialized for broker: {self.broker}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            raise
    
    def _load_cost_tables(self):
        """Load broker-specific cost tables."""
        # Default cost structures - should be loaded from external files in production
        self.spread_table = {
            # Major forex pairs (in pips)
            'EURUSD=X': 0.5,
            'GBPUSD=X': 0.7,
            'USDJPY=X': 0.6,
            'USDCHF=X': 0.8,
            'AUDUSD=X': 0.9,
            'USDCAD=X': 1.2,
            'NZDUSD=X': 1.5,
            
            # Minor pairs
            'EURGBP=X': 1.0,
            'EURJPY=X': 1.2,
            'GBPJPY=X': 1.8,
            
            # Exotic pairs
            'USDZAR=X': 15.0,
            'USDTRY=X': 25.0,
        }
        
        self.commission_table = {
            'forex': {
                'type': 'spread_markup',  # or 'fixed_per_lot', 'percentage'
                'value': 0.0  # No additional commission if spread markup
            },
            'stocks': {
                'type': 'percentage',
                'value': 0.1  # 0.1% commission
            },
            'futures': {
                'type': 'fixed_per_contract',
                'value': 2.50  # $2.50 per contract
            }
        }
        
        # Swap rates (annual rates in percentage)
        self.swap_table = {
            'EURUSD=X': {'long': -0.5, 'short': 0.2},
            'GBPUSD=X': {'long': -0.3, 'short': 0.1},
            'USDJPY=X': {'long': 0.8, 'short': -1.2},
            'AUDUSD=X': {'long': 1.2, 'short': -1.8},
            'USDCAD=X': {'long': 0.5, 'short': -0.8},
        }
    
    def calculate_spread_cost(
        self, 
        symbol: str, 
        position_size: float, 
        price: float
    ) -> float:
        """
        Calculate spread cost for a trade.
        
        Args:
            symbol: Trading symbol
            position_size: Position size (positive for long, negative for short)
            price: Current market price
            
        Returns:
            Spread cost in base currency
        """
        if symbol not in self.spread_table:
            logger.warning(f"No spread data for {symbol}, using default")
            spread_pips = 2.0  # Default spread
        else:
            spread_pips = self.spread_table[symbol]
        
        # Convert pips to price units
        if symbol.endswith('JPY=X'):
            pip_value = 0.01  # JPY pairs have different pip value
        else:
            pip_value = 0.0001  # Most forex pairs
        
        spread_price = spread_pips * pip_value
        
        # Spread cost is half spread times position size
        spread_cost = abs(position_size) * spread_price * 0.5
        
        logger.debug(f"Spread cost for {symbol}: {spread_cost:.4f}")
        return spread_cost
    
    def calculate_commission(
        self, 
        symbol: str, 
        position_size: float, 
        price: float
    ) -> float:
        """
        Calculate commission for a trade.
        
        Args:
            symbol: Trading symbol
            position_size: Position size
            price: Trade price
            
        Returns:
            Commission cost in base currency
        """
        # Determine instrument type
        if symbol.endswith('=X'):
            instrument_type = 'forex'
        elif symbol.endswith('.F'):
            instrument_type = 'futures'
        else:
            instrument_type = 'stocks'
        
        if instrument_type not in self.commission_table:
            logger.warning(f"No commission structure for {instrument_type}")
            return 0.0
        
        commission_config = self.commission_table[instrument_type]
        commission_type = commission_config['type']
        commission_value = commission_config['value']
        
        if commission_type == 'percentage':
            commission = abs(position_size * price) * (commission_value / 100)
        elif commission_type == 'fixed_per_lot':
            lots = abs(position_size) / 100000  # Standard lot size for forex
            commission = lots * commission_value
        elif commission_type == 'fixed_per_contract':
            contracts = abs(position_size)
            commission = contracts * commission_value
        elif commission_type == 'spread_markup':
            commission = 0.0  # Cost already included in spread
        else:
            logger.error(f"Unknown commission type: {commission_type}")
            commission = 0.0
        
        logger.debug(f"Commission for {symbol}: {commission:.4f}")
        return commission
    
    def calculate_slippage_cost(
        self, 
        symbol: str, 
        position_size: float, 
        price: float, 
        market_impact: Optional[float] = None
    ) -> float:
        """
        Calculate estimated slippage cost.
        
        Args:
            symbol: Trading symbol
            position_size: Position size
            price: Expected price
            market_impact: Custom market impact (overrides config)
            
        Returns:
            Estimated slippage cost
        """
        if market_impact is None:
            slippage_bps = self.slippage_bps
        else:
            slippage_bps = market_impact * 10000  # Convert to basis points
        
        # Slippage proportional to position size (square root rule)
        size_factor = np.sqrt(abs(position_size) / 100000)  # Normalize by standard lot
        
        slippage_rate = (slippage_bps / 10000) * size_factor
        slippage_cost = abs(position_size) * price * slippage_rate
        
        logger.debug(f"Slippage cost for {symbol}: {slippage_cost:.4f}")
        return slippage_cost
    
    def calculate_swap_cost(
        self, 
        symbol: str, 
        position_size: float, 
        price: float, 
        days_held: int,
        position_type: str = 'long'
    ) -> float:
        """
        Calculate swap (overnight financing) cost.
        
        Args:
            symbol: Trading symbol
            position_size: Position size
            price: Current price
            days_held: Number of days position was held
            position_type: 'long' or 'short'
            
        Returns:
            Swap cost (positive = cost, negative = credit)
        """
        if symbol not in self.swap_table:
            logger.warning(f"No swap data for {symbol}")
            return 0.0
        
        swap_rates = self.swap_table[symbol]
        
        if position_type not in swap_rates:
            logger.error(f"Invalid position type: {position_type}")
            return 0.0
        
        annual_swap_rate = swap_rates[position_type] / 100  # Convert percentage
        daily_swap_rate = annual_swap_rate / 365
        
        notional_value = abs(position_size) * price
        swap_cost = notional_value * daily_swap_rate * days_held
        
        # If position_size is negative (short), flip the cost
        if position_size < 0:
            swap_cost = -swap_cost
        
        logger.debug(f"Swap cost for {symbol}: {swap_cost:.4f}")
        return swap_cost
    
    def calculate_total_cost(
        self, 
        symbol: str, 
        position_size: float, 
        price: float, 
        days_held: int = 1,
        include_swap: bool = True
    ) -> Dict[str, float]:
        """
        Calculate total trading cost breakdown.
        
        Args:
            symbol: Trading symbol
            position_size: Position size
            price: Trade price
            days_held: Days position held (for swap calculation)
            include_swap: Whether to include swap costs
            
        Returns:
            Dictionary with cost breakdown
        """
        costs = {}
        
        # Calculate individual cost components
        costs['spread'] = self.calculate_spread_cost(symbol, position_size, price)
        costs['commission'] = self.calculate_commission(symbol, position_size, price)
        costs['slippage'] = self.calculate_slippage_cost(symbol, position_size, price)
        
        if include_swap and days_held > 0:
            position_type = 'long' if position_size > 0 else 'short'
            costs['swap'] = self.calculate_swap_cost(
                symbol, position_size, price, days_held, position_type
            )
        else:
            costs['swap'] = 0.0
        
        # Calculate total
        costs['total'] = sum(costs.values())
        
        # Calculate as basis points of notional
        notional = abs(position_size) * price
        if notional > 0:
            costs['total_bps'] = (costs['total'] / notional) * 10000
        else:
            costs['total_bps'] = 0.0
        
        logger.debug(f"Total cost breakdown for {symbol}: {costs}")
        return costs
    
    def estimate_market_impact(
        self, 
        symbol: str, 
        position_size: float, 
        average_volume: float
    ) -> float:
        """
        Estimate market impact based on position size relative to average volume.
        
        Args:
            symbol: Trading symbol
            position_size: Position size
            average_volume: Average daily volume
            
        Returns:
            Estimated market impact in basis points
        """
        if average_volume <= 0:
            logger.warning(f"Invalid average volume for {symbol}")
            return self.slippage_bps
        
        # Simple square root market impact model
        volume_ratio = abs(position_size) / average_volume
        
        # Base impact + impact proportional to sqrt of volume ratio
        base_impact = 0.5  # 0.5 bps base impact
        variable_impact = 5.0 * np.sqrt(volume_ratio)  # Variable component
        
        total_impact = base_impact + variable_impact
        
        # Cap at reasonable maximum
        max_impact = 50.0  # 50 bps maximum
        market_impact = min(total_impact, max_impact)
        
        logger.debug(f"Market impact estimate for {symbol}: {market_impact:.2f} bps")
        return market_impact
    
    def calculate_breakeven_move(
        self, 
        symbol: str, 
        position_size: float, 
        price: float
    ) -> float:
        """
        Calculate the minimum price move needed to break even on costs.
        
        Args:
            symbol: Trading symbol
            position_size: Position size
            price: Entry price
            
        Returns:
            Breakeven move in price units
        """
        total_costs = self.calculate_total_cost(
            symbol, position_size, price, include_swap=False
        )
        
        # Breakeven move = total costs / position size
        if abs(position_size) > 0:
            breakeven_move = total_costs['total'] / abs(position_size)
        else:
            breakeven_move = 0.0
        
        logger.debug(f"Breakeven move for {symbol}: {breakeven_move:.5f}")
        return breakeven_move
    
    def cost_analysis_report(
        self, 
        trades_df: pd.DataFrame
    ) -> Dict:
        """
        Generate cost analysis report for a series of trades.
        
        Args:
            trades_df: DataFrame with columns: symbol, size, price, days_held
            
        Returns:
            Cost analysis report dictionary
        """
        if trades_df.empty:
            return {'error': 'No trades provided'}
        
        report = {
            'total_trades': len(trades_df),
            'cost_breakdown': {
                'spread': 0.0,
                'commission': 0.0,
                'slippage': 0.0,
                'swap': 0.0,
                'total': 0.0
            },
            'cost_statistics': {},
            'symbol_breakdown': {}
        }
        
        all_costs = []
        
        for _, trade in trades_df.iterrows():
            symbol = trade['symbol']
            size = trade['size']
            price = trade['price']
            days_held = trade.get('days_held', 1)
            
            costs = self.calculate_total_cost(symbol, size, price, days_held)
            all_costs.append(costs)
            
            # Add to totals
            for cost_type in ['spread', 'commission', 'slippage', 'swap', 'total']:
                report['cost_breakdown'][cost_type] += costs[cost_type]
            
            # Symbol-specific breakdown
            if symbol not in report['symbol_breakdown']:
                report['symbol_breakdown'][symbol] = {
                    'trades': 0,
                    'total_cost': 0.0,
                    'avg_cost_bps': 0.0
                }
            
            report['symbol_breakdown'][symbol]['trades'] += 1
            report['symbol_breakdown'][symbol]['total_cost'] += costs['total']
        
        # Calculate statistics
        if all_costs:
            total_costs = [c['total'] for c in all_costs]
            total_bps = [c['total_bps'] for c in all_costs]
            
            report['cost_statistics'] = {
                'mean_cost': np.mean(total_costs),
                'median_cost': np.median(total_costs),
                'std_cost': np.std(total_costs),
                'mean_cost_bps': np.mean(total_bps),
                'median_cost_bps': np.median(total_bps),
                'total_cost_pct_of_volume': 0.0  # Would need volume data
            }
            
            # Calculate average cost per symbol
            for symbol in report['symbol_breakdown']:
                symbol_data = report['symbol_breakdown'][symbol]
                if symbol_data['trades'] > 0:
                    symbol_trades = trades_df[trades_df['symbol'] == symbol]
                    symbol_costs = [c for c, trade in zip(all_costs, trades_df.itertuples()) 
                                  if trade.symbol == symbol]
                    
                    if symbol_costs:
                        avg_bps = np.mean([c['total_bps'] for c in symbol_costs])
                        symbol_data['avg_cost_bps'] = avg_bps
        
        logger.info(f"Cost analysis complete: {report['cost_breakdown']['total']:.2f} total cost")
        return report
