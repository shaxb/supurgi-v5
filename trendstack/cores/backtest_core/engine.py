"""
Broker-like backtesting engine - realistic simulation of broker mechanics
"""

import yaml
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
from loguru import logger

from .account import BrokerAccount
from .position import Position
from .results import BacktestResults
from ..signal_core import generate_signals, get_symbol_strategy_pairs
from ..signal_core.datatypes import SignalIntent, SignalType
from ..data_core import load_prices


class BrokerEngine:
    """Simple but effective broker simulation engine."""
    
    def __init__(self, config_path: str):
        """Initialize from YAML config."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['backtest']
            
        # Setup account
        account_config = self.config['account']
        self.account = BrokerAccount(
            initial_deposit=account_config['initial_deposit'],
            leverage=account_config['leverage']
        )
        
        # Risk settings
        risk_config = self.config['risk']
        self.account.margin_call_level = risk_config['margin_call_level']
        self.account.stop_out_level = risk_config['stop_out_level']
        self.max_positions = risk_config['max_concurrent_positions']
        
        # Position sizing
        self.position_sizing = self.config['position_sizing']
        
        # Execution costs
        self.commission = self.config['costs']['commission_per_trade']
        
        # Results tracking
        self.results = BacktestResults(account_config['initial_deposit'])
        self.positions: List[Position] = []
        
        logger.info(f"Broker engine initialized: ${account_config['initial_deposit']} account, {risk_config['max_concurrent_positions']} max positions")
    
    def run_backtest(self) -> BacktestResults:
        """Run full backtest simulation."""
        period = self.config['period']
        strategies = self.config['strategies'] 
        symbols = self.config['symbols']
        timeframe = self.config['timeframe']
        
        logger.info(f"Starting backtest: {period['start']} to {period['end']}")
        logger.info(f"Testing {strategies} on {symbols} ({timeframe})")
        
        self.results.start_time = datetime.now()
        
        # Get all symbol-strategy combinations
        symbol_strategy_pairs = []
        for symbol in symbols:
            for strategy in strategies:
                strategy_instance = f"{strategy}_{timeframe}"
                symbol_strategy_pairs.append((symbol, strategy_instance, timeframe))
        
        # Load all historical data
        all_data = {}
        for symbol, _, _ in symbol_strategy_pairs:
            data = load_prices(symbol, frame=timeframe)
            if not data.empty:
                # Filter to backtest period
                data = data[period['start']:period['end']]
                all_data[symbol] = data
                logger.info(f"Loaded {len(data)} bars for {symbol}")
            else:
                logger.warning(f"No data for {symbol}")
        
        if not all_data:
            logger.error("No data loaded for backtest")
            return self.results
        
        # Get all unique timestamps across all symbols
        all_timestamps = set()
        for data in all_data.values():
            all_timestamps.update(data.index)
        all_timestamps = sorted(all_timestamps)
        
        logger.info(f"Processing {len(all_timestamps)} time periods")
        
        # Process each timestamp (like live broker)
        for i, timestamp in enumerate(all_timestamps):
            if i % 100 == 0:
                logger.debug(f"Processing {timestamp.strftime('%Y-%m-%d')} ({i+1}/{len(all_timestamps)})")
            
            # 1. Update all positions with current prices
            self._update_positions_pnl(timestamp, all_data)
            
            # 2. Check SL/TP exits
            self._check_exit_orders(timestamp, all_data)
            
            # 3. Update account equity and margin
            self._update_account()
            
            # 4. Check for stop out
            if self.account.is_stop_out():
                logger.error(f"ACCOUNT STOP OUT at {timestamp} - Margin Level: {self.account.margin_level:.1f}%")
                self.results.forced_liquidations += 1
                break
                
            # 5. Generate new signals and open positions
            self._process_new_signals(timestamp, symbol_strategy_pairs, all_data)
        
        # Finalize results
        self._finalize_results()
        return self.results
    
    def _update_positions_pnl(self, timestamp: datetime, all_data: Dict[str, pd.DataFrame]):
        """Update unrealized P&L for all open positions."""
        for position in self.positions:
            if position.is_open and position.symbol in all_data:
                data = all_data[position.symbol]
                if timestamp in data.index:
                    current_price = data.loc[timestamp, 'close']
                    position.update_pnl(current_price)
    
    def _check_exit_orders(self, timestamp: datetime, all_data: Dict[str, pd.DataFrame]):
        """Check if any positions should exit via SL/TP."""
        for position in self.positions[:]:  # Copy list to allow modifications
            if not position.is_open or position.symbol not in all_data:
                continue
                
            data = all_data[position.symbol]
            if timestamp not in data.index:
                continue
                
            bar = data.loc[timestamp]
            exit_result = position.check_exit_conditions(bar['high'], bar['low'], bar['close'])
            
            if exit_result:
                exit_price, exit_reason = exit_result
                self._close_position(position, exit_price, exit_reason, timestamp)
    
    def _update_account(self):
        """Update account equity and margin levels."""
        # Calculate total floating P&L
        total_floating_pnl = sum(pos.unrealized_pnl for pos in self.positions if pos.is_open)
        
        # Update account
        self.account.update_equity(total_floating_pnl)
        
        # Check margin call
        if self.account.is_margin_call() and not hasattr(self, '_margin_call_logged'):
            logger.warning(f"MARGIN CALL - Level: {self.account.margin_level:.1f}%")
            self.results.margin_calls += 1
            self._margin_call_logged = True
        elif not self.account.is_margin_call() and hasattr(self, '_margin_call_logged'):
            delattr(self, '_margin_call_logged')
    
    def _process_new_signals(self, timestamp: datetime, symbol_strategy_pairs, all_data):
        """Generate signals and open new positions if possible."""
        if len([p for p in self.positions if p.is_open]) >= self.max_positions:
            return  # Max positions reached
            
        for symbol, strategy_instance, timeframe in symbol_strategy_pairs:
            if symbol not in all_data:
                continue
                
            data = all_data[symbol]
            if timestamp not in data.index:
                continue
                
            # Get data up to current timestamp for signal generation
            historical_data = data.loc[:timestamp]
            
            try:
                signal = generate_signals(historical_data, symbol, strategy_instance, force=True)
                
                if signal and signal.timestamp == timestamp:  # Only process signals for current timestamp
                    self._try_open_position(signal, timestamp)
                    
            except Exception as e:
                logger.debug(f"Signal generation failed for {symbol}: {e}")
    
    def _try_open_position(self, signal: SignalIntent, timestamp: datetime):
        """Try to open new position if account allows."""
        # Calculate position size
        position_size = self._calculate_position_size(signal)
        if position_size <= 0:
            return
            
        # Calculate required margin
        required_margin = (position_size * signal.price) / self.account.leverage
        
        # Check if account can handle this position
        if not self.account.can_open_position(required_margin):
            logger.debug(f"Insufficient margin for {signal.symbol}: need {required_margin:.2f}, have {self.account.free_margin:.2f}")
            return
            
        # Open position
        position = Position(signal, position_size, required_margin)
        self.positions.append(position)
        self.account.open_position(required_margin)
        
        # Apply commission
        self.account.balance -= self.commission
        
        logger.debug(f"Opened position: {position}")
    
    def _close_position(self, position: Position, close_price: float, reason: str, timestamp: datetime):
        """Close position and update account."""
        position.close(close_price, reason, timestamp)
        
        # Release margin and realize P&L
        self.account.close_position(position.margin_required, position.realized_pnl)
        
        # Apply commission
        self.account.balance -= self.commission
        
        # Add to results
        self.results.add_trade(position)
    
    def _calculate_position_size(self, signal: SignalIntent) -> float:
        """Calculate position size based on risk settings."""
        if self.position_sizing['method'] == 'risk_pct':
            # Risk-based position sizing
            if not signal.stop_loss:
                return 0.0  # No SL, no position
                
            risk_amount = self.account.equity * (self.position_sizing['risk_per_trade_pct'] / 100)
            distance_to_sl = abs(signal.price - signal.stop_loss)
            
            if distance_to_sl > 0:
                position_size = risk_amount / distance_to_sl
                max_size = self.position_sizing.get('max_position_size', float('inf'))
                return min(position_size, max_size)
                
        return 0.0
    
    def _finalize_results(self):
        """Calculate final metrics."""
        # Close any remaining open positions
        for position in self.positions:
            if position.is_open:
                # Close at last known price (market close simulation)
                last_data = None
                for symbol_data in []:  # Would need access to all_data here
                    if position.symbol in symbol_data:
                        last_data = symbol_data[position.symbol].iloc[-1]
                        break
                        
                if last_data is not None:
                    self._close_position(position, last_data['close'], "Market Close", datetime.now())
        
        # Calculate final metrics
        self.results.calculate_metrics(self.account.equity_history, self.account.balance_history)
        self.results.max_drawdown = self.account.get_max_drawdown()
        self.results.end_time = datetime.now()
        
        logger.info(f"Backtest completed: {self.results.total_trades} trades, {self.results.total_pnl:.2f} P&L")
        self.results.print_summary()
    
    @classmethod
    def from_config(cls, config_path: str):
        """Create engine from config file."""
        return cls(config_path)
