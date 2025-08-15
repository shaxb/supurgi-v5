"""
Simple, clean backtesting engine
"""

import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger

from .account import BrokerAccount
from .position import Position
from .results import BacktestResults
from ..signal_core import generate_signals
from ..data_core import load_prices


class DataLoader:
    """Handles data loading and discovery."""
    
    def __init__(self):
        self.data = {}
        self.timestamps = []
    
    def load_period(self, start_date, end_date):
        """Auto-discover and load all available data."""
        data_dir = Path(__file__).parent.parent.parent.parent / "data" / "processed"
        
        # Find all parquet files
        symbols = []
        for file in data_dir.glob("*.parquet"):
            symbol = file.stem.replace("_H4", "").replace("_D", "").replace("_H1", "")
            if symbol not in symbols:
                symbols.append(symbol)
        
        logger.info(f"Found symbols: {symbols}")
        
        # Load data for each symbol
        for symbol in symbols:
            try:
                # Try H4 first, then others
                for timeframe in ["H4", "D", "H1"]:
                    file_path = data_dir / f"{symbol}_{timeframe}.parquet"
                    if file_path.exists():
                        df = pd.read_parquet(file_path)
                        df.index = pd.to_datetime(df.index)
                        
                        # Filter by date range
                        mask = (df.index >= start_date) & (df.index <= end_date)
                        df = df[mask]
                        
                        if not df.empty:
                            self.data[symbol] = df
                            logger.info(f"Loaded {symbol}: {len(df)} bars")
                            break
                            
            except Exception as e:
                logger.warning(f"Failed to load {symbol}: {e}")
        
        # Create unified timestamp index
        all_timestamps = set()
        for df in self.data.values():
            all_timestamps.update(df.index)
        
        self.timestamps = sorted(list(all_timestamps))
        logger.info(f"Total timestamps: {len(self.timestamps)}")
    
    def get_current_prices(self, timestamp):
        """Get current prices for all symbols at timestamp."""
        prices = {}
        for symbol, df in self.data.items():
            if timestamp in df.index:
                prices[symbol] = df.loc[timestamp, 'close']
        return prices


class BrokerEngine:
    """Simple broker simulation."""
    
    def __init__(self):
        # Self-initializing components
        self.account = BrokerAccount()
        self.data_loader = DataLoader()
        self.results = BacktestResults(self.account.balance)
        self.positions = []
        
        # Load config for dates
        config_path = Path(__file__).parent / 'backtest_config.yaml'
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['backtest']
    
    def run_backtest(self):
        """Run the backtest."""
        start_date = self.config['period']['start']
        end_date = self.config['period']['end']
        
        logger.info(f"Starting backtest: {start_date} to {end_date}")
        
        # Load data
        self.data_loader.load_period(start_date, end_date)
        if not self.data_loader.timestamps:
            logger.error("No data loaded")
            return self.results
        
        # Main simulation loop
        for i, timestamp in enumerate(self.data_loader.timestamps):
            if i % 100 == 0:
                logger.debug(f"Processing {timestamp} ({i+1}/{len(self.data_loader.timestamps)})")
            
            self.process_timestamp(timestamp)
            
            # Stop if account is stopped out
            if self.account.is_stopped_out:
                logger.error("Account stopped out - ending backtest")
                break
        
        # Finalize results
        self.results.finalize(self.account.balance)
        self.results.print_summary()
        
        return self.results
    
    def process_timestamp(self, timestamp):
        """Process one timestamp."""
        # Get current prices
        current_prices = self.data_loader.get_current_prices(timestamp)
        if not current_prices:
            return
        
        # Update positions with current prices
        for position in self.positions:
            if position.symbol in current_prices and not position.is_closed:
                current_price = current_prices[position.symbol]
                position.update_pnl(current_price)
                
                # Check for SL/TP
                should_close, reason = position.should_close(current_price)
                if should_close:
                    self.close_position(position, current_price, timestamp, reason)
        
        # Update account equity
        total_floating_pnl = sum(pos.unrealized_pnl for pos in self.positions if not pos.is_closed)
        self.account.update_equity(total_floating_pnl)
        
        # Record equity point
        self.results.add_equity_point(timestamp, self.account.balance, self.account.equity)
        
        # Generate new signals if account is healthy
        if not self.account.is_stopped_out:
            self.process_signals(timestamp, current_prices)
    
    def process_signals(self, timestamp, current_prices):
        """Process new trading signals."""
        for symbol in current_prices:
            if symbol not in self.data_loader.data:
                continue
                
            try:
                # Get signal
                signal = generate_signals(symbol, timestamp)
                if signal and signal.symbol in current_prices:
                    self.try_open_position(signal, current_prices[signal.symbol])
                    
            except Exception as e:
                logger.debug(f"Signal generation error for {symbol}: {e}")
    
    def try_open_position(self, signal, current_price):
        """Try to open a new position."""
        # Calculate position size
        size = self.account.calculate_position_size(current_price, signal.signal_type.value)
        if abs(size) < 0.01:  # Min size check
            return
        
        # Calculate required margin
        required_margin = (abs(size) * current_price) / self.account.leverage
        
        # Check if can open
        if not self.account.can_open_position(required_margin):
            logger.debug(f"Cannot open {signal.symbol}: insufficient margin")
            return
        
        # Open position
        position = Position(signal, size, required_margin)
        
        if self.account.open_position(required_margin):
            self.positions.append(position)
            logger.info(f"Opened {signal.symbol} {signal.signal_type.value} at {current_price}")
    
    def close_position(self, position, price, timestamp, reason):
        """Close a position."""
        if position.is_closed:
            return
        
        position.close(price, timestamp, reason)
        
        # Update account
        self.account.close_position(position.margin, position.realized_pnl)
        
        # Add to results
        self.results.add_trade(position)
