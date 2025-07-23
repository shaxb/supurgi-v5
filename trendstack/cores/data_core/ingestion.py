"""
Data Ingestion Module

Handles downloading and appending raw market data from various sources.
Supports Yahoo Finance, Alpha Vantage, and other data providers.
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from loguru import logger
import yaml


class DataIngestion:
    """
    Market data ingestion and management class.
    
    Features:
    - Multi-source data downloading (Yahoo Finance primary)
    - Incremental data updates
    - Data validation and basic quality checks
    - Caching and persistence
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize DataIngestion with configuration."""
        self.config = self._load_config(config_path)
        self.data_path = self.config['data']['data_path']
        self.symbols = self.config['data']['symbols']
        self.cache_enabled = self.config['data']['cache_enabled']
        self.cache_duration = self.config['data']['cache_duration_hours']
        
        # Ensure data directory exists
        os.makedirs(self.data_path, exist_ok=True)
        
        logger.info(f"DataIngestion initialized with {len(self.symbols)} symbols")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            raise
    
    def download_symbol_data(
        self, 
        symbol: str, 
        start_date: str = None, 
        end_date: str = None,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Download data for a single symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD=X')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format  
            interval: Data interval ('1d', '1h', '5m', etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Downloading data for {symbol}")
            
            # Default to last 2 years if no dates specified
            if not start_date:
                start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Download data using yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=False
            )
            
            if data.empty:
                logger.warning(f"No data received for {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            data = self._standardize_columns(data)
            
            # Add symbol column
            data['Symbol'] = symbol
            
            logger.info(f"Downloaded {len(data)} rows for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _standardize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across different data sources."""
        column_mapping = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        }
        
        data = data.rename(columns=column_mapping)
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                logger.error(f"Missing required column: {col}")
                
        return data
    
    def download_all_symbols(
        self, 
        start_date: str = None, 
        end_date: str = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Download data for all configured symbols.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        for symbol in self.symbols:
            data = self.download_symbol_data(symbol, start_date, end_date)
            if not data.empty:
                results[symbol] = data
                
                # Save to cache if enabled
                if self.cache_enabled:
                    self._save_to_cache(symbol, data)
        
        logger.info(f"Downloaded data for {len(results)} symbols")
        return results
    
    def _save_to_cache(self, symbol: str, data: pd.DataFrame) -> None:
        """Save data to local cache."""
        try:
            cache_file = os.path.join(self.data_path, f"{symbol.replace('=', '_')}.parquet")
            data.to_parquet(cache_file)
            logger.debug(f"Cached data for {symbol}")
        except Exception as e:
            logger.error(f"Error caching data for {symbol}: {e}")
    
    def load_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load data from local cache if available and fresh."""
        try:
            cache_file = os.path.join(self.data_path, f"{symbol.replace('=', '_')}.parquet")
            
            if not os.path.exists(cache_file):
                return None
            
            # Check if cache is fresh
            if self.cache_enabled:
                file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
                if file_age.total_seconds() > (self.cache_duration * 3600):
                    logger.debug(f"Cache expired for {symbol}")
                    return None
            
            data = pd.read_parquet(cache_file)
            logger.debug(f"Loaded {len(data)} rows from cache for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading cache for {symbol}: {e}")
            return None
    
    def get_latest_data(
        self, 
        symbol: str, 
        days: int = 30
    ) -> pd.DataFrame:
        """
        Get latest data for a symbol, using cache when possible.
        
        Args:
            symbol: Trading symbol
            days: Number of days of recent data to retrieve
            
        Returns:
            DataFrame with latest data
        """
        # First try cache
        cached_data = self.load_from_cache(symbol)
        
        if cached_data is not None and not cached_data.empty:
            # Return recent data from cache
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_data = cached_data[cached_data.index >= cutoff_date]
            
            if len(recent_data) > 0:
                logger.debug(f"Using cached data for {symbol}")
                return recent_data
        
        # Download fresh data
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        return self.download_symbol_data(symbol, start_date=start_date)
    
    def update_incremental(self, symbol: str) -> pd.DataFrame:
        """
        Incrementally update data for a symbol.
        Downloads only new data since last update.
        
        Args:
            symbol: Trading symbol to update
            
        Returns:
            Complete updated DataFrame
        """
        # Load existing data
        existing_data = self.load_from_cache(symbol)
        
        if existing_data is None or existing_data.empty:
            # No existing data, download full history
            logger.info(f"No existing data for {symbol}, downloading full history")
            return self.download_symbol_data(symbol)
        
        # Find last date in existing data
        last_date = existing_data.index.max()
        start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Download new data
        new_data = self.download_symbol_data(symbol, start_date=start_date)
        
        if new_data.empty:
            logger.info(f"No new data available for {symbol}")
            return existing_data
        
        # Combine existing and new data
        combined_data = pd.concat([existing_data, new_data])
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
        combined_data = combined_data.sort_index()
        
        # Save updated data to cache
        if self.cache_enabled:
            self._save_to_cache(symbol, combined_data)
        
        logger.info(f"Added {len(new_data)} new rows for {symbol}")
        return combined_data
    
    def validate_data(self, data: pd.DataFrame, symbol: str) -> Tuple[bool, List[str]]:
        """
        Validate data quality and completeness.
        
        Args:
            data: DataFrame to validate
            symbol: Symbol name for logging
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if data.empty:
            issues.append("DataFrame is empty")
            return False, issues
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for null values
        null_counts = data[required_cols].isnull().sum()
        if null_counts.sum() > 0:
            issues.append(f"Null values found: {null_counts.to_dict()}")
        
        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in data.columns and (data[col] <= 0).any():
                issues.append(f"Non-positive values in {col}")
        
        # Check OHLC logic
        if all(col in data.columns for col in price_cols):
            high_low_check = (data['high'] >= data['low']).all()
            if not high_low_check:
                issues.append("High < Low found")
            
            ohlc_check = (
                (data['high'] >= data['open']) & 
                (data['high'] >= data['close']) &
                (data['low'] <= data['open']) & 
                (data['low'] <= data['close'])
            ).all()
            if not ohlc_check:
                issues.append("OHLC logic violations found")
        
        # Check for data gaps (missing trading days)
        if len(data) > 1:
            date_diff = data.index.to_series().diff().dt.days
            max_gap = date_diff.max()
            if max_gap > 7:  # More than a week gap
                issues.append(f"Large data gap detected: {max_gap} days")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            logger.warning(f"Data validation issues for {symbol}: {issues}")
        else:
            logger.debug(f"Data validation passed for {symbol}")
        
        return is_valid, issues
