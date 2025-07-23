"""
Data Cleaning Module

Handles data quality improvements including gap filling, outlier removal,
roll adjustments for futures, and other data preprocessing tasks.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
import yaml


class DataCleaner:
    """
    Data cleaning and preprocessing class.
    
    Features:
    - Gap filling using various methods
    - Outlier detection and removal
    - Roll adjustment for futures contracts
    - Missing data interpolation
    - Data quality scoring
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize DataCleaner with configuration."""
        self.config = self._load_config(config_path)
        self.quality_config = self.config['data']['quality']
        self.fill_gaps = self.quality_config['fill_gaps']
        self.remove_outliers = self.quality_config['remove_outliers']
        self.outlier_threshold = self.quality_config['outlier_threshold']
        self.min_data_points = self.quality_config['min_data_points']
        
        logger.info("DataCleaner initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            raise
    
    def clean_data(
        self, 
        data: pd.DataFrame, 
        symbol: str = "Unknown"
    ) -> pd.DataFrame:
        """
        Apply full cleaning pipeline to data.
        
        Args:
            data: Raw OHLCV data
            symbol: Symbol name for logging
            
        Returns:
            Cleaned DataFrame
        """
        if data.empty:
            logger.warning(f"Empty data provided for {symbol}")
            return data
        
        logger.info(f"Cleaning data for {symbol}: {len(data)} rows")
        
        # Make a copy to avoid modifying original
        cleaned_data = data.copy()
        
        # Step 1: Remove duplicates
        cleaned_data = self._remove_duplicates(cleaned_data, symbol)
        
        # Step 2: Fill gaps if enabled
        if self.fill_gaps:
            cleaned_data = self._fill_price_gaps(cleaned_data, symbol)
        
        # Step 3: Remove outliers if enabled
        if self.remove_outliers:
            cleaned_data = self._remove_outliers_data(cleaned_data, symbol)
        
        # Step 4: Validate OHLC relationships
        cleaned_data = self._fix_ohlc_logic(cleaned_data, symbol)
        
        # Step 5: Handle missing values
        cleaned_data = self._handle_missing_values(cleaned_data, symbol)
        
        # Step 6: Calculate quality score
        quality_score = self._calculate_quality_score(data, cleaned_data)
        logger.info(f"Data quality score for {symbol}: {quality_score:.2f}")
        
        logger.info(f"Cleaning complete for {symbol}: {len(cleaned_data)} rows remaining")
        return cleaned_data
    
    def _remove_duplicates(
        self, 
        data: pd.DataFrame, 
        symbol: str
    ) -> pd.DataFrame:
        """Remove duplicate timestamps, keeping the last occurrence."""
        initial_count = len(data)
        
        # Remove duplicates based on index (timestamp)
        data_clean = data[~data.index.duplicated(keep='last')]
        
        removed_count = initial_count - len(data_clean)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate rows for {symbol}")
        
        return data_clean
    
    def _fill_price_gaps(
        self, 
        data: pd.DataFrame, 
        symbol: str
    ) -> pd.DataFrame:
        """
        Fill gaps in price data using forward fill method.
        
        Args:
            data: OHLCV data with potential gaps
            symbol: Symbol name for logging
            
        Returns:
            Data with gaps filled
        """
        if len(data) < 2:
            return data
        
        # Create complete date range (business days for forex)
        start_date = data.index.min()
        end_date = data.index.max()
        
        # For forex, assume 24/5 trading (Monday to Friday)
        if symbol.endswith('=X'):  # Forex pairs
            complete_range = pd.bdate_range(start=start_date, end=end_date, freq='D')
        else:
            complete_range = pd.bdate_range(start=start_date, end=end_date, freq='B')
        
        # Reindex to complete range
        data_complete = data.reindex(complete_range)
        
        # Count missing values before filling
        missing_before = data_complete.isnull().sum().sum()
        
        if missing_before > 0:
            # Forward fill prices (carry last observation forward)
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in data_complete.columns:
                    data_complete[col] = data_complete[col].fillna(method='ffill')
            
            # Set volume to 0 for filled periods
            if 'volume' in data_complete.columns:
                data_complete['volume'] = data_complete['volume'].fillna(0)
            
            # For gaps at the beginning, use backward fill
            for col in price_cols:
                if col in data_complete.columns:
                    data_complete[col] = data_complete[col].fillna(method='bfill')
            
            missing_after = data_complete.isnull().sum().sum()
            filled_count = missing_before - missing_after
            
            logger.info(f"Filled {filled_count} missing values for {symbol}")
        
        return data_complete.dropna()
    
    def _remove_outliers_data(
        self, 
        data: pd.DataFrame, 
        symbol: str
    ) -> pd.DataFrame:
        """
        Remove outliers using statistical methods.
        
        Args:
            data: OHLCV data
            symbol: Symbol name for logging
            
        Returns:
            Data with outliers removed
        """
        if len(data) < 10:  # Need sufficient data for outlier detection
            return data
        
        # Calculate returns for outlier detection
        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
            
            # Use z-score method for outlier detection
            z_scores = np.abs((returns - returns.mean()) / returns.std())
            outlier_mask = z_scores > self.outlier_threshold
            
            outlier_count = outlier_mask.sum()
            if outlier_count > 0:
                # Remove outlier rows
                outlier_dates = returns[outlier_mask].index
                data_clean = data.drop(outlier_dates)
                
                logger.info(f"Removed {outlier_count} outlier rows for {symbol}")
                return data_clean
        
        return data
    
    def _fix_ohlc_logic(
        self, 
        data: pd.DataFrame, 
        symbol: str
    ) -> pd.DataFrame:
        """
        Fix OHLC logical inconsistencies.
        
        Ensures:
        - High >= Low
        - High >= Open, Close
        - Low <= Open, Close
        """
        price_cols = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in price_cols):
            return data
        
        data_fixed = data.copy()
        fixes_applied = 0
        
        # Fix High < Low issues
        invalid_high_low = data_fixed['high'] < data_fixed['low']
        if invalid_high_low.any():
            # Swap high and low values
            temp_high = data_fixed.loc[invalid_high_low, 'high'].copy()
            data_fixed.loc[invalid_high_low, 'high'] = data_fixed.loc[invalid_high_low, 'low']
            data_fixed.loc[invalid_high_low, 'low'] = temp_high
            fixes_applied += invalid_high_low.sum()
        
        # Fix High < Open/Close issues
        for price_col in ['open', 'close']:
            invalid_mask = data_fixed['high'] < data_fixed[price_col]
            if invalid_mask.any():
                data_fixed.loc[invalid_mask, 'high'] = data_fixed.loc[invalid_mask, price_col]
                fixes_applied += invalid_mask.sum()
        
        # Fix Low > Open/Close issues  
        for price_col in ['open', 'close']:
            invalid_mask = data_fixed['low'] > data_fixed[price_col]
            if invalid_mask.any():
                data_fixed.loc[invalid_mask, 'low'] = data_fixed.loc[invalid_mask, price_col]
                fixes_applied += invalid_mask.sum()
        
        if fixes_applied > 0:
            logger.info(f"Fixed {fixes_applied} OHLC logic violations for {symbol}")
        
        return data_fixed
    
    def _handle_missing_values(
        self, 
        data: pd.DataFrame, 
        symbol: str
    ) -> pd.DataFrame:
        """Handle remaining missing values after gap filling."""
        missing_counts = data.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing == 0:
            return data
        
        logger.warning(f"Found {total_missing} missing values for {symbol}: {missing_counts.to_dict()}")
        
        # Drop rows with any missing critical price data
        price_cols = ['open', 'high', 'low', 'close']
        critical_missing = data[price_cols].isnull().any(axis=1)
        
        if critical_missing.any():
            rows_dropped = critical_missing.sum()
            data_clean = data[~critical_missing]
            logger.info(f"Dropped {rows_dropped} rows with missing price data for {symbol}")
            return data_clean
        
        return data
    
    def _calculate_quality_score(
        self, 
        original_data: pd.DataFrame, 
        cleaned_data: pd.DataFrame
    ) -> float:
        """
        Calculate data quality score (0-100).
        
        Factors:
        - Data completeness
        - Outlier percentage
        - Gap frequency
        - OHLC consistency
        """
        if original_data.empty:
            return 0.0
        
        score = 100.0  # Start with perfect score
        
        # Completeness penalty
        data_retention = len(cleaned_data) / len(original_data)
        if data_retention < 1.0:
            score -= (1.0 - data_retention) * 30  # Up to 30 points penalty
        
        # Missing data penalty
        if not cleaned_data.empty:
            missing_ratio = cleaned_data.isnull().sum().sum() / (len(cleaned_data) * len(cleaned_data.columns))
            score -= missing_ratio * 20  # Up to 20 points penalty
        
        # Minimum data points check
        if len(cleaned_data) < self.min_data_points:
            score -= 25  # 25 points penalty for insufficient data
        
        # OHLC consistency check
        if not cleaned_data.empty and all(col in cleaned_data.columns for col in ['open', 'high', 'low', 'close']):
            # Check for remaining OHLC violations
            high_low_violations = (cleaned_data['high'] < cleaned_data['low']).sum()
            ohlc_violations = (
                (cleaned_data['high'] < cleaned_data['open']) |
                (cleaned_data['high'] < cleaned_data['close']) |
                (cleaned_data['low'] > cleaned_data['open']) |
                (cleaned_data['low'] > cleaned_data['close'])
            ).sum()
            
            violation_ratio = (high_low_violations + ohlc_violations) / len(cleaned_data)
            score -= violation_ratio * 25  # Up to 25 points penalty
        
        return max(0.0, min(100.0, score))
    
    def apply_roll_adjustment(
        self, 
        data: pd.DataFrame, 
        roll_dates: List[str], 
        method: str = 'difference'
    ) -> pd.DataFrame:
        """
        Apply roll adjustments for futures contracts.
        
        Args:
            data: Futures price data
            roll_dates: List of roll dates in 'YYYY-MM-DD' format
            method: 'difference' or 'ratio' adjustment method
            
        Returns:
            Roll-adjusted price data
        """
        if data.empty or not roll_dates:
            return data
        
        adjusted_data = data.copy()
        price_cols = ['open', 'high', 'low', 'close']
        
        for roll_date in roll_dates:
            roll_timestamp = pd.to_datetime(roll_date)
            
            # Find the closest date in our data
            if roll_timestamp not in data.index:
                # Find nearest date
                date_diffs = abs(data.index - roll_timestamp)
                roll_timestamp = data.index[date_diffs.argmin()]
            
            if roll_timestamp not in data.index:
                continue
            
            # Get data before and after roll
            before_roll = adjusted_data.loc[:roll_timestamp]
            after_roll = adjusted_data.loc[roll_timestamp:]
            
            if len(before_roll) < 2 or len(after_roll) < 2:
                continue
            
            # Calculate adjustment factor
            if method == 'difference':
                # Price difference method
                last_old = before_roll['close'].iloc[-2]  # Last price before roll
                first_new = after_roll['close'].iloc[0]   # First price after roll
                adjustment = first_new - last_old
                
                # Apply adjustment to all prices before roll
                for col in price_cols:
                    if col in adjusted_data.columns:
                        adjusted_data.loc[:roll_timestamp, col] += adjustment
            
            elif method == 'ratio':
                # Price ratio method
                last_old = before_roll['close'].iloc[-2]
                first_new = after_roll['close'].iloc[0]
                
                if last_old != 0:
                    adjustment_ratio = first_new / last_old
                    
                    # Apply adjustment to all prices before roll
                    for col in price_cols:
                        if col in adjusted_data.columns:
                            adjusted_data.loc[:roll_timestamp, col] *= adjustment_ratio
        
        logger.info(f"Applied roll adjustments for {len(roll_dates)} roll dates")
        return adjusted_data
    
    def detect_data_gaps(
        self, 
        data: pd.DataFrame, 
        max_gap_days: int = 3
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
        """
        Detect gaps in time series data.
        
        Args:
            data: Time series data
            max_gap_days: Maximum allowed gap in days
            
        Returns:
            List of (start_date, end_date, gap_days) tuples
        """
        if len(data) < 2:
            return []
        
        gaps = []
        dates = data.index.to_series()
        date_diffs = dates.diff().dt.days
        
        # Find gaps larger than threshold
        large_gaps = date_diffs > max_gap_days
        
        if large_gaps.any():
            gap_indices = large_gaps[large_gaps].index
            
            for gap_end in gap_indices:
                gap_start_idx = dates.index.get_loc(gap_end) - 1
                gap_start = dates.iloc[gap_start_idx]
                gap_days = (gap_end - gap_start).days
                
                gaps.append((gap_start, gap_end, gap_days))
        
        return gaps
