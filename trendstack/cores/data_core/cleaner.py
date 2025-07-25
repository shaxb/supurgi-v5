import pandas as pd
from loguru import logger

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean OHLCV data - remove duplicates, fill gaps, remove outliers."""
    if data.empty:
        return data
    
    original_len = len(data)
    
    # Remove duplicates
    data = data[~data.index.duplicated(keep='first')]
    
    # Remove rows with all NaN values
    data = data.dropna(how='all')
    
    # Forward fill missing values (limited)
    data = data.ffill(limit=3)
    
    # Remove extreme outliers (price jumps > 50%)
    for col in ['open', 'high', 'low', 'close']:
        if col in data.columns:
            pct_change = data[col].pct_change().abs()
            outliers = pct_change > 0.5
            if outliers.any():
                logger.warning(f"Removing {outliers.sum()} outlier rows for {col}")
                data = data[~outliers]
    
    # Ensure OHLC logic (high >= low, high >= open/close, low <= open/close)
    if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
        # Fix high/low inconsistencies
        data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1)
        data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1)
    
    cleaned_len = len(data)
    if cleaned_len < original_len:
        logger.info(f"Cleaned data: {original_len} -> {cleaned_len} rows")
    
    return data
