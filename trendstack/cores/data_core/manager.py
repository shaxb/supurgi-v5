# Simple data manager - retrieve, clean, store, serve
import os
import yaml
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict
from loguru import logger

from .api import Frame, CostSpec
from .config import CONFIG
from .ingestion import download_raw_data
from .cleaner import clean_data

DATA_PATH = Path(CONFIG['data_path'])
DATA_PATH.mkdir(exist_ok=True)
(DATA_PATH / "raw").mkdir(exist_ok=True)
(DATA_PATH / "processed").mkdir(exist_ok=True)

# Load symbols once  
_symbols_path = Path(__file__).parent / "symbols.yaml"
with open(_symbols_path) as f:
    SYMBOLS = yaml.safe_load(f)

# Load costs once
_costs_path = Path(__file__).parent / "costs.yaml" 
with open(_costs_path) as f:
    COSTS = yaml.safe_load(f)

def get_cleaned_data(symbol: str, frame: Frame = "D", 
                    start: Optional[str] = None, 
                    end: Optional[str] = None) -> pd.DataFrame:
    """Return cleaned OHLCV data."""
    
    # Check if processed file exists
    processed_file = DATA_PATH / "processed" / f"{symbol.replace('=', '_')}_{frame}.parquet"
    
    if processed_file.exists():
        logger.info(f"📁 Using cached data for {symbol} from {processed_file}")
        data = pd.read_parquet(processed_file)
        
        # Filter by date range if provided
        if start or end:
            original_len = len(data)
            if start:
                data = data[data.index >= start]
            if end:
                data = data[data.index <= end]
            logger.info(f"🔍 Filtered data: {original_len} -> {len(data)} bars")
        
        return data
    
    # If no processed file, create it
    logger.info(f"📥 No cached data found for {symbol}, downloading...")
    refresh_data(symbol)
    
    # Try again
    if processed_file.exists():
        data = pd.read_parquet(processed_file)
        if start or end:
            if start:
                data = data[data.index >= start]
            if end:
                data = data[data.index <= end]
        return data
    
    # Return empty if still nothing
    return pd.DataFrame()

def refresh_data(symbol: str, source: Optional[str] = None) -> None:
    """Incrementally update data - only download missing tail."""
    
    if source is None:
        source = SYMBOLS.get(symbol, "yfinance")
    
    # File paths
    raw_file = DATA_PATH / "raw" / f"{symbol.replace('=', '_')}.parquet"
    processed_file = DATA_PATH / "processed" / f"{symbol.replace('=', '_')}_D.parquet"
    
    logger.info(f"🔄 Refreshing data for {symbol}...")
    
    # Check for existing raw data
    existing_raw = pd.DataFrame()
    last_date = None
    
    if raw_file.exists():
        existing_raw = pd.read_parquet(raw_file)
        if not existing_raw.empty:
            last_date = existing_raw.index[-1]
            logger.info(f"📅 Found existing data for {symbol} up to {last_date.date()}")
        else:
            logger.info(f"📄 Empty raw file found for {symbol}")
    else:
        logger.info(f"🆕 No existing data for {symbol}")
    
    # Download new data
    if last_date:
        # Incremental: get only new data from last_date + 1
        start_date = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        logger.info(f"📡 Downloading incremental data from {start_date}...")
        new_data = download_raw_data(symbol, start=start_date)
        
        if not new_data.empty:
            # Append new data
            logger.info(f"➕ Appending {len(new_data)} new bars to existing {len(existing_raw)} bars")
            combined_raw = pd.concat([existing_raw, new_data])
            combined_raw = combined_raw[~combined_raw.index.duplicated(keep='last')]
            combined_raw = combined_raw.sort_index()
            logger.info(f"✅ Added {len(new_data)} new bars for {symbol}")
        else:
            logger.info(f"ℹ️  No new data available for {symbol} (normal for weekends/holidays)")
            combined_raw = existing_raw
    else:
        # First time: download full history
        logger.info(f"📡 Downloading full history for {symbol}...")
        combined_raw = download_raw_data(symbol, period="5y")
        if not combined_raw.empty:
            logger.info(f"📊 Downloaded full history for {symbol}: {len(combined_raw)} bars")
    
    if combined_raw.empty:
        logger.warning(f"⚠️  No data available for {symbol}")
        return
    
    # Save raw data (truth copy)
    logger.info(f"💾 Saving raw data: {len(combined_raw)} bars")
    combined_raw.to_parquet(raw_file)
    
    # Clean and save processed data
    logger.info(f"🧹 Cleaning data...")
    clean_data_result = clean_data(combined_raw)
    logger.info(f"💾 Saving processed data: {len(clean_data_result)} bars")
    clean_data_result.to_parquet(processed_file)
    
    logger.info(f"✅ Updated {symbol}: {len(clean_data_result)} total bars")

def get_symbol_mapping() -> Dict[str, str]:
    """Return symbol -> source mapping."""
    return SYMBOLS.copy()
