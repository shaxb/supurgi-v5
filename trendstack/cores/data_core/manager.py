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

def get_data(symbol: str, frame: Frame = "D", 
             start: Optional[str] = None, 
             end: Optional[str] = None) -> pd.DataFrame:
    """Smart data loader - gets exactly what you need, when you need it.
    
    Automatically handles:
    - Checking if data exists for the requested range
    - Downloading only missing data
    - Cleaning and caching data
    - Returning the exact range requested
    """
    
    processed_file = DATA_PATH / "processed" / f"{symbol.replace('=', '_')}_{frame}.parquet"
    
    # Load existing data if available
    existing_data = pd.DataFrame()
    if processed_file.exists():
        existing_data = pd.read_parquet(processed_file)
        logger.info(f"📁 Found cached data for {symbol}: {len(existing_data)} bars")
    
    # Determine what data we need
    if start is None and end is None:
        # No specific range - need all data up to today
        needed_start = None
        needed_end = pd.Timestamp.now().strftime('%Y-%m-%d')
    else:
        needed_start = start
        needed_end = end
    
    # Check if we have the needed data
    needs_update = False
    
    if existing_data.empty:
        logger.info(f"📥 No data for {symbol}, downloading...")
        needs_update = True
    else:
        # Check if requested range is covered
        data_start = existing_data.index[0]
        data_end = existing_data.index[-1]
        
        # Remove timezone for comparison
        if hasattr(data_start, 'tz') and data_start.tz is not None:
            data_start = data_start.tz_localize(None)
        if hasattr(data_end, 'tz') and data_end.tz is not None:
            data_end = data_end.tz_localize(None)
        
        # Check if we need more recent data
        if needed_end:
            requested_end = pd.Timestamp(needed_end)
            if data_end < requested_end:
                logger.info(f"� Need newer data: have until {data_end.date()}, need until {requested_end.date()}")
                needs_update = True
        
        # Check if we need older data  
        if needed_start:
            requested_start = pd.Timestamp(needed_start)
            if data_start > requested_start:
                logger.info(f"� Need older data: have from {data_start.date()}, need from {requested_start.date()}")
                needs_update = True
    
    # Update data if needed
    if needs_update:
        refresh_data(symbol, frame)
        if processed_file.exists():
            existing_data = pd.read_parquet(processed_file)
            logger.info(f"✅ Updated data for {symbol}: {len(existing_data)} bars")
    
    # Filter to requested range
    if not existing_data.empty:
        if start:
            existing_data = existing_data[existing_data.index >= start]
        if end:
            existing_data = existing_data[existing_data.index <= end]
        
        if start or end:
            logger.info(f"🔍 Filtered to requested range: {len(existing_data)} bars")
    
    return existing_data

def refresh_data(symbol: str, frame: Frame = "D", source: Optional[str] = None) -> None:
    """Smart data updater - only downloads what's missing."""
    
    if source is None:
        source = SYMBOLS.get(symbol, "yfinance")
    
    # File paths
    raw_file = DATA_PATH / "raw" / f"{symbol.replace('=', '_')}_{frame}.parquet"
    processed_file = DATA_PATH / "processed" / f"{symbol.replace('=', '_')}_{frame}.parquet"
    
    logger.info(f"🔄 Refreshing data for {symbol} ({frame})...")
    
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
        new_data = download_raw_data(symbol, frame=frame, start=start_date)
        
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
        # Use shorter period for intraday data due to API limits
        period = "60d" if frame in ["H4", "H1"] else "5y"
        combined_raw = download_raw_data(symbol, frame=frame, period=period)
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
