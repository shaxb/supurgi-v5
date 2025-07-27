import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

def download_raw_data(symbol: str, frame: str = "D", period: str = "5y", start: str = None) -> pd.DataFrame:
    """Download raw OHLCV data from yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        
        # Map timeframes to yfinance intervals
        interval_map = {"D": "1d", "H4": "4h", "H1": "1h"}
        interval = interval_map.get(frame, "1d")
        
        if start:
            # Download from specific start date to today
            data = ticker.history(start=start, end=datetime.now().strftime('%Y-%m-%d'), interval=interval)
        else:
            # Download full period
            data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            logger.warning(f"No data found for {symbol}")
            return pd.DataFrame()
        
        # Standardize column names
        data.columns = [col.lower() for col in data.columns]
        
        # Remove timezone for simplicity
        if hasattr(data.index, 'tz') and data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        logger.info(f"Downloaded {len(data)} bars for {symbol} ({frame})")
        return data
        
    except Exception as e:
        logger.error(f"Failed to download {symbol}: {e}")
        return pd.DataFrame()
