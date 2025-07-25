# Data maintenance - runs independently to keep all data fresh
import time
import schedule
from datetime import datetime
from loguru import logger
from .manager import get_symbol_mapping, refresh_data
from .config import CONFIG

def update_all_symbols():
    """Update all symbols with latest data."""
    symbols = list(get_symbol_mapping().keys())
    
    logger.info(f"Starting update for {len(symbols)} symbols")
    
    for symbol in symbols:
        try:
            refresh_data(symbol)
            logger.info(f"Updated {symbol}")
        except Exception as e:
            logger.error(f"Failed to update {symbol}: {e}")
    
    logger.info("Completed symbol updates")

def start_maintenance():
    """Start the data maintenance scheduler."""
    
    # Schedule updates based on config
    update_hours = CONFIG.get('update_interval_hours', 24)
    
    schedule.every(update_hours).hours.do(update_all_symbols)
    
    logger.info(f"Data maintenance started - updates every {update_hours} hours")
    
    # Run initial update
    update_all_symbols()
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    start_maintenance()
