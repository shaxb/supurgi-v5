"""
Simple orchestrator - main trading loop
"""

import pandas as pd
from datetime import datetime
from loguru import logger

from trendstack.cores.signal_core import generate_signals, get_symbol_strategy_pairs
from trendstack.cores.data_core import load_prices


def run_orchestrator():
    """Main orchestrator loop - simple and effective."""
    
    logger.info("Orchestrator starting...")
    
    try:
        # Get all active symbol-strategy-timeframe combinations
        pairs = get_symbol_strategy_pairs()
        logger.info(f"Processing {len(pairs)} symbol-strategy combinations")
        
        # Main loop
        for symbol, strategy_instance, timeframe in pairs:
            logger.debug(f"Processing {symbol} {strategy_instance} ({timeframe})")
            
            try:
                # Load data for this symbol and timeframe
                data = load_prices(symbol, frame=timeframe)
                
                if data.empty:
                    logger.warning(f"No data for {symbol} {timeframe}")
                    continue
                
                # Generate signal
                signal = generate_signals(data, symbol, strategy_instance)
                
                if signal:
                    logger.info(f"SIGNAL: {symbol} {signal.signal_type.value} (confidence: {signal.confidence:.2f})")
                    
                    # TODO: Send signal to execution core
                    # execution_core.execute(signal)
                    
                else:
                    logger.debug(f"No signal for {symbol} {strategy_instance}")
                    
            except Exception as e:
                logger.error(f"Error processing {symbol} {strategy_instance}: {e}")
                continue
        
        logger.info("Orchestrator cycle completed")
        
    except Exception as e:
        logger.error(f"Orchestrator error: {e}")


if __name__ == "__main__":
    # For testing orchestrator directly
    run_orchestrator()
