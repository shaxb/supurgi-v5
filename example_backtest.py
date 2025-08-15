"""
Clean, simple backtest example
"""

from trendstack.cores.backtest_core.engine import BrokerEngine
from loguru import logger

def run_example_backtest():
    """Run the simplified backtest."""
    
    # Create self-initializing engine
    engine = BrokerEngine()
    
    # Run backtest (everything is self-contained)
    results = engine.run_backtest()
    
    # Optional: Show trade details
    if results.trades:
        logger.info(f"First trade: {results.trades[0]}")
    
    return results

if __name__ == "__main__":
    run_example_backtest()
