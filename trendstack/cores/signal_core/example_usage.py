"""
Example usage of signal_core API for orchestrator

Simple demonstration of how orchestrator calls signal generation with watchlist.
"""

import pandas as pd
from datetime import datetime
from trendstack.cores.signal_core import (
    generate_signals, 
    get_available_strategies, 
    clear_cooldowns,
    get_watchlist,
    get_symbol_strategies,
    get_strategy_timeframe,
    get_symbol_strategy_pairs,
    update_strategy_config
)


def create_sample_data():
    """Create sample OHLCV data."""
    return pd.DataFrame({
        'open': [100, 101, 102, 101, 100, 99, 98, 99, 100, 101],
        'high': [101, 102, 103, 102, 101, 100, 99, 100, 101, 102],
        'low': [99, 100, 101, 100, 99, 98, 97, 98, 99, 100],
        'close': [101, 102, 101, 100, 99, 98, 99, 100, 101, 102],
        'volume': [1000, 1200, 1100, 900, 800, 1300, 1400, 1000, 1100, 1200]
    }, index=pd.date_range('2024-01-01', periods=10, freq='1H'))


def simple_orchestrator_loop():
    """
    Clean orchestrator loop using the new structure.
    """
    print("=== Simple Orchestrator Loop ===")
    print("Each strategy instance has its own timeframe:")
    
    # Super clean main loop - each strategy instance has its own timeframe
    for symbol, strategy_instance, timeframe in get_symbol_strategy_pairs():
        print(f"\nProcessing {symbol} {strategy_instance} ({timeframe})")
        
        # data = data_core.load_prices(symbol, timeframe)  # Real implementation
        data = create_sample_data()  # Mock for example
        
        signal = generate_signals(data, symbol, strategy_instance)  # Returns single signal or None
        
        if signal:
            print(f"  Generated signal: {signal.signal_type.value} (conf: {signal.confidence:.2f})")
            if signal.confidence > 0.7:
                print(f"    HIGH CONFIDENCE - Execute!")
                # execution_core.execute(signal)  # Real implementation
        else:
            print(f"  No signal (cooldown or conditions not met)")


if __name__ == "__main__":
    simple_orchestrator_loop()
    # traditional_orchestrator_loop()
