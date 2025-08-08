"""
Example usage of the broker backtesting engine
"""

from trendstack.cores.backtest_core import BrokerEngine

def run_example_backtest():
    """Simple example of running a backtest."""
    
    # Create engine from config
    engine = BrokerEngine.from_config('trendstack/cores/backtest_core/backtest_config.yaml')
    
    # Run backtest
    results = engine.run_backtest()
    
    # Results are automatically printed, but you can also:
    print("\nTrade Details:")
    trades_df = results.to_dataframe()
    if not trades_df.empty:
        print(trades_df.head(10))
    
    print(f"\nFinal Summary:")
    summary = results.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    run_example_backtest()
