"""
Example usage of the MT5-compatible broker system
"""

from trendstack.cores.services import BrokerAPI, BacktestBroker, MT5Broker


def backtest_example():
    """Example using backtest broker"""
    print("=== Backtest Broker Example ===")
    
    # Create and set backtest broker
    backtest_broker = BacktestBroker(initial_balance=10000.0, leverage=100)
    BrokerAPI.set_broker(backtest_broker)
    
    # Set some sample prices
    backtest_broker.set_price("EURUSD", 1.0950, 1.0952)
    
    # Check account
    account = BrokerAPI.account_info()
    print(f"Balance: ${account.balance:.2f}")
    print(f"Equity: ${account.equity:.2f}")
    
    # Place a buy order
    request = {
        "action": 1,  # TRADE_ACTION_DEAL
        "symbol": "EURUSD",
        "volume": 0.1,
        "type": 0,    # ORDER_TYPE_BUY
    }
    
    result = BrokerAPI.order_send(request)
    print(f"Order result: {result.retcode} - {result.comment}")
    
    # Check positions
    positions = BrokerAPI.positions_get()
    print(f"Open positions: {len(positions)}")
    for pos in positions:
        print(f"  {pos.symbol}: {pos.volume} @ {pos.price_open}")
    
    # Update price and check profit
    backtest_broker.set_price("EURUSD", 1.0960, 1.0962)
    account = BrokerAPI.account_info()
    print(f"New equity: ${account.equity:.2f}")


def live_example():
    """Example using live MT5 broker (requires MT5)"""
    print("\n=== Live MT5 Broker Example ===")
    
    # Create and set MT5 broker (without credentials - assumes already logged in)
    mt5_broker = MT5Broker()
    
    if not mt5_broker.is_connected():
        print("MT5 not available or not connected")
        return
    
    BrokerAPI.set_broker(mt5_broker)
    
    # Check account
    account = BrokerAPI.account_info()
    if account:
        print(f"Balance: ${account.balance:.2f}")
        print(f"Equity: ${account.equity:.2f}")
        print(f"Free margin: ${account.free_margin:.2f}")
    
    # Check positions
    positions = BrokerAPI.positions_get()
    print(f"Open positions: {len(positions)}")
    for pos in positions:
        print(f"  {pos.symbol}: {pos.volume} @ {pos.price_open} (P&L: ${pos.profit:.2f})")
    
    # Check pending orders
    orders = BrokerAPI.orders_get()
    print(f"Pending orders: {len(orders)}")


if __name__ == "__main__":
    # Run backtest example
    backtest_example()
    
    # Run live example (will show "not available" if MT5 not installed)
    live_example()
