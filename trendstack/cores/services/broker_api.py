"""
MT5-compatible BrokerAPI - Simple global broker that mimics MetaTrader 5
"""


class BrokerAPI:
    """Global broker API that mimics MT5's interface - simple but effective"""
    
    _broker = None
    
    @staticmethod
    def set_broker(broker):
        """Set the broker implementation (MT5 or backtest)"""
        BrokerAPI._broker = broker
    
    @staticmethod
    def account_info():
        """Get account info - mimics mt5.account_info()"""
        if BrokerAPI._broker:
            return BrokerAPI._broker.account_info()
        return None
    
    @staticmethod
    def positions_get(symbol=None):
        """Get positions - mimics mt5.positions_get()"""
        if BrokerAPI._broker:
            return BrokerAPI._broker.positions_get(symbol)
        return []
    
    @staticmethod
    def order_send(request):
        """Send order - mimics mt5.order_send()"""
        if BrokerAPI._broker:
            return BrokerAPI._broker.order_send(request)
        return None
    
    @staticmethod
    def orders_get():
        """Get pending orders - mimics mt5.orders_get()"""  
        if BrokerAPI._broker:
            return BrokerAPI._broker.orders_get()
        return []
