"""
Live MT5 Broker - Real MetaTrader 5 broker implementation
"""

from typing import List, Dict, Any, Optional
import logging

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logging.warning("MetaTrader5 module not available - live trading disabled")


class MT5Broker:
    """Simple but effective live MT5 broker"""
    
    def __init__(self, path: str = None, login: int = None, password: str = None, server: str = None):
        self.path = path
        self.login = login 
        self.password = password
        self.server = server
        self.connected = False
        
        if not MT5_AVAILABLE:
            logging.error("MT5 module not available - cannot create MT5Broker")
            return
            
        # Initialize MT5
        if not mt5.initialize(path=self.path):
            logging.error(f"MT5 initialize failed: {mt5.last_error()}")
            return
            
        # Login if credentials provided
        if self.login and self.password and self.server:
            if mt5.login(self.login, self.password, self.server):
                self.connected = True
                logging.info("MT5 login successful")
            else:
                logging.error(f"MT5 login failed: {mt5.last_error()}")
        else:
            # Assume already logged in
            self.connected = True
            logging.info("MT5 initialized (using existing login)")
    
    def __del__(self):
        """Cleanup MT5 connection"""
        if MT5_AVAILABLE and self.connected:
            mt5.shutdown()
    
    def account_info(self):
        """Get account info - direct MT5 call"""
        if not self.connected or not MT5_AVAILABLE:
            return None
        return mt5.account_info()
    
    def positions_get(self, symbol: str = None):
        """Get positions - direct MT5 call"""
        if not self.connected or not MT5_AVAILABLE:
            return []
        
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
            
        return list(positions) if positions else []
    
    def order_send(self, request: Dict[str, Any]):
        """Send order - direct MT5 call"""
        if not self.connected or not MT5_AVAILABLE:
            return None
        
        # Convert dict to MT5 request if needed
        if isinstance(request, dict):
            # MT5 expects specific format
            mt5_request = {
                "action": request.get("action", 1),  # TRADE_ACTION_DEAL
                "symbol": request.get("symbol", ""),
                "volume": float(request.get("volume", 0.0)),
                "type": request.get("type", 0),  # ORDER_TYPE_BUY
                "price": float(request.get("price", 0.0)) if request.get("price") else None,
                "sl": float(request.get("sl", 0.0)) if request.get("sl") else 0.0,
                "tp": float(request.get("tp", 0.0)) if request.get("tp") else 0.0,
                "deviation": request.get("deviation", 10),
                "magic": request.get("magic", 0),
                "comment": request.get("comment", ""),
                "type_time": request.get("type_time", mt5.ORDER_TIME_GTC if MT5_AVAILABLE else 0),
                "type_filling": request.get("type_filling", mt5.ORDER_FILLING_IOC if MT5_AVAILABLE else 0)
            }
            
            # Remove None values
            mt5_request = {k: v for k, v in mt5_request.items() if v is not None}
            
            return mt5.order_send(mt5_request)
        else:
            return mt5.order_send(request)
    
    def orders_get(self):
        """Get pending orders - direct MT5 call"""
        if not self.connected or not MT5_AVAILABLE:
            return []
        
        orders = mt5.orders_get()
        return list(orders) if orders else []
    
    def copy_rates_from_pos(self, symbol: str, timeframe, start_pos: int, count: int):
        """Get historical data - MT5 convenience method"""
        if not self.connected or not MT5_AVAILABLE:
            return None
        return mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)
    
    def symbol_info_tick(self, symbol: str):
        """Get current tick - MT5 convenience method"""
        if not self.connected or not MT5_AVAILABLE:
            return None
        return mt5.symbol_info_tick(symbol)
    
    def is_connected(self) -> bool:
        """Check if connected to MT5"""
        return self.connected and MT5_AVAILABLE


# MT5 Constants (if MT5 not available, provide fallbacks)
if MT5_AVAILABLE:
    # Trade actions
    TRADE_ACTION_DEAL = mt5.TRADE_ACTION_DEAL
    TRADE_ACTION_PENDING = mt5.TRADE_ACTION_PENDING
    TRADE_ACTION_SLTP = mt5.TRADE_ACTION_SLTP
    TRADE_ACTION_MODIFY = mt5.TRADE_ACTION_MODIFY
    TRADE_ACTION_REMOVE = mt5.TRADE_ACTION_REMOVE
    TRADE_ACTION_CLOSE_BY = mt5.TRADE_ACTION_CLOSE_BY
    
    # Order types
    ORDER_TYPE_BUY = mt5.ORDER_TYPE_BUY
    ORDER_TYPE_SELL = mt5.ORDER_TYPE_SELL
    ORDER_TYPE_BUY_LIMIT = mt5.ORDER_TYPE_BUY_LIMIT
    ORDER_TYPE_SELL_LIMIT = mt5.ORDER_TYPE_SELL_LIMIT
    ORDER_TYPE_BUY_STOP = mt5.ORDER_TYPE_BUY_STOP
    ORDER_TYPE_SELL_STOP = mt5.ORDER_TYPE_SELL_STOP
    
    # Timeframes
    TIMEFRAME_M1 = mt5.TIMEFRAME_M1
    TIMEFRAME_M5 = mt5.TIMEFRAME_M5
    TIMEFRAME_M15 = mt5.TIMEFRAME_M15
    TIMEFRAME_M30 = mt5.TIMEFRAME_M30
    TIMEFRAME_H1 = mt5.TIMEFRAME_H1
    TIMEFRAME_H4 = mt5.TIMEFRAME_H4
    TIMEFRAME_D1 = mt5.TIMEFRAME_D1
else:
    # Fallback constants
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_PENDING = 0
    TRADE_ACTION_SLTP = 2
    TRADE_ACTION_MODIFY = 3
    TRADE_ACTION_REMOVE = 4
    TRADE_ACTION_CLOSE_BY = 5
    
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TYPE_BUY_LIMIT = 2
    ORDER_TYPE_SELL_LIMIT = 3
    ORDER_TYPE_BUY_STOP = 4
    ORDER_TYPE_SELL_STOP = 5
    
    TIMEFRAME_M1 = 1
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_M30 = 30
    TIMEFRAME_H1 = 16385
    TIMEFRAME_H4 = 16388
    TIMEFRAME_D1 = 16408
