"""
BrokerAPI - Global broker abstraction
"""

from typing import List, Dict, Any


class BrokerAPI:
    """Simple global broker API - can connect to real or fake broker"""
    
    _broker = None
    
    @staticmethod
    def get_balance():
        """Get account balance"""
        if BrokerAPI._broker:
            return BrokerAPI._broker.get_balance()
        return 0.0
    
    @staticmethod
    def get_positions():
        """Get current positions"""
        if BrokerAPI._broker:
            return BrokerAPI._broker.get_positions()
        return []
    
    @staticmethod
    def place_order(symbol: str, size: float, order_type: str = "market"):
        """Place an order"""
        if BrokerAPI._broker:
            return BrokerAPI._broker.place_order(symbol, size, order_type)
        return None
    
    @staticmethod
    def close_position(position_id: str):
        """Close a position"""
        if BrokerAPI._broker:
            return BrokerAPI._broker.close_position(position_id)
        return None
    
    @staticmethod
    def set_broker(broker):
        """Set the broker implementation"""
        BrokerAPI._broker = broker
        
    @staticmethod
    def is_connected():
        """Check if broker is connected"""
        return BrokerAPI._broker is not None
