"""
DataService - Global data provider abstraction
"""

import pandas as pd
from typing import Optional


class DataService:
    """Simple global data service - can provide live or historical data"""
    
    _data_provider = None
    
    @staticmethod
    def get_data(symbol: str, timeframe: str = "H4", bars: int = 100):
        """Get price data for symbol"""
        if DataService._data_provider:
            return DataService._data_provider.get_data(symbol, timeframe, bars)
        return pd.DataFrame()  # Empty dataframe if no provider
    
    @staticmethod
    def get_current_price(symbol: str):
        """Get current price for symbol"""
        if DataService._data_provider:
            return DataService._data_provider.get_current_price(symbol)
        return None
    
    @staticmethod
    def set_provider(provider):
        """Set the data provider implementation"""
        DataService._data_provider = provider
        
    @staticmethod
    def is_connected():
        """Check if data provider is connected"""
        return DataService._data_provider is not None
