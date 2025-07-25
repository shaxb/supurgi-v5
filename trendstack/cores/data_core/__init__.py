"""
TrendStack Data Core - Simple and Effective

Design: retrieve data -> clean data -> store data -> give data when asked

Public API (use this):
- load_prices(symbol, frame="D", start=None, end=None) -> DataFrame
- load_costs(symbol) -> CostSpec  
- refresh_symbol(symbol, source=None) -> None
- list_symbols() -> Dict[str, str]

Data maintenance runs independently to keep all data fresh.
"""

# Simple API exports
from .api import load_prices, load_costs, refresh_symbol, list_symbols, CostSpec, Frame

__all__ = [
    'load_prices', 
    'load_costs', 
    'refresh_symbol', 
    'list_symbols',
    'CostSpec',
    'Frame'
]
__version__ = '1.0.0'
