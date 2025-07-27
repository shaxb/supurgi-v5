"""
TrendStack Data Core - Simple and Effective

Design: One smart API that handles everything automatically

Public API:
- load_prices(symbol, frame="D", start=None, end=None) -> DataFrame  # Smart loader - handles everything
- load_costs(symbol) -> CostSpec

Just call load_prices() with what you want - it figures out the rest.
"""

# Simple API exports
from .api import load_prices, load_costs, CostSpec, Frame

__all__ = [
    'load_prices', 
    'load_costs', 
    'CostSpec',
    'Frame'
]
__version__ = '1.0.0'
