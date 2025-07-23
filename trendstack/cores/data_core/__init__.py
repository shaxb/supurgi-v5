"""
TrendStack Data Core - Market Data Pipeline

This module handles market data ingestion, cleaning, and preprocessing
for the TrendStack algorithmic trading system.

Modules:
- ingestion: Download and append raw market data
- cleaner: Gap filling, outlier removal, and roll adjustments  
- costs: Trading cost calculations (spreads, commissions, swaps)
"""

from .ingestion import DataIngestion
from .cleaner import DataCleaner
from .costs import CostCalculator

__all__ = ['DataIngestion', 'DataCleaner', 'CostCalculator']
__version__ = '0.1.0'
