"""
Simplified Backtest Core

Clean, self-initializing broker simulation
"""

from .engine import BrokerEngine
from .results import BacktestResults

__all__ = [
    'BrokerEngine',
    'BrokerAccount', 
    'Position',
    'BacktestResults'
]
