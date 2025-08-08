"""
Backtest Core - Broker-like simulation engine

Main API:
    BrokerEngine - Realistic broker simulation with margin, SL/TP, position sizing
"""

from .engine import BrokerEngine
from .account import BrokerAccount  
from .position import Position
from .results import BacktestResults

__all__ = [
    'BrokerEngine',
    'BrokerAccount', 
    'Position',
    'BacktestResults'
]
