"""
Signal Core - Trading Signal Generation System

Core components:
- Signal generation strategies (momentum, mean reversion, etc.)
- Signal filtering and processing
- Strategy plugin management
- Signal intent data types

Public API:
- generate_signals(data, strategy, params) -> SignalIntent[]
- register_strategy(name, strategy_class)
- list_strategies() -> List[str]
"""

from .api import generate_signals, register_strategy, list_strategies
from .datatypes import SignalIntent, SignalType, SignalStrength
from .momentum import MomentumStrategy
from .filters import EMAFilter, VolatilityRegimeFilter

__all__ = [
    'generate_signals',
    'register_strategy', 
    'list_strategies',
    'SignalIntent',
    'SignalType',
    'SignalStrength',
    'MomentumStrategy',
    'EMAFilter',
    'VolatilityRegimeFilter'
]

__version__ = '1.0.0'
