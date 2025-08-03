"""
Signal Core - Simple signal generation for orchestrator

Main API:
    generate_signals() - Generate signals with cooldown management
"""

# Import strategies to trigger auto-registration
from . import momentum

# Public API - only expose what orchestrator needs
from .api import generate_signals, get_available_strategies, clear_cooldowns, get_watchlist, get_symbol_strategies, get_strategy_timeframe, get_symbol_strategy_pairs, update_strategy_config

# For strategy development only
from .datatypes import SignalIntent, SignalType, SignalStrength, StrategyConfig
from .registry import BaseStrategy


__all__ = [
    # Main API for orchestrator
    'generate_signals',
    'get_available_strategies', 
    'clear_cooldowns',
    'get_watchlist',
    'get_symbol_strategies',
    'get_strategy_timeframe',
    'get_symbol_strategy_pairs',
    'update_strategy_config',
    
    # For strategy development
    'SignalIntent',
    'SignalType', 
    'SignalStrength',
    'StrategyConfig',
    'BaseStrategy'
]
