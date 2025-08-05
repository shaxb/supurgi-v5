"""
Signal Core API - Simple interface for orchestrator
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

from .datatypes import SignalIntent, StrategyConfig
from .registry import StrategyRegistry
from .symbols import get_strategy_config


# Global strategy registry
_strategy_registry = StrategyRegistry()

# Strategy cooldown tracking
_strategy_cooldowns = {}


def generate_signals(
    data: pd.DataFrame, 
    symbol: str,
    strategy_instance: str,
    force: bool = False
) -> Optional[SignalIntent]:
    """
    Generate trading signal for orchestrator.
    
    Args:
        data: OHLCV DataFrame with columns ['open', 'high', 'low', 'close']
        symbol: Symbol identifier (e.g., 'AAPL', 'EURUSD')
        strategy_instance: Strategy instance name (e.g., 'momentum_H4', 'momentum_D')
        force: Skip cooldown check if True
        
    Returns:
        Latest SignalIntent or None if no signal/on cooldown
    """
    
    # Extract base strategy name from instance (e.g., 'momentum_H4' -> 'momentum')
    strategy = strategy_instance.split('_')[0]
    
    # Check strategy cooldown unless forced
    if not force and _is_on_cooldown(strategy_instance, symbol):
        return None
    
    try:
        # Get strategy from registry
        strategy_class = _strategy_registry.get_strategy(strategy)
        if strategy_class is None:
            logger.warning(f"Strategy '{strategy}' not found")
            return None
        
        # Get config directly from symbols.yaml
        symbol_config = get_strategy_config(symbol, strategy_instance)
        
        strategy_config = StrategyConfig(
            name=strategy,
            parameters=symbol_config
        )
        
        # Initialize and run strategy
        strategy_instance_obj = strategy_class(strategy_config)
        signal = strategy_instance_obj.generate_signals(data, symbol)
        
        # Return the signal if generated
        if signal:
            _update_cooldown(strategy_instance, symbol, strategy_instance_obj.get_cooldown())
            logger.debug(f"Generated signal for {symbol} using {strategy_instance}")
            return signal
        
        return None
        
    except Exception as e:
        logger.error(f"Signal generation failed for {symbol}: {e}")
        return None


def _is_on_cooldown(strategy: str, symbol: str) -> bool:
    """Check if strategy is on cooldown for symbol."""
    key = f"{strategy}_{symbol}"
    if key not in _strategy_cooldowns:
        return False
    
    cooldown_until = _strategy_cooldowns[key]
    return datetime.now() < cooldown_until


def _update_cooldown(strategy: str, symbol: str, cooldown_minutes: int):
    """Update strategy cooldown for symbol."""
    if cooldown_minutes > 0:
        key = f"{strategy}_{symbol}"
        _strategy_cooldowns[key] = datetime.now() + timedelta(minutes=cooldown_minutes)


def get_available_strategies() -> List[str]:
    """Get list of available strategy names."""
    return [info['name'] for info in _strategy_registry.list_strategies()]


def clear_cooldowns():
    """Clear all strategy cooldowns - for testing/debugging."""
    global _strategy_cooldowns
    _strategy_cooldowns = {}


# Expose symbols functions for orchestrator
from .symbols import get_symbol_strategy_pairs, update_strategy_config
