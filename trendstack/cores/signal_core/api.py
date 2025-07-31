"""
Signal Core Public API

Main interface for signal generation used by orchestrator and backtesting systems.
"""

from typing import List, Dict, Any, Optional, Type
import pandas as pd
from datetime import datetime
from loguru import logger

from .datatypes import SignalIntent, StrategyConfig, SignalBuffer
from .registry import StrategyRegistry


# Global strategy registry
_strategy_registry = StrategyRegistry()


def generate_signals(
    data: pd.DataFrame, 
    strategy: str, 
    params: Optional[Dict[str, Any]] = None,
    symbol: str = "UNKNOWN"
) -> List[SignalIntent]:
    """
    Generate trading signals using specified strategy.
    
    Args:
        data: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        strategy: Name of registered strategy
        params: Strategy parameters (optional)
        symbol: Symbol identifier
        
    Returns:
        List of SignalIntent objects
        
    Example:
        >>> data = load_prices('AAPL')
        >>> signals = generate_signals(data, 'momentum', {'period': 14})
    """
    try:
        # Get strategy from registry
        strategy_class = _strategy_registry.get_strategy(strategy)
        if strategy_class is None:
            raise ValueError(f"Strategy '{strategy}' not found. Available: {list_strategies()}")
        
        # Create strategy config
        config = StrategyConfig(
            name=strategy,
            parameters=params or {}
        )
        
        # Initialize strategy
        strategy_instance = strategy_class(config)
        
        # Generate signals
        signals = strategy_instance.generate_signals(data, symbol)
        
        logger.info(f"Generated {len(signals)} signals using {strategy} strategy for {symbol}")
        return signals
        
    except Exception as e:
        logger.error(f"Error generating signals with {strategy}: {e}")
        return []


def register_strategy(name: str, strategy_class: Type) -> bool:
    """
    Register a new signal generation strategy.
    
    Args:
        name: Strategy name for lookup
        strategy_class: Strategy class implementing generate_signals method
        
    Returns:
        True if registered successfully
        
    Example:
        >>> register_strategy('my_strategy', MyStrategyClass)
    """
    try:
        _strategy_registry.register(name, strategy_class)
        logger.info(f"Registered strategy: {name}")
        return True
    except Exception as e:
        logger.error(f"Failed to register strategy {name}: {e}")
        return False


def list_strategies() -> List[str]:
    """
    Get list of available strategy names.
    
    Returns:
        List of registered strategy names
    """
    return _strategy_registry.list_strategies()


def get_strategy_info(strategy_name: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a registered strategy.
    
    Args:
        strategy_name: Name of the strategy
        
    Returns:
        Dictionary with strategy metadata or None if not found
    """
    return _strategy_registry.get_strategy_info(strategy_name)


def create_signal_buffer(max_size: int = 1000) -> SignalBuffer:
    """
    Create a new signal buffer for collecting signals over time.
    
    Args:
        max_size: Maximum number of signals to keep in buffer
        
    Returns:
        SignalBuffer instance
    """
    return SignalBuffer(max_size)


def batch_generate_signals(
    data_dict: Dict[str, pd.DataFrame],
    strategy: str,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, List[SignalIntent]]:
    """
    Generate signals for multiple symbols at once.
    
    Args:
        data_dict: Dictionary mapping symbol -> OHLCV DataFrame
        strategy: Strategy name
        params: Strategy parameters
        
    Returns:
        Dictionary mapping symbol -> List[SignalIntent]
    """
    results = {}
    
    for symbol, data in data_dict.items():
        try:
            signals = generate_signals(data, strategy, params, symbol)
            results[symbol] = signals
        except Exception as e:
            logger.error(f"Failed to generate signals for {symbol}: {e}")
            results[symbol] = []
    
    total_signals = sum(len(signals) for signals in results.values())
    logger.info(f"Batch generated {total_signals} signals for {len(data_dict)} symbols")
    
    return results


def validate_signal_data(data: pd.DataFrame) -> bool:
    """
    Validate that DataFrame has required columns for signal generation.
    
    Args:
        data: OHLCV DataFrame
        
    Returns:
        True if data is valid
    """
    required_columns = ['open', 'high', 'low', 'close']
    
    if not isinstance(data, pd.DataFrame):
        logger.error("Data must be a pandas DataFrame")
        return False
    
    if data.empty:
        logger.error("Data DataFrame is empty")
        return False
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Check for NaN values in critical columns
    for col in required_columns:
        if data[col].isna().any():
            logger.warning(f"Column '{col}' contains NaN values")
    
    return True


# Convenience function for quick signal generation
def quick_signals(symbol: str, timeframe: str = "D", strategy: str = "momentum") -> List[SignalIntent]:
    """
    Quick signal generation for a symbol using default parameters.
    
    Args:
        symbol: Symbol to analyze
        timeframe: Timeframe (D, H4, H1)
        strategy: Strategy to use
        
    Returns:
        List of signals
    """
    try:
        # Import here to avoid circular imports
        from ..data_core import load_prices
        
        # Load data
        data = load_prices(symbol, frame=timeframe)
        
        if data.empty:
            logger.warning(f"No data available for {symbol}")
            return []
        
        # Generate signals
        return generate_signals(data, strategy, symbol=symbol)
        
    except Exception as e:
        logger.error(f"Quick signal generation failed for {symbol}: {e}")
        return []
