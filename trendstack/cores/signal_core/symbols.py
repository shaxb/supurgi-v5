"""
Symbol configuration for orchestrator watchlist
"""

import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path
from loguru import logger


# Default symbols config location
SYMBOLS_CONFIG_PATH = Path(__file__).parent / "symbols.yaml"


def load_symbols() -> Dict[str, Dict[str, Any]]:
    """Load symbols configuration from YAML."""
    try:
        if SYMBOLS_CONFIG_PATH.exists():
            with open(SYMBOLS_CONFIG_PATH, 'r') as f:
                return yaml.safe_load(f) or {}
        else:
            logger.warning(f"Symbols config not found: {SYMBOLS_CONFIG_PATH}")
            return {}
    except Exception as e:
        logger.error(f"Failed to load symbols config: {e}")
        return {}


def load_symbols() -> Dict[str, Dict[str, Any]]:
    """Load symbols configuration from YAML."""
    try:
        if SYMBOLS_CONFIG_PATH.exists():
            with open(SYMBOLS_CONFIG_PATH, 'r') as f:
                return yaml.safe_load(f) or {}
        else:
            logger.warning(f"Symbols config not found: {SYMBOLS_CONFIG_PATH}")
            return {}
    except Exception as e:
        logger.error(f"Failed to load symbols config: {e}")
        return {}


def get_strategy_config(symbol: str, strategy_instance: str) -> Dict[str, Any]:
    """Get strategy-specific config for symbol (excluding timeframe)."""
    symbols_config = load_symbols()
    symbol_config = symbols_config.get(symbol, {})
    strategies = symbol_config.get('strategies', {})
    strategy_config = strategies.get(strategy_instance, {}).copy()
    
    # Remove timeframe from strategy config (it's used by orchestrator for data loading)
    strategy_config.pop('timeframe', None)
    return strategy_config


def get_symbol_strategy_pairs() -> List[tuple]:
    """Get all active (symbol, strategy_instance, timeframe) tuples for orchestrator loop."""
    pairs = []
    symbols_config = load_symbols()
    
    for symbol, config in symbols_config.items():
        if not config.get('active', True):
            continue
            
        strategies = config.get('strategies', {})
        for strategy_instance, strategy_config in strategies.items():
            timeframe = strategy_config.get('timeframe', 'H4')
            pairs.append((symbol, strategy_instance, timeframe))
    
    return pairs


def update_strategy_config(symbol: str, strategy_instance: str, new_config: Dict[str, Any]) -> bool:
    """
    Update strategy configuration in symbols.yaml (for optimizer).
    
    Args:
        symbol: Symbol to update
        strategy_instance: Strategy instance (e.g., 'momentum_H4')
        new_config: New configuration parameters
        
    Returns:
        True if updated successfully
    """
    try:
        symbols_config = load_symbols()
        
        if symbol not in symbols_config:
            logger.error(f"Symbol {symbol} not found in configuration")
            return False
        
        if 'strategies' not in symbols_config[symbol]:
            symbols_config[symbol]['strategies'] = {}
        
        if strategy_instance not in symbols_config[symbol]['strategies']:
            logger.error(f"Strategy {strategy_instance} not found for {symbol}")
            return False
        
        # Update the configuration
        symbols_config[symbol]['strategies'][strategy_instance].update(new_config)
        
        # Save back to file
        with open(SYMBOLS_CONFIG_PATH, 'w') as f:
            yaml.safe_dump(symbols_config, f, default_flow_style=False)
        
        logger.info(f"Updated {symbol} {strategy_instance} config")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update strategy config: {e}")
        return False
