"""
Strategy Registry - Plugin Manager

Manages registration and discovery of signal generation strategies.
"""

from typing import Dict, Type, List, Optional, Any
from abc import ABC, abstractmethod
import inspect
from loguru import logger

from .datatypes import SignalIntent, StrategyConfig


class BaseStrategy(ABC):
    """
    Base class for all signal generation strategies.
    
    All strategies must inherit from this class and implement generate_signals.
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize strategy with configuration.
        
        Args:
            config: Strategy configuration containing parameters
        """
        self.config = config
        self.name = config.name
        self.parameters = config.parameters
    
    @abstractmethod
    def generate_signals(self, data, symbol: str) -> List[SignalIntent]:
        """
        Generate trading signals from market data.
        
        Args:
            data: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            symbol: Symbol identifier
            
        Returns:
            List of SignalIntent objects
        """
        pass
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get parameter value with default."""
        return self.parameters.get(key, default)
    
    def validate_data(self, data) -> bool:
        """
        Validate input data format.
        
        Args:
            data: Input DataFrame
            
        Returns:
            True if data is valid
        """
        required_columns = ['open', 'high', 'low', 'close']
        
        if data.empty:
            logger.warning(f"{self.name}: Input data is empty")
            return False
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"{self.name}: Missing columns: {missing_columns}")
            return False
        
        return True
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


class StrategyRegistry:
    """
    Registry for managing signal generation strategies.
    
    Provides plugin-like functionality for strategy discovery and instantiation.
    """
    
    def __init__(self):
        self._strategies: Dict[str, Type[BaseStrategy]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(self, name: str, strategy_class: Type[BaseStrategy]) -> None:
        """
        Register a strategy class.
        
        Args:
            name: Unique strategy name
            strategy_class: Strategy class inheriting from BaseStrategy
            
        Raises:
            ValueError: If strategy is invalid or name already exists
        """
        # Validate strategy class
        if not issubclass(strategy_class, BaseStrategy):
            raise ValueError(f"Strategy {strategy_class.__name__} must inherit from BaseStrategy")
        
        if not hasattr(strategy_class, 'generate_signals'):
            raise ValueError(f"Strategy {strategy_class.__name__} must implement generate_signals method")
        
        # Check if name already exists
        if name in self._strategies:
            logger.warning(f"Strategy '{name}' already registered, overwriting")
        
        # Register strategy
        self._strategies[name] = strategy_class
        
        # Extract metadata
        self._metadata[name] = self._extract_metadata(strategy_class)
        
        logger.debug(f"Registered strategy: {name}")
    
    def get_strategy(self, name: str) -> Optional[Type[BaseStrategy]]:
        """
        Get strategy class by name.
        
        Args:
            name: Strategy name
            
        Returns:
            Strategy class or None if not found
        """
        return self._strategies.get(name)
    
    def list_strategies(self) -> List[str]:
        """
        Get list of registered strategy names.
        
        Returns:
            List of strategy names
        """
        return list(self._strategies.keys())
    
    def unregister(self, name: str) -> bool:
        """
        Remove strategy from registry.
        
        Args:
            name: Strategy name
            
        Returns:
            True if removed, False if not found
        """
        if name in self._strategies:
            del self._strategies[name]
            del self._metadata[name]
            logger.debug(f"Unregistered strategy: {name}")
            return True
        return False
    
    def get_strategy_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about a strategy.
        
        Args:
            name: Strategy name
            
        Returns:
            Dictionary with strategy metadata or None
        """
        return self._metadata.get(name)
    
    def clear(self) -> None:
        """Clear all registered strategies."""
        self._strategies.clear()
        self._metadata.clear()
        logger.debug("Cleared all strategies from registry")
    
    def _extract_metadata(self, strategy_class: Type[BaseStrategy]) -> Dict[str, Any]:
        """
        Extract metadata from strategy class.
        
        Args:
            strategy_class: Strategy class
            
        Returns:
            Dictionary with metadata
        """
        metadata = {
            'class_name': strategy_class.__name__,
            'module': strategy_class.__module__,
            'docstring': strategy_class.__doc__ or "No description available",
            'parameters': []
        }
        
        # Try to extract parameter information from __init__ signature
        try:
            init_signature = inspect.signature(strategy_class.__init__)
            for param_name, param in init_signature.parameters.items():
                if param_name not in ['self', 'config']:
                    metadata['parameters'].append({
                        'name': param_name,
                        'default': param.default if param.default != inspect.Parameter.empty else None,
                        'annotation': str(param.annotation) if param.annotation != inspect.Parameter.empty else None
                    })
        except Exception as e:
            logger.debug(f"Could not extract parameters for {strategy_class.__name__}: {e}")
        
        return metadata
    
    def auto_discover(self, module_path: str) -> int:
        """
        Automatically discover and register strategies from a module.
        
        Args:
            module_path: Python module path to scan
            
        Returns:
            Number of strategies discovered and registered
        """
        try:
            import importlib
            module = importlib.import_module(module_path)
            
            discovered = 0
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                
                # Check if it's a strategy class
                if (inspect.isclass(attr) and 
                    issubclass(attr, BaseStrategy) and 
                    attr != BaseStrategy):
                    
                    # Use class name as strategy name (convert to lowercase)
                    strategy_name = attr_name.lower().replace('strategy', '')
                    
                    try:
                        self.register(strategy_name, attr)
                        discovered += 1
                    except Exception as e:
                        logger.warning(f"Failed to register {attr_name}: {e}")
            
            logger.info(f"Auto-discovered {discovered} strategies from {module_path}")
            return discovered
            
        except Exception as e:
            logger.error(f"Failed to auto-discover strategies from {module_path}: {e}")
            return 0
    
    def __len__(self) -> int:
        """Return number of registered strategies."""
        return len(self._strategies)
    
    def __contains__(self, name: str) -> bool:
        """Check if strategy is registered."""
        return name in self._strategies
    
    def __repr__(self) -> str:
        return f"StrategyRegistry({len(self._strategies)} strategies)"
