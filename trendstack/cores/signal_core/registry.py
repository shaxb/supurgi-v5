"""
Strategy Registry - Simple Plugin Manager

Manages registration and discovery of signal generation strategies.
"""

from typing import Dict, Type, List, Optional, Any
from abc import ABC, abstractmethod
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
    def generate_signals(self, data, symbol: str) -> Optional[SignalIntent]:
        """
        Generate trading signal from market data.
        
        Args:
            data: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            symbol: Symbol identifier
            
        Returns:
            Single SignalIntent object or None if no signal
        """
        pass
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get parameter value with default."""
        return self.parameters.get(key, default)
    
    def get_cooldown(self) -> int:
        """
        Get strategy cooldown in minutes.
        
        Returns:
            Cooldown period in minutes (0 = no cooldown)
        """
        return self.get_parameter('cooldown_minutes', 5)  # Default 5 minutes
    
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
    """Simple registry for managing signal generation strategies."""
    
    def __init__(self):
        self._strategies: Dict[str, Type[BaseStrategy]] = {}
    
    def register(self, name: str, strategy_class: Type[BaseStrategy]) -> None:
        """
        Register a strategy class.
        
        Args:
            name: Unique strategy name
            strategy_class: Strategy class inheriting from BaseStrategy
            
        Raises:
            ValueError: If strategy is invalid
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
