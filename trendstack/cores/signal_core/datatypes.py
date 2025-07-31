"""
Signal Core Data Types

Defines core data structures for signal generation and processing.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List
import pandas as pd
from datetime import datetime


class SignalType(Enum):
    """Signal direction type."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"


class SignalStrength(Enum):
    """Signal strength classification."""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


@dataclass
class SignalIntent:
    """
    Core signal data structure.
    
    Represents a trading signal with all necessary information
    for decision making and execution.
    """
    # Basic signal properties
    symbol: str
    timestamp: datetime
    signal_type: SignalType
    strength: SignalStrength
    
    # Price information
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Signal metadata
    strategy_name: str = ""
    confidence: float = 0.0  # 0.0 to 1.0
    risk_reward_ratio: Optional[float] = None
    
    # Additional context
    timeframe: str = "D"
    indicators: Dict[str, Any] = None
    notes: str = ""
    
    def __post_init__(self):
        """Initialize default values."""
        if self.indicators is None:
            self.indicators = {}
    
    @property
    def is_entry_signal(self) -> bool:
        """Check if this is an entry signal."""
        return self.signal_type in [SignalType.BUY, SignalType.SELL]
    
    @property
    def is_exit_signal(self) -> bool:
        """Check if this is an exit signal."""
        return self.signal_type in [SignalType.EXIT_LONG, SignalType.EXIT_SHORT]
    
    @property
    def is_bullish(self) -> bool:
        """Check if signal is bullish."""
        return self.signal_type == SignalType.BUY
    
    @property
    def is_bearish(self) -> bool:
        """Check if signal is bearish."""
        return self.signal_type == SignalType.SELL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'signal_type': self.signal_type.value,
            'strength': self.strength.value,
            'price': self.price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'strategy_name': self.strategy_name,
            'confidence': self.confidence,
            'risk_reward_ratio': self.risk_reward_ratio,
            'timeframe': self.timeframe,
            'indicators': self.indicators,
            'notes': self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignalIntent':
        """Create signal from dictionary."""
        return cls(
            symbol=data['symbol'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            signal_type=SignalType(data['signal_type']),
            strength=SignalStrength(data['strength']),
            price=data['price'],
            stop_loss=data.get('stop_loss'),
            take_profit=data.get('take_profit'),
            strategy_name=data.get('strategy_name', ''),
            confidence=data.get('confidence', 0.0),
            risk_reward_ratio=data.get('risk_reward_ratio'),
            timeframe=data.get('timeframe', 'D'),
            indicators=data.get('indicators', {}),
            notes=data.get('notes', '')
        )


@dataclass
class StrategyConfig:
    """Configuration for signal generation strategies."""
    name: str
    parameters: Dict[str, Any]
    enabled: bool = True
    weight: float = 1.0  # For multi-strategy combinations
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """Get parameter value with default."""
        return self.parameters.get(key, default)


class SignalBuffer:
    """
    Buffer for managing multiple signals over time.
    Useful for strategy backtesting and signal aggregation.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.signals: List[SignalIntent] = []
    
    def add_signal(self, signal: SignalIntent) -> None:
        """Add a signal to the buffer."""
        self.signals.append(signal)
        
        # Maintain max size
        if len(self.signals) > self.max_size:
            self.signals = self.signals[-self.max_size:]
    
    def get_signals(
        self, 
        symbol: Optional[str] = None,
        signal_type: Optional[SignalType] = None,
        strategy: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[SignalIntent]:
        """Get filtered signals from buffer."""
        filtered = self.signals
        
        if symbol:
            filtered = [s for s in filtered if s.symbol == symbol]
        
        if signal_type:
            filtered = [s for s in filtered if s.signal_type == signal_type]
        
        if strategy:
            filtered = [s for s in filtered if s.strategy_name == strategy]
        
        if limit:
            filtered = filtered[-limit:]
        
        return filtered
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert signals to DataFrame for analysis."""
        if not self.signals:
            return pd.DataFrame()
        
        data = [signal.to_dict() for signal in self.signals]
        return pd.DataFrame(data)
    
    def clear(self) -> None:
        """Clear all signals from buffer."""
        self.signals.clear()
    
    def __len__(self) -> int:
        """Return number of signals in buffer."""
        return len(self.signals)
