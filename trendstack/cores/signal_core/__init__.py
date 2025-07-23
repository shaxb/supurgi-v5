"""
Signal Core Module

Central signal generation and management system for TrendStack.
Handles technical analysis, pattern recognition, and signal processing.
"""

from .generators import (
    SignalGenerator,
    TechnicalSignalGenerator,
    PatternSignalGenerator,
    MLSignalGenerator,
    CompositeSignalGenerator
)

from .processors import (
    SignalProcessor,
    SignalFilter,
    SignalAggregator,
    SignalValidator
)

from .indicators import (
    TechnicalIndicators,
    CustomIndicators,
    VolumeIndicators,
    VolatilityIndicators
)

from .patterns import (
    PatternDetector,
    CandlestickPatterns,
    ChartPatterns,
    VolumePatterns
)

__all__ = [
    # Generators
    'SignalGenerator',
    'TechnicalSignalGenerator', 
    'PatternSignalGenerator',
    'MLSignalGenerator',
    'CompositeSignalGenerator',
    
    # Processors
    'SignalProcessor',
    'SignalFilter',
    'SignalAggregator',
    'SignalValidator',
    
    # Indicators
    'TechnicalIndicators',
    'CustomIndicators',
    'VolumeIndicators',
    'VolatilityIndicators',
    
    # Patterns
    'PatternDetector',
    'CandlestickPatterns',
    'ChartPatterns',
    'VolumePatterns'
]

# Version info
__version__ = "1.0.0"
__author__ = "TrendStack Team"
__description__ = "Advanced signal generation and processing system"
