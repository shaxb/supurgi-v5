"""
Risk Core Module

Comprehensive risk management system for TrendStack.
Handles position sizing, risk assessment, and portfolio risk management.
"""

from .manager import (
    RiskManager,
    PositionSizer,
    RiskLimits,
    DrawdownController
)

from .calculator import (
    RiskCalculator,
    VaRCalculator,
    PortfolioRisk,
    CorrelationAnalyzer
)

from .monitor import (
    RiskMonitor,
    AlertManager,
    RiskReporter,
    RealTimeRiskTracker
)

from .models import (
    RiskModel,
    MarketRiskModel,
    CreditRiskModel,
    OperationalRiskModel
)

__all__ = [
    # Manager
    'RiskManager',
    'PositionSizer',
    'RiskLimits',
    'DrawdownController',
    
    # Calculator
    'RiskCalculator',
    'VaRCalculator',
    'PortfolioRisk',
    'CorrelationAnalyzer',
    
    # Monitor
    'RiskMonitor',
    'AlertManager',
    'RiskReporter',
    'RealTimeRiskTracker',
    
    # Models
    'RiskModel',
    'MarketRiskModel',
    'CreditRiskModel',
    'OperationalRiskModel'
]

# Version info
__version__ = "1.0.0"
__author__ = "TrendStack Team"
__description__ = "Advanced risk management and control system"
