"""
TimeService - Global time provider for time hijacking
"""

from datetime import datetime


class TimeService:
    """Simple global time service - can return real or fake time"""
    
    _fake_time = None
    _is_fake = False
    
    @staticmethod
    def now():
        """Get current time - real or fake depending on mode"""
        if TimeService._is_fake and TimeService._fake_time:
            return TimeService._fake_time
        return datetime.now()
    
    @staticmethod
    def set_fake_time(timestamp):
        """Set fake time for backtesting"""
        TimeService._is_fake = True
        TimeService._fake_time = timestamp
        
    @staticmethod
    def set_real_time():
        """Switch back to real time for live trading"""
        TimeService._is_fake = False
        TimeService._fake_time = None
        
    @staticmethod
    def is_backtesting():
        """Check if we're in backtest mode"""
        return TimeService._is_fake
