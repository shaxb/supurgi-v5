# cores/data_core/api.py
from dataclasses import dataclass
import pandas as pd
from typing import Literal, Optional, Dict

Frame = Literal["D", "H4", "H1"]

@dataclass(frozen=True)
class CostSpec:
    spread: float        # in price units (e.g. 0.00015 for EURUSD)
    commission: float    # $ per side per lot (or % notional)
    swap_long: float     # $/lot/day
    swap_short: float    # $/lot/day

def load_prices(symbol: str, frame: Frame = "D",
                start: Optional[str] = None,
                end: Optional[str] = None) -> pd.DataFrame:
    """Smart data loader - gets what you need, when you need it.
    
    Automatically:
    - Downloads missing data for the requested range
    - Uses cached data when available
    - Returns clean OHLCV data for the exact timeframe requested
    """
    from .manager import get_data
    return get_data(symbol, frame, start, end)

def load_costs(symbol: str) -> CostSpec:
    """Get trading costs for a symbol."""
    from .costs import load_cost_specs
    cost_specs = load_cost_specs()
    if symbol not in cost_specs:
        raise ValueError(f"No cost spec found for {symbol}")
    return cost_specs[symbol]
