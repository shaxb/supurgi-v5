# cores/data_core/api.py
from dataclasses import dataclass
import pandas as pd
from typing import Literal, Optional, Dict

Frame = Literal["D", "H4", "H1"]  # start with "D"

@dataclass(frozen=True)
class CostSpec:
    spread: float        # in price units (e.g. 0.00015 for EURUSD)
    commission: float    # $ per side per lot (or % notional)
    swap_long: float     # $/lot/day
    swap_short: float    # $/lot/day

def load_prices(symbol: str, frame: Frame = "D",
                start: Optional[str] = None,
                end: Optional[str] = None) -> pd.DataFrame:
    """Return cleaned OHLCV in UTC index, no gaps, tz-aware, float32 columns."""
    from .manager import get_cleaned_data
    return get_cleaned_data(symbol, frame, start, end)

def load_costs(symbol: str) -> CostSpec:
    """Return cost spec from costs.yml; raise if missing."""
    from .costs import load_cost_specs
    cost_specs = load_cost_specs()
    if symbol not in cost_specs:
        raise ValueError(f"No cost spec found for {symbol}")
    return cost_specs[symbol]

def refresh_symbol(symbol: str, source: str | None = None) -> None:
    """Pull latest raw, append, clean, write processed parquet."""
    from .manager import refresh_data
    refresh_data(symbol, source)

def list_symbols() -> Dict[str, str]:
    """symbol -> source mapping from symbols.yaml"""
    from .manager import get_symbol_mapping
    return get_symbol_mapping()
