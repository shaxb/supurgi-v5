import yaml
from pathlib import Path
from .api import CostSpec

def load_cost_specs():
    """Load cost specifications from costs.yaml."""
    costs_path = Path(__file__).parent / "costs.yaml"
    
    with open(costs_path, 'r') as f:
        costs_data = yaml.safe_load(f)
    
    # Convert to CostSpec objects
    cost_specs = {}
    for symbol, data in costs_data.items():
        cost_specs[symbol] = CostSpec(
            spread=data['spread'],
            commission=data['commission'],
            swap_long=data['swap_long'],
            swap_short=data['swap_short']
        )
    
    return cost_specs
