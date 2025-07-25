import yaml
import os
from pathlib import Path

# Load config from local config.yaml
CONFIG_PATH = Path(__file__).parent / "config.yaml"

def load_config():
    """Load data core configuration."""
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

# Global config object
CONFIG = load_config()

# Ensure data directory exists
data_path = Path(CONFIG['data_path'])
data_path.mkdir(exist_ok=True)
