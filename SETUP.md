# TrendStack Setup Instructions

## Virtual Environment Setup

1. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   ```

2. **Activate environment:**
   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Key Libraries Installed

### Trading & Backtesting
- **backtrader** (1.9.78.123) - Python backtesting library for trading strategies
- **MetaTrader5** (5.0.5120) - MT5 trading platform integration

### Data Analysis & Processing  
- **pandas** (2.3.1) - Data manipulation and analysis
- **numpy** (2.3.1) - Numerical computing
- **scipy** (1.16.0) - Scientific computing
- **yfinance** (0.2.65) - Yahoo Finance data downloader

### Technical Analysis
- **ta** (0.11.0) - Technical Analysis library with 130+ indicators

### Visualization & Reporting
- **matplotlib** (3.10.3) - Static plotting
- **plotly** (6.2.0) - Interactive plotting and dashboards  
- **seaborn** (0.13.2) - Statistical visualization
- **quantstats** (0.0.68) - Portfolio performance analytics

### Communication & Notifications
- **python-telegram-bot** (13.15) - Telegram bot integration

### Workflow & Scheduling
- **schedule** (1.2.2) - Simple job scheduling
- **prefect** (3.4.10) - Modern workflow orchestration
- **loguru** (0.7.3) - Advanced logging

### Data Handling
- **pydantic** (2.11.7) - Data validation and settings
- **pyyaml** (6.0.2) - YAML parser and emitter
- **requests** (2.32.4) - HTTP library

## Notes

- **empyrical** was skipped due to Python 3.13 compatibility issues
- **vectorbt-lite** was not available in the package index
- Total packages installed: 129
- Compatible with Python 3.13

## Getting Started

After setup, test the installation:
```python
import backtrader
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

print("✅ TrendStack environment ready!")
```

## Environment Activation

Remember to activate your virtual environment each time:
- Windows: `.venv\Scripts\activate`
- Linux/Mac: `source .venv/bin/activate`
