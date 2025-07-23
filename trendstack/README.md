# TrendStack - Algorithmic Trading System

A modular, production-ready algorithmic trading system built with Python, designed for systematic trading strategies with robust risk management and real-time execution.

## 🚀 **Quick Start**

### Prerequisites
- Python 3.11+ 
- Virtual environment (recommended)
- MetaTrader 5 terminal (for live trading)

### Installation
```bash
# Clone repository
git clone https://github.com/shaxb/supurgi-v5.git
cd supurgi-v5

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install TrendStack package in development mode
cd trendstack
pip install -e .
```

### Configuration
1. Copy `config.yaml.example` to `config.yaml`
2. Configure your broker credentials, risk parameters, and data paths
3. **Never commit secrets to version control**

## 🏗️ **Architecture Overview**

TrendStack follows a modular core-based architecture:

```
trendstack/
├─ cores/                    # Core business logic modules
│   ├─ data_core/           # Data ingestion & cleaning
│   ├─ signal_core/         # Signal generation (momentum, carry)
│   ├─ risk_core/           # Position sizing & risk management
│   ├─ execution_core/      # Order routing & scheduling
│   └─ monitor_core/        # Performance monitoring & alerts
├─ research/                # Backtesting & analysis tools
├─ docker/                  # Containerization configs
└─ scripts/                 # Deployment & maintenance
```

### Core Modules

| Core | Purpose | Key Features |
|------|---------|--------------|
| **data_core** | Market data pipeline | Yahoo Finance, gap filling, roll adjustments |
| **signal_core** | Strategy signals | Momentum, carry trade, volatility filtering |
| **risk_core** | Risk management | Vol parity sizing, Kelly criterion, drawdown limits |
| **execution_core** | Trade execution | MT5 integration, order scheduling |
| **monitor_core** | System monitoring | Live metrics, Telegram alerts, dashboards |

## 📊 **Key Features**

- **Backtesting**: Integrated with Backtrader for strategy validation
- **Live Trading**: MetaTrader 5 integration for automated execution  
- **Risk Management**: Multiple risk controls and position sizing algorithms
- **Monitoring**: Real-time performance tracking with alerts
- **Research Tools**: Jupyter notebooks for strategy development
- **Containerized**: Docker support for production deployment

## 🔧 **Development Workflow**

### 1. Strategy Development
```python
# In research/notebooks/
import sys
sys.path.append('../..')
from cores.signal_core import momentum
from cores.risk_core import sizing

# Develop and test strategies
```

### 2. Backtesting
```bash
python research/backtest_bt.py --strategy momentum --start 2020-01-01
```

### 3. Paper Trading
```bash
# Configure paper trading in config.yaml
python scripts/run_live.sh --mode paper
```

### 4. Live Deployment
```bash
# Production deployment
docker-compose -f docker/compose.yml up -d
```

## 📈 **Supported Strategies**

- **Momentum**: Cross-sectional and time-series momentum
- **Carry Trade**: Interest rate differential strategies  
- **Mean Reversion**: Statistical arbitrage approaches
- **Multi-Factor**: Combined signal strategies

## 🛡️ **Risk Controls**

- **Position Sizing**: Volatility parity, Kelly criterion
- **Drawdown Limits**: Automatic position reduction
- **VAR Monitoring**: Value-at-Risk based stops
- **Correlation Limits**: Portfolio diversification constraints

## 📊 **Monitoring & Alerts**

- **Performance Metrics**: Sharpe, Sortino, Calmar ratios
- **Live Dashboards**: Real-time P&L and risk metrics
- **Alert System**: Telegram/Slack notifications
- **Slippage Tracking**: Execution quality monitoring

## 🐳 **Docker Deployment**

```bash
# Build and run
cd docker
docker-compose up -d

# View logs
docker-compose logs -f trendstack

# Scale services
docker-compose up -d --scale worker=3
```

## 🧪 **Testing**

```bash
# Run all tests
pytest

# Run specific core tests
pytest cores/data_core/tests/

# Coverage report  
pytest --cov=trendstack --cov-report=html
```

## 📚 **Documentation**

- **API Reference**: Auto-generated from docstrings
- **Strategy Guide**: `/docs/strategies.md`
- **Deployment Guide**: `/docs/deployment.md`
- **Risk Management**: `/docs/risk.md`

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-strategy`)
3. Commit changes (`git commit -am 'Add amazing strategy'`)
4. Push to branch (`git push origin feature/amazing-strategy`)
5. Create Pull Request

## ⚠️ **Disclaimer**

This software is for educational and research purposes. Trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.

## 📄 **License**

MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 **Links**

- **Repository**: https://github.com/shaxb/supurgi-v5
- **Issues**: https://github.com/shaxb/supurgi-v5/issues
- **Wiki**: https://github.com/shaxb/supurgi-v5/wiki
