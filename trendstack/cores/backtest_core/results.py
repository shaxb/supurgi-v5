"""
Backtest results with broker-like metrics
"""

from typing import List, Dict
from datetime import datetime
import pandas as pd


class BacktestResults:
    """Simple but comprehensive backtest results."""
    
    def __init__(self, initial_deposit: float):
        self.initial_deposit = initial_deposit
        
        # Trade statistics
        self.trades = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # P&L tracking
        self.total_pnl = 0.0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        
        # Account progression
        self.equity_curve = []
        self.balance_curve = []
        self.drawdown_curve = []
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        
        # Broker-specific metrics
        self.margin_calls = 0
        self.forced_liquidations = 0
        self.max_concurrent_positions = 0
        
        # Timestamps
        self.start_time = None
        self.end_time = None
    
    def add_trade(self, position):
        """Add completed trade to results."""
        self.trades.append({
            'symbol': position.symbol,
            'type': position.signal_type.value,
            'entry_price': position.entry_price,
            'exit_price': position.close_price,
            'entry_time': position.open_time,
            'exit_time': position.close_time,
            'pnl': position.realized_pnl,
            'close_reason': position.close_reason
        })
        
        self.total_trades += 1
        
        if position.realized_pnl > 0:
            self.winning_trades += 1
            self.gross_profit += position.realized_pnl
        else:
            self.losing_trades += 1
            self.gross_loss += abs(position.realized_pnl)
        
        self.total_pnl += position.realized_pnl
    
    def calculate_metrics(self, equity_history: List[float], balance_history: List[float]):
        """Calculate final performance metrics."""
        self.equity_curve = equity_history
        self.balance_curve = balance_history
        
        # Basic metrics
        if self.total_trades > 0:
            self.win_rate = (self.winning_trades / self.total_trades) * 100
            
        if self.gross_loss > 0:
            self.profit_factor = self.gross_profit / self.gross_loss
        else:
            self.profit_factor = float('inf') if self.gross_profit > 0 else 0
            
        # Drawdown calculation
        self._calculate_drawdown()
        
        # Sharpe ratio (simplified)
        if len(equity_history) > 1:
            returns = pd.Series(equity_history).pct_change().dropna()
            if returns.std() > 0:
                self.sharpe_ratio = (returns.mean() / returns.std()) * (252 ** 0.5)  # Annualized
    
    def _calculate_drawdown(self):
        """Calculate maximum drawdown from equity curve."""
        if len(self.equity_curve) < 2:
            return
            
        peak = self.equity_curve[0]
        max_dd = 0.0
        
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            else:
                drawdown = (peak - equity) / peak * 100
                max_dd = max(max_dd, drawdown)
                self.drawdown_curve.append(drawdown)
                
        self.max_drawdown = max_dd
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        final_equity = self.equity_curve[-1] if self.equity_curve else self.initial_deposit
        total_return = ((final_equity / self.initial_deposit) - 1) * 100
        
        return {
            # Performance
            'Total Return (%)': round(total_return, 2),
            'Total P&L ($)': round(self.total_pnl, 2),
            'Final Equity ($)': round(final_equity, 2),
            
            # Trade statistics
            'Total Trades': self.total_trades,
            'Win Rate (%)': round(self.win_rate, 1),
            'Profit Factor': round(self.profit_factor, 2),
            
            # Risk metrics
            'Max Drawdown (%)': round(self.max_drawdown, 2),
            'Sharpe Ratio': round(self.sharpe_ratio, 2),
            
            # Broker metrics
            'Margin Calls': self.margin_calls,
            'Forced Liquidations': self.forced_liquidations
        }
    
    def print_summary(self):
        """Print formatted results summary."""
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        
        summary = self.get_summary()
        for key, value in summary.items():
            print(f"{key:<25}: {value}")
        
        print("="*50)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame for analysis."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)
