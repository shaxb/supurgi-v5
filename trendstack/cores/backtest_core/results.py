"""
Simple backtest results
"""

from loguru import logger


class BacktestResults:
    """Simple results collection and reporting."""
    
    def __init__(self, initial_deposit):
        self.initial_deposit = initial_deposit
        self.trades = []
        self.final_balance = 0
        self.equity_curve = []
    
    def add_trade(self, position):
        """Add completed position to results."""
        if not position.is_closed:
            return
            
        self.trades.append({
            'symbol': position.symbol,
            'type': position.signal_type.value,
            'entry_price': position.entry_price,
            'exit_price': position.close_price,
            'size': position.size,
            'pnl': position.realized_pnl,
            'open_time': position.open_time,
            'close_time': position.close_time,
            'reason': position.close_reason
        })
    
    def add_equity_point(self, timestamp, balance, equity):
        """Add equity curve point."""
        self.equity_curve.append({
            'timestamp': timestamp,
            'balance': balance,
            'equity': equity
        })
    
    def finalize(self, final_balance):
        """Finalize results with final balance."""
        self.final_balance = final_balance
    
    def print_summary(self):
        """Print backtest summary."""
        if not self.trades:
            logger.warning("No trades executed")
            return
            
        total_pnl = sum(trade['pnl'] for trade in self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        losing_trades = len([t for t in self.trades if t['pnl'] < 0])
        win_rate = winning_trades / len(self.trades) * 100 if self.trades else 0
        
        logger.info("=== BACKTEST RESULTS ===")
        logger.info(f"Initial Deposit: ${self.initial_deposit:,.2f}")
        logger.info(f"Final Balance: ${self.final_balance:,.2f}")
        logger.info(f"Total P&L: ${total_pnl:,.2f}")
        logger.info(f"Return: {(self.final_balance/self.initial_deposit-1)*100:.2f}%")
        logger.info(f"Total Trades: {len(self.trades)}")
        logger.info(f"Win Rate: {win_rate:.1f}% ({winning_trades}W/{losing_trades}L)")
        logger.info("========================")
