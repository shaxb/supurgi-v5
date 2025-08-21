"""
Backtest Broker - MT5-compatible broker for backtesting
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class AccountInfo:
    """Mimics MT5 AccountInfo structure"""
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    currency: str = "USD"


@dataclass  
class Position:
    """Mimics MT5 Position structure"""
    ticket: int
    symbol: str
    type: int  # 0=buy, 1=sell
    volume: float
    price_open: float
    price_current: float
    profit: float
    swap: float = 0.0
    commission: float = 0.0


@dataclass
class Order:
    """Mimics MT5 Order structure"""
    ticket: int
    symbol: str
    type: int
    volume: float
    price_open: float
    sl: float = 0.0
    tp: float = 0.0


@dataclass
class OrderSendResult:
    """Mimics MT5 OrderSendResult"""
    retcode: int  # 10009 = success
    deal: int = 0
    order: int = 0
    volume: float = 0.0
    price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    comment: str = ""


class BacktestBroker:
    """Simple but effective backtest broker that mimics MT5"""
    
    def __init__(self, initial_balance: float = 10000.0, leverage: int = 100):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        
        # Internal state
        self.positions: Dict[int, Position] = {}
        self.orders: Dict[int, Order] = {}
        self.next_ticket = 1
        
        # Simple price feed (would be replaced with real data)
        self.prices: Dict[str, Dict[str, float]] = {}
    
    def set_price(self, symbol: str, bid: float, ask: float):
        """Set current prices for symbol"""
        self.prices[symbol] = {"bid": bid, "ask": ask}
    
    def account_info(self) -> Optional[AccountInfo]:
        """Get account info - mimics mt5.account_info()"""
        # Calculate equity and margin
        equity = self.balance
        margin = 0.0
        
        for pos in self.positions.values():
            # Add unrealized P&L to equity
            if pos.symbol in self.prices:
                current_price = (self.prices[pos.symbol]["bid"] if pos.type == 0 
                               else self.prices[pos.symbol]["ask"])
                if pos.type == 0:  # Buy
                    profit = (current_price - pos.price_open) * pos.volume
                else:  # Sell
                    profit = (pos.price_open - current_price) * pos.volume
                
                pos.price_current = current_price
                pos.profit = profit
                equity += profit
                
                # Simple margin calculation
                margin += pos.volume * current_price / self.leverage
        
        free_margin = equity - margin
        margin_level = (equity / margin * 100) if margin > 0 else 0
        
        return AccountInfo(
            balance=self.balance,
            equity=equity,
            margin=margin,
            free_margin=free_margin,
            margin_level=margin_level
        )
    
    def positions_get(self, symbol: str = None) -> List[Position]:
        """Get positions - mimics mt5.positions_get()"""
        positions = list(self.positions.values())
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        return positions
    
    def order_send(self, request: Dict[str, Any]) -> Optional[OrderSendResult]:
        """Send order - mimics mt5.order_send()"""
        action = request.get("action")
        symbol = request.get("symbol")
        volume = request.get("volume", 0.0)
        type_order = request.get("type")
        
        if symbol not in self.prices:
            return OrderSendResult(retcode=10018, comment="Invalid symbol")
        
        # Market order
        if action == 1:  # TRADE_ACTION_DEAL
            price = (self.prices[symbol]["ask"] if type_order == 0 
                    else self.prices[symbol]["bid"])
            
            # Create position
            ticket = self.next_ticket
            self.next_ticket += 1
            
            position = Position(
                ticket=ticket,
                symbol=symbol,
                type=type_order,
                volume=volume,
                price_open=price,
                price_current=price,
                profit=0.0
            )
            
            self.positions[ticket] = position
            
            return OrderSendResult(
                retcode=10009,  # Success
                deal=ticket,
                order=ticket,
                volume=volume,
                price=price,
                comment="Market order executed"
            )
        
        # Pending order
        elif action == 0:  # TRADE_ACTION_PENDING
            ticket = self.next_ticket
            self.next_ticket += 1
            
            order = Order(
                ticket=ticket,
                symbol=symbol,
                type=type_order,
                volume=volume,
                price_open=request.get("price", 0.0),
                sl=request.get("sl", 0.0),
                tp=request.get("tp", 0.0)
            )
            
            self.orders[ticket] = order
            
            return OrderSendResult(
                retcode=10009,
                order=ticket,
                comment="Pending order placed"
            )
        
        # Close position
        elif action == 2:  # TRADE_ACTION_SLTP or close
            position_ticket = request.get("position", 0)
            if position_ticket in self.positions:
                pos = self.positions[position_ticket]
                
                # Calculate final profit
                close_price = (self.prices[symbol]["bid"] if pos.type == 0 
                             else self.prices[symbol]["ask"])
                
                if pos.type == 0:  # Buy position
                    final_profit = (close_price - pos.price_open) * pos.volume
                else:  # Sell position
                    final_profit = (pos.price_open - close_price) * pos.volume
                
                # Update balance
                self.balance += final_profit
                
                # Remove position
                del self.positions[position_ticket]
                
                return OrderSendResult(
                    retcode=10009,
                    deal=self.next_ticket,
                    volume=pos.volume,
                    price=close_price,
                    comment="Position closed"
                )
        
        return OrderSendResult(retcode=10013, comment="Invalid request")
    
    def orders_get(self) -> List[Order]:
        """Get pending orders - mimics mt5.orders_get()"""
        return list(self.orders.values())
