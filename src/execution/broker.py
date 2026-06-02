from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Callable


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    volume: float
    order_type: OrderType
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    order_id: Optional[str] = None
    status: str = "pending"
    created_at: Optional[str] = None
    executed_at: Optional[str] = None
    executed_price: Optional[float] = None
    profit: Optional[float] = None


@dataclass
class Position:
    symbol: str
    side: OrderSide
    volume: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    unrealized_pnl: float
    realized_pnl: float = 0.0


class Broker:
    def get_positions(self) -> list[Position]:
        raise NotImplementedError

    def place_order(self, order: Order) -> Optional[Order]:
        raise NotImplementedError

    def close_position(self, symbol: str, side: OrderSide) -> bool:
        raise NotImplementedError

    def get_account_balance(self) -> float:
        raise NotImplementedError


class PaperBroker(Broker):
    def __init__(self, initial_balance: float = 10_000):
        self.balance = initial_balance
        self.equity = initial_balance
        self.positions: dict[str, Position] = {}
        self.closed_orders: list[Order] = []
        self._order_counter = 0
        self.on_close: Optional[Callable] = None

    def get_positions(self) -> list[Position]:
        to_close = []
        for key, pos in self.positions.items():
            try:
                import MetaTrader5 as mt5
                tick = mt5.symbol_info_tick(pos.symbol)
                if tick:
                    pos.current_price = tick.bid if pos.side == OrderSide.SELL else tick.ask
                else:
                    pos.current_price = pos.entry_price
            except Exception:
                pos.current_price = pos.entry_price

            pos.unrealized_pnl = self._calculate_pnl(pos, pos.current_price)

            if pos.side == OrderSide.BUY:
                if pos.stop_loss and pos.current_price <= pos.stop_loss:
                    to_close.append((key, pos, 'loss'))
                elif pos.take_profit and pos.current_price >= pos.take_profit:
                    to_close.append((key, pos, 'win'))
            else:
                if pos.stop_loss and pos.current_price >= pos.stop_loss:
                    to_close.append((key, pos, 'loss'))
                elif pos.take_profit and pos.current_price <= pos.take_profit:
                    to_close.append((key, pos, 'win'))

        for key, pos, result in to_close:
            pnl = self._calculate_pnl(pos, pos.current_price)
            self.balance += (pos.volume * 1000) + pnl
            del self.positions[key]
            if self.on_close:
                self.on_close(key, pos.current_price, datetime.now().isoformat(), pnl, result)
            import logging
            logging.getLogger('trading_bot').info(
                f"CLOSED {pos.symbol} {pos.side.value.upper()} "
                f"@ {pos.current_price} | PnL: ${pnl:.2f} | Result: {result.upper()}"
            )

        return list(self.positions.values())

    def place_order(self, order: Order) -> Optional[Order]:
        self._order_counter += 1
        order.order_id = f"paper_{self._order_counter}"
        order.created_at = datetime.now().isoformat()

        if self.balance < order.volume * 1000:
            order.status = "rejected"
            self.closed_orders.append(order)
            return order

        order.status = "executed"
        order.executed_at = datetime.now().isoformat()
        order.executed_price = order.price or self._mock_price(order.symbol)

        position_key = f"{order.symbol}_{order.side.value}"

        if position_key in self.positions:
            existing = self.positions[position_key]
            total_vol = existing.volume + order.volume
            existing.entry_price = (
                (existing.entry_price * existing.volume)
                + (order.executed_price * order.volume)
            ) / total_vol
            existing.volume = total_vol
        else:
            self.positions[position_key] = Position(
                symbol=order.symbol,
                side=order.side,
                volume=order.volume,
                entry_price=order.executed_price,
                current_price=order.executed_price,
                stop_loss=order.stop_loss or 0,
                take_profit=order.take_profit or 0,
                unrealized_pnl=0.0,
            )

        self.balance -= order.volume * 1000
        self.closed_orders.append(order)
        return order

    def close_position(self, symbol: str, side: OrderSide) -> bool:
        position_key = f"{symbol}_{side.value}"
        if position_key not in self.positions:
            return False

        pos = self.positions[position_key]
        exit_price = self._mock_price(symbol)
        pnl = self._calculate_pnl(pos, exit_price)
        pos.realized_pnl += pnl
        pos.unrealized_pnl = 0
        self.balance += (pos.volume * 1000) + pnl
        del self.positions[position_key]
        if self.on_close:
            self.on_close(position_key, exit_price, datetime.now().isoformat(), pnl, 'manual')
        return True

    def get_account_balance(self) -> float:
        return self.balance

    def update_prices(self, prices: dict[str, float]):
        for pos in self.positions.values():
            price = prices.get(pos.symbol)
            if price:
                pos.current_price = price
                pos.unrealized_pnl = self._calculate_pnl(pos, price)

    @staticmethod
    def _mock_price(symbol: str) -> float:
        try:
            import MetaTrader5 as mt5
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                return tick.bid
        except Exception:
            pass
        return {"EURUSD": 1.08, "GBPUSD": 1.26, "USDJPY": 150.0,
                "USDCHF": 0.88, "AUDUSD": 0.65, "USDCAD": 1.36,
                "NZDUSD": 0.60}.get(symbol, 1.0)

    @staticmethod
    def _calculate_pnl(position: Position, exit_price: float) -> float:
        diff = exit_price - position.entry_price
        if position.side == OrderSide.SELL:
            diff = -diff
        return diff * position.volume * 100_000
