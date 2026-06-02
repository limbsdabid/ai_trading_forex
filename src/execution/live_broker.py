import logging
import time
from datetime import datetime
from typing import Optional

import MetaTrader5 as mt5

from .broker import Broker, Order, Position, OrderSide, OrderType

log = logging.getLogger("trading_bot")

MT5_RETCODES = {
    10004: "TRADE_RETCODE_REQUOTE",
    10006: "TRADE_RETCODE_REJECT",
    10007: "TRADE_RETCODE_CANCEL",
    10008: "TRADE_RETCODE_PLACED",
    10009: "TRADE_RETCODE_DONE",
    10010: "TRADE_RETCODE_DONE_PARTIAL",
    10011: "TRADE_RETCODE_ERROR",
    10012: "TRADE_RETCODE_TIMEOUT",
    10013: "TRADE_RETCODE_INVALID",
    10014: "TRADE_RETCODE_INVALID_VOLUME",
    10015: "TRADE_RETCODE_INVALID_PRICE",
    10016: "TRADE_RETCODE_INVALID_STOPS",
    10017: "TRADE_RETCODE_TRADE_DISABLED",
    10018: "TRADE_RETCODE_MARKET_CLOSED",
    10019: "TRADE_RETCODE_NO_MONEY",
    10020: "TRADE_RETCODE_PRICE_CHANGED",
    10021: "TRADE_RETCODE_PRICE_OFF",
    10022: "TRADE_RETCODE_INVALID_EXPIRATION",
    10023: "TRADE_RETCODE_ORDER_CHANGED",
    10024: "TRADE_RETCODE_TOO_MANY_REQUESTS",
    10025: "TRADE_RETCODE_NO_CHANGES",
    10026: "TRADE_RETCODE_SERVER_DISABLES_AT",
    10027: "TRADE_RETCODE_UNKNOWN_SYMBOL",
    10028: "TRADE_RETCODE_NO_ORDER_TYPE_INVESTOR",
}


class LiveBroker(Broker):
    def __init__(self, magic: int = 234000, deviation: int = 20,
                 max_retries: int = 3, retry_delay: float = 1.0):
        self.magic = magic
        self.deviation = deviation
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _ensure_connection(self) -> bool:
        if mt5.terminal_info():
            return True
        log.warning("MT5 terminal disconnected, attempting reconnect...")
        return False

    def _send_request(self, request: dict, label: str) -> Optional[mt5.OrderSendResult]:
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            if not self._ensure_connection():
                time.sleep(self.retry_delay)
                continue

            result = mt5.order_send(request)
            if result is None:
                err = mt5.last_error()
                log.warning(f"{label} attempt {attempt}/{self.max_retries} failed: "
                            f"mt5 returned None, error={err}")
                last_error = None
                time.sleep(self.retry_delay)
                continue

            code_name = MT5_RETCODES.get(result.retcode, f"UNKNOWN_{result.retcode}")
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return result
            if result.retcode == mt5.TRADE_RETCODE_DONE_PARTIAL:
                log.warning(f"{label}: partial fill ({result.volume})")
                return result

            log.warning(f"{label} attempt {attempt}/{self.max_retries}: "
                        f"retcode={result.retcode} ({code_name}), "
                        f"comment='{result.comment}'")
            last_error = result

            if result.retcode in (mt5.TRADE_RETCODE_MARKET_CLOSED,
                                  mt5.TRADE_RETCODE_NO_MONEY,
                                  mt5.TRADE_RETCODE_INVALID_VOLUME,
                                  mt5.TRADE_RETCODE_TRADE_DISABLED,
                                  mt5.TRADE_RETCODE_SERVER_DISABLES_AT):
                log.error(f"{label}: non-retryable error ({code_name}), giving up")
                return result

            time.sleep(self.retry_delay)

        if last_error:
            return last_error
        return None

    def place_order(self, order: Order) -> Optional[Order]:
        tick = mt5.symbol_info_tick(order.symbol)
        if tick is None:
            log.error(f"Cannot place {order.symbol} order: symbol not found")
            order.status = "rejected"
            return order

        price = tick.ask if order.side == OrderSide.BUY else tick.bid
        mt5_type = mt5.ORDER_TYPE_BUY if order.side == OrderSide.BUY else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": order.symbol,
            "volume": order.volume,
            "type": mt5_type,
            "price": price,
            "sl": order.stop_loss or 0,
            "tp": order.take_profit or 0,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": "SMC Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        label = f"OPEN {order.side.value.upper()} {order.symbol} vol={order.volume}"
        result = self._send_request(request, label)

        if result is None:
            order.status = "error"
            log.error(f"{label}: all retries exhausted, mt5 returned None")
            return order

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            order.status = "rejected"
            code_name = MT5_RETCODES.get(result.retcode, f"UNKNOWN_{result.retcode}")
            log.warning(f"{label}: rejected ({code_name})")
            return order

        order.order_id = str(result.order)
        order.status = "executed"
        order.executed_at = datetime.now().isoformat()
        order.executed_price = price
        log.info(f"{label}: executed, ticket={result.order}")
        return order

    def close_position(self, symbol: str, side: OrderSide) -> bool:
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            log.warning(f"Cannot close {symbol}: no positions data")
            return False

        target_type = 0 if side == OrderSide.BUY else 1
        opposite_type = mt5.ORDER_TYPE_SELL if side == OrderSide.BUY else mt5.ORDER_TYPE_BUY

        for pos in positions:
            if pos.type != target_type or pos.magic != self.magic:
                continue

            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                log.warning(f"Cannot close {symbol} ticket={pos.ticket}: no tick")
                continue

            price = tick.bid if side == OrderSide.BUY else tick.ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": pos.volume,
                "type": opposite_type,
                "position": pos.ticket,
                "price": price,
                "deviation": self.deviation,
                "magic": self.magic,
                "comment": "SMC Bot Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            label = f"CLOSE {symbol} ticket={pos.ticket}"
            result = self._send_request(request, label)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                log.info(f"{label}: closed successfully")
                return True

            log.warning(f"{label}: close failed")
            return False

        log.warning(f"Cannot close {symbol} {side.value}: no matching position found")
        return False

    def get_positions(self) -> list[Position]:
        positions = mt5.positions_get()
        if positions is None:
            return []

        result = []
        for pos in positions:
            if pos.magic != self.magic:
                continue
            tick = mt5.symbol_info_tick(pos.symbol)
            current_price = tick.bid if pos.type == 0 else tick.ask

            result.append(Position(
                symbol=pos.symbol,
                side=OrderSide.BUY if pos.type == 0 else OrderSide.SELL,
                volume=pos.volume,
                entry_price=pos.price_open,
                current_price=current_price,
                stop_loss=pos.sl or 0,
                take_profit=pos.tp or 0,
                unrealized_pnl=pos.profit + pos.swap,
                realized_pnl=0.0,
            ))

        return result

    def get_account_balance(self) -> float:
        info = mt5.account_info()
        if info is None:
            log.error("Failed to get account info from MT5")
            return 0.0
        return info.balance
