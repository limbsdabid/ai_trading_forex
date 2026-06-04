from dataclasses import dataclass
from typing import Optional


@dataclass
class TradeSizing:
    symbol: str
    volume: float
    stop_loss: float
    take_profit: float
    risk_amount: float
    entry_price: float


class RiskManager:
    def __init__(self, account_balance: float = 10_000,
                 risk_per_trade: float = 0.02,
                 max_daily_risk: float = 0.06,
                 max_positions: int = 5):
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade
        self.max_daily_risk = max_daily_risk
        self.max_positions = max_positions
        self._daily_risk_used = 0.0
        self._open_symbols: set[str] = set()   # tracks which symbols have open positions
        self._trades_today = 0

    @property
    def _open_positions(self) -> int:
        return len(self._open_symbols)

    def can_trade(self, symbol: str) -> tuple[bool, str]:
        """Check if a new trade is allowed for this symbol.
        Returns (allowed, reason) so callers can log why it was blocked."""
        if symbol.upper() in self._open_symbols:
            return False, f"{symbol}: already has an open position"
        if self._open_positions >= self.max_positions:
            return False, f"{symbol}: max positions reached ({self.max_positions})"
        if self._daily_risk_used >= self.max_daily_risk:
            return False, f"{symbol}: daily risk limit reached ({self.max_daily_risk*100:.0f}%)"
        return True, ""

    def calculate_size(self, entry_price: float, stop_loss: float,
                       symbol: str) -> Optional[TradeSizing]:
        allowed, reason = self.can_trade(symbol)
        if not allowed:
            return None

        risk_amount = self.account_balance * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        if price_risk == 0:
            return None

        pip_value = self._pip_value(symbol, entry_price)
        if pip_value == 0:
            return None

        sl_pips = price_risk / pip_value
        volume = risk_amount / (sl_pips * 10) if sl_pips > 0 else 0
        volume = self._normalize_volume(symbol, volume)
        if volume <= 0:
            return None

        take_profit = 0.0

        return TradeSizing(
            symbol=symbol,
            volume=round(volume, 2),
            stop_loss=round(stop_loss, 5),
            take_profit=round(take_profit, 5),
            risk_amount=round(risk_amount, 2),
            entry_price=round(entry_price, 5),
        )

    def open_trade(self, symbol: str):
        self._open_symbols.add(symbol.upper())
        self._trades_today += 1
        self._daily_risk_used += self.risk_per_trade

    def close_trade(self, symbol: str):
        self._open_symbols.discard(symbol.upper())

    def reset_daily(self):
        self._daily_risk_used = 0.0
        self._trades_today = 0

    @staticmethod
    def _pip_value(symbol: str, price: float) -> float:
        if "JPY" in symbol:
            return 0.01
        return 0.0001

    @staticmethod
    def _normalize_volume(symbol: str, volume: float) -> float:
        step = 0.01
        return max(0.01, round(volume / step) * step)

    def update_balance(self, new_balance: float):
        self.account_balance = new_balance