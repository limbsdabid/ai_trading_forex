import pandas as pd
from .base import Strategy, Signal, SignalType


class MACrossoverStrategy(Strategy):
    def __init__(self, fast_period: int = 20, slow_period: int = 50,
                 rsi_oversold: int = 30, rsi_overbought: int = 70):
        super().__init__(
            name="MA_Crossover",
            params={
                "fast_period": fast_period,
                "slow_period": slow_period,
                "rsi_oversold": rsi_oversold,
                "rsi_overbought": rsi_overbought,
            },
        )
        self._prev_fast = None
        self._prev_slow = None
        self._prev_rsi = None

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        df = self.add_indicators(data)
        if len(df) < self.params["slow_period"] + 1:
            return Signal(SignalType.HOLD, symbol, 0.0)

        last = df.iloc[-1]
        prev = df.iloc[-2]

        fast_col = f"sma_{self.params['fast_period']}"
        slow_col = f"sma_{self.params['slow_period']}"

        current_fast = last[fast_col]
        current_slow = last[slow_col]
        prev_fast = prev[fast_col]
        prev_slow = prev[slow_col]

        if pd.isna(current_fast) or pd.isna(current_slow):
            return Signal(SignalType.HOLD, symbol, 0.0)

        rsi = last.get("rsi", 50)
        price = last["close"]

        bull_cross = prev_fast <= prev_slow and current_fast > current_slow
        bear_cross = prev_fast >= prev_slow and current_fast < current_slow

        above_slow = price > current_slow
        below_slow = price < current_slow

        if bull_cross and rsi < 70:
            confidence = min(0.5 + (70 - rsi) / 100, 0.9)
            return Signal(SignalType.BUY, symbol, round(confidence, 2))

        if bear_cross and rsi > 30:
            confidence = min(0.5 + (rsi - 30) / 100, 0.9)
            return Signal(SignalType.SELL, symbol, round(confidence, 2))

        if bull_cross:
            return Signal(SignalType.CLOSE_SELL, symbol, 0.7)
        if bear_cross:
            return Signal(SignalType.CLOSE_BUY, symbol, 0.7)

        return Signal(SignalType.HOLD, symbol, 0.0)
