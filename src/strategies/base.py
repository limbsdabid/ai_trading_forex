from dataclasses import dataclass
from enum import Enum
from typing import Optional
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np


class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_BUY = "close_buy"
    CLOSE_SELL = "close_sell"


@dataclass
class Signal:
    type: SignalType
    symbol: str
    confidence: float
    timestamp: Optional[str] = None
    metadata: Optional[dict] = None


class Strategy(ABC):
    def __init__(self, name: str, params: dict = None):
        self.name = name
        self.params = params or {}

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        ...

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
        df["rsi"] = self._rsi(df["close"], 14)
        macd_line, signal_line = self._macd(df["close"])
        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        bb_upper, bb_lower = self._bollinger(df["close"], 20)
        df["bb_upper"] = bb_upper
        df["bb_lower"] = bb_lower
        df["atr"] = self._atr(df, 14)
        return df

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _macd(series: pd.Series) -> tuple:
        ema_12 = series.ewm(span=12, adjust=False).mean()
        ema_26 = series.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        return macd_line, signal_line

    @staticmethod
    def _bollinger(series: pd.Series, period: int = 20) -> tuple:
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        return upper, lower

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
