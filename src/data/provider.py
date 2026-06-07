from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

import pandas as pd
import numpy as np


FOREX_MAJORS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "USDCAD", "NZDUSD",
]


@dataclass
class PriceData:
    symbol: str
    timeframe: str
    data: pd.DataFrame

    def __post_init__(self):
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
        if "time" in self.data.columns:
            self.data = self.data.set_index("time")


class DataProvider:
    def __init__(self, mt5_login: Optional[int] = None,
                 mt5_password: Optional[str] = None,
                 mt5_server: Optional[str] = None):
        self.mt5_login = mt5_login
        self.mt5_password = mt5_password
        self.mt5_server = mt5_server
        self._mt5_connected = False

    def connect_mt5(self) -> bool:
        try:
            import MetaTrader5 as mt5
            if not mt5.initialize():
                return False
            if self.mt5_login and self.mt5_password:
                if not mt5.login(self.mt5_login,
                                 password=self.mt5_password,
                                 server=self.mt5_server):
                    mt5.shutdown()
                    return False
            account_info = mt5.account_info()
            if account_info is None:
                mt5.shutdown()
                return False
            if self.mt5_login and int(account_info.login) != int(self.mt5_login):
                mt5.shutdown()
                raise ValueError(
                    "MT5 account mismatch: terminal is logged into "
                    f"{account_info.login}, but MT5_LOGIN is {self.mt5_login}. "
                    "Open/login to the correct MT5 account before running the bot."
                )
            import time
            time.sleep(1)
            for sym in FOREX_MAJORS:
                mt5.symbol_select(sym, True)
            self._mt5_connected = True
            return True
        except ImportError:
            return False

    def disconnect_mt5(self):
        if self._mt5_connected:
            try:
                import MetaTrader5 as mt5
                mt5.shutdown()
            except ImportError:
                pass
            self._mt5_connected = False

    def fetch_rates(self, symbol: str, timeframe: str = "H1",
                    bars: int = 500) -> Optional[PriceData]:
        if self._mt5_connected:
            import MetaTrader5 as mt5
            tf_map = {
                "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
            }
            return self._fetch_mt5(symbol, tf_map.get(timeframe, mt5.TIMEFRAME_H1), bars)
        return self._fetch_simulated(symbol, timeframe, bars)

    def _fetch_mt5(self, symbol: str, mt5_tf: int, bars: int) -> Optional[PriceData]:
        try:
            import MetaTrader5 as mt5
            if not mt5.symbol_select(symbol, True):
                return None
            import time
            time.sleep(0.5)
            rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, bars)
            if rates is None or len(rates) == 0:
                return None
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.rename(columns={"tick_volume": "volume"}, inplace=True)
            return PriceData(symbol=symbol, timeframe=str(mt5_tf), data=df)
        except Exception:
            return None

    def _fetch_simulated(self, symbol: str, timeframe: str,
                         bars: int) -> PriceData:
        now = datetime.now()
        tf_minutes = {"M1": 1, "M5": 5, "M15": 15, "M30": 30,
                      "H1": 60, "H4": 240, "D1": 1440, "W1": 10080}
        interval = tf_minutes.get(timeframe, 60)
        times = [now - timedelta(minutes=i * interval)
                 for i in range(bars - 1, -1, -1)]
        base = {"EURUSD": 1.08, "GBPUSD": 1.26, "USDJPY": 150.0,
                "USDCHF": 0.88, "AUDUSD": 0.65, "USDCAD": 1.36,
                "NZDUSD": 0.60}.get(symbol, 1.0)
        close = base
        prices = []
        for t in times:
            change = np.random.normal(0, 0.0005)
            close *= (1 + change)
            o = close * (1 + np.random.normal(0, 0.0002))
            h = max(o, close) * (1 + abs(np.random.normal(0, 0.0003)))
            l = min(o, close) * (1 - abs(np.random.normal(0, 0.0003)))
            v = int(np.random.uniform(100, 10000))
            prices.append({
                "time": t,
                "open": round(o, 5),
                "high": round(h, 5),
                "low": round(l, 5),
                "close": round(close, 5),
                "volume": v,
            })
        df = pd.DataFrame(prices)
        return PriceData(symbol=symbol, timeframe=timeframe, data=df)
