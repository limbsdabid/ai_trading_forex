"""Build SMC gate snapshot labels for ML dataset exports.

These labels are for dataset filtering and diagnostics only. Do not include
them directly in model feature lists unless the live inference path can provide
the same values without lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.smc import SETUP_EXPIRY_M5_CANDLES
from src.strategies.smc.structure import find_swings, find_zones, get_confluence, get_h4_bias


NEAR_ZONE_ATR_MULTIPLIER = 0.25


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    close = df["close"]
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - close.shift()).abs()
    low_close = (df["low"] - close.shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def _zone_distance(price: float, zone: pd.Series) -> float:
    if zone["bot"] <= price <= zone["top"]:
        return 0.0
    return float(min(abs(price - zone["bot"]), abs(price - zone["top"])))


def _latest_atr(df: pd.DataFrame, fallback: float) -> float:
    values = _atr(df, 14).dropna()
    if len(values) > 0 and values.iloc[-1] > 0:
        return float(values.iloc[-1])
    return fallback


def _precompute_choch_events(m5: pd.DataFrame) -> pd.DataFrame:
    """Precompute swing-break CHoCH events once for fast H1 labeling."""
    events = pd.DataFrame(index=m5.index)
    events["bullish_choch"] = False
    events["bearish_choch"] = False

    highs, lows = find_swings(m5)
    swing_high = pd.Series(np.nan, index=m5.index)
    swing_low = pd.Series(np.nan, index=m5.index)

    if len(highs) > 0:
        swing_high.loc[highs["t"]] = highs["p"].to_numpy()
    if len(lows) > 0:
        swing_low.loc[lows["t"]] = lows["p"].to_numpy()

    prev_swing_high = swing_high.ffill().shift(1)
    prev_swing_low = swing_low.ffill().shift(1)
    events["bullish_choch"] = m5["high"].gt(prev_swing_high)
    events["bearish_choch"] = m5["low"].lt(prev_swing_low)
    return events.fillna(False)


def _choch_within_expiry(
    m5_events: pd.DataFrame,
    start_time: pd.Timestamp,
    bias: str,
) -> tuple[int, int]:
    future = m5_events.loc[m5_events.index >= start_time].head(SETUP_EXPIRY_M5_CANDLES + 1)
    column = "bullish_choch" if bias == "bullish" else "bearish_choch"
    for age, (_, row) in enumerate(future.iterrows()):
        if bool(row[column]):
            return 1, age
    return 0, -1


def add_smc_setup_labels(
    h1: pd.DataFrame,
    h4: pd.DataFrame,
    m15: pd.DataFrame,
    m5: pd.DataFrame,
) -> pd.DataFrame:
    """Append live-style SMC gate snapshots to H1 rows."""
    h1 = _ensure_datetime_index(h1)
    h4 = _ensure_datetime_index(h4)
    m15 = _ensure_datetime_index(m15)
    m5 = _ensure_datetime_index(m5)
    m5_events = _precompute_choch_events(m5)
    all_obs, all_fvgs = find_zones(m15)

    records: list[dict] = []

    for ts, row in h1.iterrows():
        h4_window = h4.loc[:ts].tail(300)
        m15_window = m15.loc[:ts].tail(1000)
        m5_window = m5.loc[:ts].tail(500)

        record = {
            "time": ts,
            "g2_bias": "neutral",
            "g2_pass": 0,
            "g3_zones": 0,
            "g3_pass": 0,
            "g4_status": "FAIL",
            "g4_distance_atr": np.nan,
            "setup_candidate": 0,
            "setup_choch_within_6": 0,
            "setup_choch_age": -1,
            "setup_ready_to_trade": 0,
        }

        if len(h4_window) < 10 or len(m15_window) < 20 or len(m5_window) < 20:
            records.append(record)
            continue

        bias = get_h4_bias(h4_window)
        record["g2_bias"] = bias
        record["g2_pass"] = int(bias != "neutral")
        if bias == "neutral":
            records.append(record)
            continue

        obs = all_obs[all_obs["t"] <= ts].tail(50) if len(all_obs) > 0 else all_obs
        fvgs = all_fvgs[all_fvgs["t"] <= ts].tail(50) if len(all_fvgs) > 0 else all_fvgs
        zones = get_confluence(obs, fvgs, bias)
        record["g3_zones"] = int(len(zones))
        record["g3_pass"] = int(len(zones) > 0)
        if len(zones) == 0:
            records.append(record)
            continue

        price = float(m5_window["close"].iloc[-1])
        recent_zones = zones[zones["t"] >= m15_window.index[-100]]
        zone_pool = recent_zones if len(recent_zones) > 0 else zones

        nearest_distance = float("inf")
        for _, zone in zone_pool.iterrows():
            nearest_distance = min(nearest_distance, _zone_distance(price, zone))

        fallback_atr = float((row["high"] - row["low"]) if {"high", "low"}.issubset(h1.columns) else 0.0005)
        atr = _latest_atr(m5_window, fallback=fallback_atr)
        atr = atr if atr > 0 else fallback_atr
        record["g4_distance_atr"] = nearest_distance / atr if atr > 0 else np.nan

        if nearest_distance == 0.0:
            record["g4_status"] = "PASS"
        elif nearest_distance <= atr * NEAR_ZONE_ATR_MULTIPLIER:
            record["g4_status"] = "NEAR"

        record["setup_candidate"] = int(record["g4_status"] in {"PASS", "NEAR"})
        if record["setup_candidate"]:
            choch, age = _choch_within_expiry(m5_events, ts, bias)
            record["setup_choch_within_6"] = choch
            record["setup_choch_age"] = age
            record["setup_ready_to_trade"] = int(choch == 1)

        records.append(record)

    labels = pd.DataFrame(records).set_index("time")
    return h1.join(labels, how="left")
