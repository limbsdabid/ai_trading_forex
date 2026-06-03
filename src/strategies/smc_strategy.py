import os
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

from .base import Strategy, Signal, SignalType
from src.risk import RiskManager
from src.ml.filter import MLFilter
from src.strategies.session_filter import get_session_threshold


SPREAD_COST = 0.0001


def find_swings(df):
    highs, lows = [], []
    for i in range(2, len(df) - 2):
        if df['high'].iloc[i] == df['high'].iloc[i-2:i+3].max():
            highs.append({'t': df.index[i], 'p': df['high'].iloc[i]})
        if df['low'].iloc[i] == df['low'].iloc[i-2:i+3].min():
            lows.append({'t': df.index[i], 'p': df['low'].iloc[i]})
    return pd.DataFrame(highs) if highs else pd.DataFrame(), pd.DataFrame(lows) if lows else pd.DataFrame()


def get_h4_bias(df):
    if len(df) < 10:
        return 'neutral'
    highs, lows = find_swings(df)
    if len(highs) < 2 or len(lows) < 2:
        return 'neutral'
    close = df['close'].iloc[-1]
    prev_high = highs['p'].iloc[-2]
    prev_low = lows['p'].iloc[-2]
    if close > prev_high:
        return 'bullish'
    if close < prev_low:
        return 'bearish'
    return 'neutral'


def find_zones(df, min_gap=0.00005, impulse_min=0.0010):
    fvgs, obs = [], []
    for i in range(2, len(df)):
        c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
        if c3['low'] > c1['high']:
            fvgs.append({'t': df.index[i], 'd': 'bullish', 'top': c3['low'], 'bot': c1['high'], 'mid': (c3['low']+c1['high'])/2})
        if c3['high'] < c1['low']:
            fvgs.append({'t': df.index[i], 'd': 'bearish', 'top': c1['low'], 'bot': c3['high'], 'mid': (c1['low']+c3['high'])/2})
    for i in range(1, len(df)):
        prev, curr = df.iloc[i-1], df.iloc[i]
        if prev['close'] < prev['open'] and curr['close']-curr['open'] >= impulse_min:
            obs.append({'t': df.index[i], 'd': 'bullish', 'top': max(prev['open'],prev['close']), 'bot': min(prev['open'],prev['close']), 'mid': (max(prev['open'],prev['close'])+min(prev['open'],prev['close']))/2})
        if prev['close'] > prev['open'] and curr['open']-curr['close'] >= impulse_min:
            obs.append({'t': df.index[i], 'd': 'bearish', 'top': max(prev['open'],prev['close']), 'bot': min(prev['open'],prev['close']), 'mid': (max(prev['open'],prev['close'])+min(prev['open'],prev['close']))/2})
    df_o = pd.DataFrame(obs) if obs else pd.DataFrame()
    df_f = pd.DataFrame(fvgs) if fvgs else pd.DataFrame()
    return df_o, df_f


def get_confluence(obs, fvg, bias, max_dist=0.0005):
    if len(obs) == 0 or len(fvg) == 0:
        return pd.DataFrame()
    if bias != 'neutral':
        obs = obs[obs['d'] == bias].copy() if len(obs) > 0 else pd.DataFrame()
        fvg = fvg[fvg['d'] == bias].copy() if len(fvg) > 0 else pd.DataFrame()
    zones = []
    for _, o in obs.iterrows():
        for _, f in fvg.iterrows():
            if o['d'] != f['d']:
                continue
            if abs(o['mid'] - f['mid']) <= max_dist:
                zones.append({'t': max(o['t'], f['t']), 'd': o['d'], 'top': max(o['top'],f['top']), 'bot': min(o['bot'],f['bot']), 'mid': (max(o['top'],f['top'])+min(o['bot'],f['bot']))/2})
    return pd.DataFrame(zones).sort_values('t') if zones else pd.DataFrame()


def detect_choch_m5(m5_avail, bias):
    if len(m5_avail) < 20:
        return False
    highs, lows = find_swings(m5_avail)

    if bias == 'bullish' and len(lows) >= 3:
        # True higher lows: each successive low must be HIGHER than the previous
        hl1 = lows['p'].iloc[-3]  # oldest
        hl2 = lows['p'].iloc[-2]  # middle
        hl3 = lows['p'].iloc[-1]  # newest
        if hl2 > hl1 and hl3 > hl2:  # ascending lows = bullish structure
            # Confirm: price breaks above a recent swing high (CHoCH break)
            recent_highs = highs[highs['t'] > lows['t'].iloc[-2]]
            if len(recent_highs) > 0 and m5_avail['high'].iloc[-1] > recent_highs['p'].iloc[-1]:
                return True

    elif bias == 'bearish' and len(highs) >= 3:
        # True lower highs: each successive high must be LOWER than the previous
        lh1 = highs['p'].iloc[-3]  # oldest
        lh2 = highs['p'].iloc[-2]  # middle
        lh3 = highs['p'].iloc[-1]  # newest
        if lh2 < lh1 and lh3 < lh2:  # descending highs = bearish structure
            # Confirm: price breaks below a recent swing low (CHoCH break)
            recent_lows = lows[lows['t'] > highs['t'].iloc[-2]]
            if len(recent_lows) > 0 and m5_avail['low'].iloc[-1] < recent_lows['p'].iloc[-1]:
                return True

    return False


def get_next_liquidity(m5_avail, bias):
    if len(m5_avail) < 10:
        return None
    highs, lows = find_swings(m5_avail)
    if bias == 'bullish' and len(highs) > 0:
        return highs['p'].iloc[-1]
    elif bias == 'bearish' and len(lows) > 0:
        return lows['p'].iloc[-1]
    return None


class SMCStrategy(Strategy):
    def __init__(self, risk_manager: RiskManager, data_provider=None,
                 params: dict = None, ml_threshold: float = 0.55):
        super().__init__("smc", params)
        self.risk_manager = risk_manager
        self.data_provider = data_provider
        self.ml_filter = MLFilter(threshold=ml_threshold)
        if self.ml_filter.available:
            import logging
            logging.getLogger("trading_bot").info(
                f"MLFilter active — threshold={ml_threshold}"
            )

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        if self.data_provider is None:
            return Signal(type=SignalType.HOLD, symbol=symbol, confidence=0.0)

        current_time = datetime.now()

        h4  = self._fetch_data(symbol, 'H4',  300)
        m15 = self._fetch_data(symbol, 'M15', 1000)
        m5  = self._fetch_data(symbol, 'M5',  500)
        h1  = self._fetch_data(symbol, 'H1',  100)  # for ML filter

        if h4 is None or m15 is None or m5 is None:
            return Signal(type=SignalType.HOLD, symbol=symbol, confidence=0.0)

        bias = get_h4_bias(h4)
        if bias == 'neutral':
            return Signal(type=SignalType.HOLD, symbol=symbol, confidence=0.0)

        obs, fvgs = find_zones(m15)
        zones = get_confluence(obs, fvgs, bias)
        if len(zones) == 0:
            return Signal(type=SignalType.HOLD, symbol=symbol, confidence=0.0)

        price = m5['close'].iloc[-1]
        recent_zones = zones[zones['t'] >= m15.index[-20]]
        in_zone = False
        for _, z in recent_zones.iterrows():
            if z['bot'] <= price <= z['top']:
                in_zone = True
                break
        if not in_zone:
            return Signal(type=SignalType.HOLD, symbol=symbol, confidence=0.0)

        if not detect_choch_m5(m5, bias):
            return Signal(type=SignalType.HOLD, symbol=symbol, confidence=0.0)

        # ── ML Filter gate (session-adjusted threshold) ──────────────────
        import logging
        log = logging.getLogger("trading_bot")
        ml_score = 0.5
        if h1 is not None and self.ml_filter.available:
            ml_score = self.ml_filter.score(h1)

        session_threshold, session = get_session_threshold(self.ml_filter.threshold)
        log.debug(
            f"{symbol}: {session.description} | "
            f"base={self.ml_filter.threshold:.2f} → "
            f"threshold={session_threshold:.2f} | score={ml_score:.3f}"
        )

        if ml_score < session_threshold:
            log.info(
                f"{symbol}: ML blocked [{session.description}] "
                f"score={ml_score:.3f} < threshold={session_threshold:.2f}"
            )
            return Signal(type=SignalType.HOLD, symbol=symbol, confidence=ml_score)
        # ──────────────────────────────────────────────────────────────────
        m5_highs, m5_lows = find_swings(m5)
        if bias == 'bullish' and len(m5_lows) > 0:
            sl_price = m5_lows['p'].iloc[-1] - 0.0001
        elif bias == 'bearish' and len(m5_highs) > 0:
            sl_price = m5_highs['p'].iloc[-1] + 0.0001
        else:
            sl_price = price - 0.0001 if bias == 'bullish' else price + 0.0001

        sl_pips = abs(price - sl_price) * 10000
        sl_pips = max(round(sl_pips) + 1, 10)

        liquidity_price = get_next_liquidity(m5, bias)
        if liquidity_price is not None:
            tp_price = liquidity_price
            if bias == 'bullish' and tp_price <= price:
                tp_price = price + sl_pips * 2 * 0.0001
            elif bias == 'bearish' and tp_price >= price:
                tp_price = price - sl_pips * 2 * 0.0001
        else:
            tp_price = (price + sl_pips * 2 * 0.0001 if bias == 'bullish'
                        else price - sl_pips * 2 * 0.0001)

        sl = price - sl_pips * 0.0001 if bias == 'bullish' else price + sl_pips * 0.0001
        tp = tp_price

        sizing = self.risk_manager.calculate_size(entry_price=price, stop_loss=sl, symbol=symbol)
        if sizing is None or sizing.volume <= 0:
            return Signal(type=SignalType.HOLD, symbol=symbol, confidence=0.0)

        sizing.volume = min(sizing.volume, 1.0)

        direction   = 'buy' if bias == 'bullish' else 'sell'
        signal_type = SignalType.BUY if bias == 'bullish' else SignalType.SELL

        # Blend SMC confidence (0.7 base) with ML score
        confidence = round(0.5 * 0.7 + 0.5 * ml_score, 3)

        return Signal(
            type=signal_type,
            symbol=symbol,
            confidence=confidence,
            metadata={
                'sl':        sl,
                'tp':        tp,
                'sl_pips':   sl_pips,
                'volume':    sizing.volume,
                'bias':      bias,
                'entry':     price,
                'direction': direction,
                'ml_score':  ml_score,
            }
        )

    def _fetch_data(self, symbol: str, timeframe: str, bars: int):
        result = self.data_provider.fetch_rates(symbol, timeframe, bars)
        if result is None:
            return None
        df = result.data
        if 'time' in df.columns:
            df.set_index('time', inplace=True)
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                pass
        return df