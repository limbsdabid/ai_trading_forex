import logging
import os
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

from .base import Strategy, Signal, SignalType
from src.risk import RiskManager
from src.ml.filter import MLFilter

# Module-level logger — same name used throughout the bot
_log = logging.getLogger("trading_bot")

SPREAD_COST = 0.0001


# ─────────────────────────────────────────────────────────────────────────────
# SMC helper functions  (all logic identical to original)
# ─────────────────────────────────────────────────────────────────────────────

def find_swings(df):
    highs, lows = [], []
    for i in range(2, len(df) - 2):
        if df['high'].iloc[i] == df['high'].iloc[i-2:i+3].max():
            highs.append({'t': df.index[i], 'p': df['high'].iloc[i]})
        if df['low'].iloc[i] == df['low'].iloc[i-2:i+3].min():
            lows.append({'t': df.index[i], 'p': df['low'].iloc[i]})
    return (
        pd.DataFrame(highs) if highs else pd.DataFrame(),
        pd.DataFrame(lows)  if lows  else pd.DataFrame(),
    )


def get_h4_bias(df):
    if len(df) < 10:
        return 'neutral'
    highs, lows = find_swings(df)
    if len(highs) < 2 or len(lows) < 2:
        return 'neutral'
    close     = df['close'].iloc[-1]
    prev_high = highs['p'].iloc[-2]
    prev_low  = lows['p'].iloc[-2]
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
            fvgs.append({
                't': df.index[i], 'd': 'bullish',
                'top': c3['low'],  'bot': c1['high'],
                'mid': (c3['low'] + c1['high']) / 2,
            })
        if c3['high'] < c1['low']:
            fvgs.append({
                't': df.index[i], 'd': 'bearish',
                'top': c1['low'],  'bot': c3['high'],
                'mid': (c1['low'] + c3['high']) / 2,
            })

    for i in range(1, len(df)):
        prev, curr = df.iloc[i-1], df.iloc[i]
        if prev['close'] < prev['open'] and curr['close'] - curr['open'] >= impulse_min:
            obs.append({
                't': df.index[i], 'd': 'bullish',
                'top': max(prev['open'], prev['close']),
                'bot': min(prev['open'], prev['close']),
                'mid': (max(prev['open'], prev['close']) + min(prev['open'], prev['close'])) / 2,
            })
        if prev['close'] > prev['open'] and curr['open'] - curr['close'] >= impulse_min:
            obs.append({
                't': df.index[i], 'd': 'bearish',
                'top': max(prev['open'], prev['close']),
                'bot': min(prev['open'], prev['close']),
                'mid': (max(prev['open'], prev['close']) + min(prev['open'], prev['close'])) / 2,
            })

    df_o = pd.DataFrame(obs)  if obs  else pd.DataFrame()
    df_f = pd.DataFrame(fvgs) if fvgs else pd.DataFrame()
    return df_o, df_f


def get_confluence(obs, fvg, bias, max_dist=0.0020):
    """
    Find zones where Order Blocks and FVGs align.
    max_dist: max mid-to-mid distance to count as confluence (0.0020 = 20 pips).
    Falls back to OB-only zones if no OB+FVG confluence found.
    """
    if bias != 'neutral':
        obs = obs[obs['d'] == bias].copy() if len(obs) > 0 else pd.DataFrame()
        fvg = fvg[fvg['d'] == bias].copy() if len(fvg) > 0 else pd.DataFrame()

    zones = []

    # Primary: OB + FVG confluence
    if len(obs) > 0 and len(fvg) > 0:
        for _, o in obs.iterrows():
            for _, f in fvg.iterrows():
                if o['d'] != f['d']:
                    continue
                if abs(o['mid'] - f['mid']) <= max_dist:
                    zones.append({
                        't':    max(o['t'], f['t']),
                        'd':    o['d'],
                        'top':  max(o['top'], f['top']),
                        'bot':  min(o['bot'], f['bot']),
                        'mid':  (max(o['top'], f['top']) + min(o['bot'], f['bot'])) / 2,
                        'type': 'confluence',
                    })

    # Fallback: OB-only zones (still high-quality SMC zones)
    if len(zones) == 0 and len(obs) > 0:
        for _, o in obs.iterrows():
            zones.append({
                't':    o['t'],
                'd':    o['d'],
                'top':  o['top'],
                'bot':  o['bot'],
                'mid':  o['mid'],
                'type': 'ob_only',
            })

    return pd.DataFrame(zones).sort_values('t') if zones else pd.DataFrame()


def detect_choch_m5(m5_avail, bias):
    """
    Detect Change of Character (CHoCH) or Break of Structure (BOS) on M5.

    CHoCH = higher lows (bullish) or lower highs (bearish) + structure break.
    BOS   = simpler fallback: price breaks above/below the most recent swing.
    """
    if len(m5_avail) < 20:
        return False

    highs, lows = find_swings(m5_avail)

    if bias == 'bullish':
        # CHoCH: 3 ascending lows + break above swing high
        if len(lows) >= 3:
            hl1, hl2, hl3 = lows['p'].iloc[-3], lows['p'].iloc[-2], lows['p'].iloc[-1]
            if hl2 > hl1 and hl3 > hl2:
                recent_highs = highs[highs['t'] > lows['t'].iloc[-2]]
                if len(recent_highs) > 0 and m5_avail['high'].iloc[-1] > recent_highs['p'].iloc[-1]:
                    return True
        # BOS fallback
        if len(highs) >= 2 and len(lows) >= 1:
            last_high = highs['p'].iloc[-1]
            if highs['t'].iloc[-1] > lows['t'].iloc[-1]:
                if m5_avail['high'].iloc[-1] > last_high:
                    return True

    elif bias == 'bearish':
        # CHoCH: 3 descending highs + break below swing low
        if len(highs) >= 3:
            lh1, lh2, lh3 = highs['p'].iloc[-3], highs['p'].iloc[-2], highs['p'].iloc[-1]
            if lh2 < lh1 and lh3 < lh2:
                recent_lows = lows[lows['t'] > highs['t'].iloc[-2]]
                if len(recent_lows) > 0 and m5_avail['low'].iloc[-1] < recent_lows['p'].iloc[-1]:
                    return True
        # BOS fallback
        if len(lows) >= 2 and len(highs) >= 1:
            last_low = lows['p'].iloc[-1]
            if lows['t'].iloc[-1] > highs['t'].iloc[-1]:
                if m5_avail['low'].iloc[-1] < last_low:
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


# ─────────────────────────────────────────────────────────────────────────────
# SMCStrategy
# ─────────────────────────────────────────────────────────────────────────────

class SMCStrategy(Strategy):

    def __init__(
        self,
        risk_manager: RiskManager,
        data_provider=None,
        params: dict = None,
        ml_threshold: float = 0.55,
        ml_thresholds: dict = None,
    ):
        super().__init__("smc", params)
        self.risk_manager  = risk_manager
        self.data_provider = data_provider
        self.ml_threshold  = ml_threshold       # fallback default
        self.ml_thresholds = ml_thresholds or {}  # per-symbol overrides
        self.use_mtl       = False
        self.ab_test       = False

        # Per-symbol MLFilter cache — loaded lazily on first use per symbol.
        # Key insight: each symbol needs its own MLFilter instance that loads
        # the matching {SYMBOL}_ml_filter.pkl model file.
        self._ml_filters: dict[str, MLFilter] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_threshold(self, symbol: str) -> float:
        """Return the ML threshold for a symbol, falling back to ml_threshold."""
        return self.ml_thresholds.get(symbol.upper(), self.ml_threshold)

    def _get_ml_filter(self, symbol: str) -> MLFilter:
        """
        Lazy-load and cache the per-symbol MLFilter.

        [FIX-ML-1] symbol=symbol is now passed to MLFilter so it loads
        {SYMBOL}_ml_filter.pkl instead of the generic EURUSD model.
        Falls back to EURUSD model if the symbol-specific file is missing.
        """
        if symbol not in self._ml_filters:
            threshold = self._get_threshold(symbol)
            try:
                self._ml_filters[symbol] = MLFilter(
                    symbol=symbol,          # ← FIX: was missing, caused all
                                            #   pairs to use EURUSD model
                    threshold=threshold,
                    use_mtl=self.use_mtl,
                    ab_test=self.ab_test,
                )
                if self._ml_filters[symbol].available:
                    mode = "MTL" if self.use_mtl else symbol
                    _log.info(
                        f"[ML] MLFilter [{mode}] loaded for {symbol} "
                        f"— threshold={threshold}"
                    )
                    if self.ab_test and not self.use_mtl:
                        _log.info(f"[ML] A/B test enabled for {symbol}")
            except Exception as e:
                # [CLEAN-2] Graceful fallback: missing / corrupt pkl → EURUSD
                _log.warning(
                    f"[ML] Could not load {symbol} model ({e}), "
                    f"falling back to EURUSD model"
                )
                self._ml_filters[symbol] = MLFilter(
                    symbol="EURUSD",
                    threshold=threshold,
                    use_mtl=self.use_mtl,
                    ab_test=self.ab_test,
                )

        return self._ml_filters[symbol]

    # ─────────────────────────────────────────────────────────────────────────
    # Main signal generation
    # ─────────────────────────────────────────────────────────────────────────

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:

        if self.data_provider is None:
            return Signal(type=SignalType.HOLD, symbol=symbol, confidence=0.0)

        h4  = self._fetch_data(symbol, 'H4',  300)
        m15 = self._fetch_data(symbol, 'M15', 1000)
        m5  = self._fetch_data(symbol, 'M5',  500)
        # [FIX-ML-2] 200 H1 bars (was 100) to match training-time lookback so
        # rolling / lagged features are fully populated on the most recent row.
        h1  = self._fetch_data(symbol, 'H1',  200)
        if h1 is not None:
            MLFilter.update_global_pair_cache(symbol, h1)

        # ── Gate 1: Data ─────────────────────────────────────────────────────
        if h4 is None or m15 is None or m5 is None:
            _log.info(
                f"{symbol}: [G1-FAIL] No data "
                f"h4={h4 is not None} m15={m15 is not None} m5={m5 is not None}"
            )
            return Signal(type=SignalType.HOLD, symbol=symbol, confidence=0.0)

        # ── Gate 2: H4 Bias ──────────────────────────────────────────────────
        bias = get_h4_bias(h4)
        if bias == 'neutral':
            _log.info(f"{symbol}: [G2-FAIL] H4 bias=neutral")
            return Signal(type=SignalType.HOLD, symbol=symbol, confidence=0.0)
        _log.info(f"{symbol}: [G2-PASS] H4 bias={bias}")

        # ── Gate 3: M15 Confluent Zones ──────────────────────────────────────
        obs, fvgs = find_zones(m15)
        zones = get_confluence(obs, fvgs, bias)
        if len(zones) == 0:
            _log.info(
                f"{symbol}: [G3-FAIL] No zones — "
                f"OBs={len(obs)} FVGs={len(fvgs)}"
            )
            return Signal(type=SignalType.HOLD, symbol=symbol, confidence=0.0)
        _log.info(
            f"{symbol}: [G3-PASS] {len(zones)} zones "
            f"(OBs={len(obs)} FVGs={len(fvgs)})"
        )

        # ── Gate 4: Price in Zone ────────────────────────────────────────────
        price        = m5['close'].iloc[-1]
        recent_zones = zones[zones['t'] >= m15.index[-100]]   # ~25-hour window
        in_zone      = False
        for _, z in recent_zones.iterrows():
            if z['bot'] <= price <= z['top']:
                in_zone = True
                break

        if not in_zone:
            nearest = (
                min(abs(price - z['mid']) for _, z in zones.iterrows())
                if len(zones) > 0 else 0
            )
            _log.info(
                f"{symbol}: [G4-FAIL] Price {price:.5f} not in zone — "
                f"nearest={nearest:.5f} ({len(recent_zones)} recent zones)"
            )
            return Signal(type=SignalType.HOLD, symbol=symbol, confidence=0.0)
        _log.info(f"{symbol}: [G4-PASS] Price {price:.5f} inside zone")

        # ── Gate 5: M5 CHoCH ─────────────────────────────────────────────────
        if not detect_choch_m5(m5, bias):
            _log.info(f"{symbol}: [G5-FAIL] No CHoCH on M5 for bias={bias}")
            return Signal(type=SignalType.HOLD, symbol=symbol, confidence=0.0)
        _log.info(f"{symbol}: [G5-PASS] CHoCH confirmed on M5")

        # ── Gate 6: ML Filter ────────────────────────────────────────────────
        # Runs after all SMC structural gates pass so model inference is only
        # paid for setups that are already structurally valid.
        ml_score  = 0.5   # neutral default if H1 unavailable
        threshold = self._get_threshold(symbol)

        if h1 is not None:
            try:
                ml_filter = self._get_ml_filter(symbol)  # lazy-load + cache

                if self.ab_test and not self.use_mtl:
                    # A/B test: compare old vs MTL model scores side-by-side
                    old_score, mtl_score = ml_filter.score_both(
                        h1, symbol, signal_type="HOLD"
                    )
                    ml_score = old_score
                    allow    = ml_score >= threshold
                else:
                    allow, ml_score = ml_filter.should_trade(h1, symbol)

                _log.info(
                    f"{symbol}: [G6-{'PASS' if allow else 'FAIL'}] "
                    f"ML score={ml_score:.3f} threshold={threshold:.2f}"
                )

                if not allow:
                    return Signal(
                        type=SignalType.HOLD,
                        symbol=symbol,
                        confidence=ml_score,
                    )

            except Exception:
                # Never let a broken model file kill the scan cycle.
                # log.exception writes the full traceback to the log file.
                _log.exception(
                    f"{symbol}: [G6-ERROR] ML filter threw — "
                    f"proceeding without ML gate for this bar"
                )
        else:
            _log.warning(
                f"{symbol}: [G6-SKIP] H1 data unavailable, "
                f"ML gate skipped for this bar"
            )

        # ── SL / TP calculation ───────────────────────────────────────────────
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
            tp_price = (
                price + sl_pips * 2 * 0.0001 if bias == 'bullish'
                else price - sl_pips * 2 * 0.0001
            )

        sl = price - sl_pips * 0.0001 if bias == 'bullish' else price + sl_pips * 0.0001
        tp = tp_price

        # ── Position sizing ───────────────────────────────────────────────────
        sizing = self.risk_manager.calculate_size(
            entry_price=price, stop_loss=sl, symbol=symbol
        )
        if sizing is None or sizing.volume <= 0:
            return Signal(type=SignalType.HOLD, symbol=symbol, confidence=0.0)
        sizing.volume = min(sizing.volume, 1.0)

        direction   = 'buy' if bias == 'bullish' else 'sell'
        signal_type = SignalType.BUY if bias == 'bullish' else SignalType.SELL

        # Blend SMC structural confidence (0.7 base) with ML score (0–1)
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
            },
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

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
