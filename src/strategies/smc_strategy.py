import logging
import os
from dataclasses import dataclass
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
NEAR_ZONE_ATR_MULTIPLIER = 0.25
SETUP_EXPIRY_M5_CANDLES = 6


@dataclass
class SetupMemory:
    symbol: str
    bias: str
    state: str
    zone_status: str
    zone_top: float
    zone_bot: float
    zone_mid: float
    created_at: object
    last_m5_time: object
    candles_waited: int = 0
    expires_after: int = SETUP_EXPIRY_M5_CANDLES


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
        self._setups: dict[str, SetupMemory] = {}
        self._gate_stats = {
            "G4_PASS": 0,
            "G4_NEAR": 0,
            "G4_FAIL": 0,
            "G5_PASS": 0,
            "G5_WAIT": 0,
            "G5_EXPIRED": 0,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_threshold(self, symbol: str) -> float:
        """Return the ML threshold for a symbol, falling back to ml_threshold."""
        return self.ml_thresholds.get(symbol.upper(), self.ml_threshold)

    def _zone_distance(self, price: float, zone: pd.Series) -> float:
        if zone["bot"] <= price <= zone["top"]:
            return 0.0
        return min(abs(price - zone["bot"]), abs(price - zone["top"]))

    def _latest_atr(self, df: pd.DataFrame, fallback: float = 0.0005) -> float:
        try:
            atr = self._atr(df, 14).dropna()
            if len(atr) > 0 and atr.iloc[-1] > 0:
                return float(atr.iloc[-1])
        except Exception:
            pass
        return fallback

    def _remember_setup(
        self,
        symbol: str,
        bias: str,
        zone_status: str,
        zone: pd.Series,
        m5_time,
    ) -> SetupMemory:
        setup = self._setups.get(symbol)

        if setup is None or setup.bias != bias:
            setup = SetupMemory(
                symbol=symbol,
                bias=bias,
                state="WAITING_FOR_CHOCH",
                zone_status=zone_status,
                zone_top=float(zone["top"]),
                zone_bot=float(zone["bot"]),
                zone_mid=float(zone["mid"]),
                created_at=m5_time,
                last_m5_time=m5_time,
            )
            self._setups[symbol] = setup
            _log.info(
                f"{symbol}: [SETUP_CREATED] bias={bias} zone={zone_status} "
                f"top={setup.zone_top:.5f} bot={setup.zone_bot:.5f}"
            )
        elif m5_time != setup.last_m5_time:
            setup.candles_waited += 1
            setup.last_m5_time = m5_time

        return setup

    def _expire_setup(self, symbol: str, reason: str) -> None:
        setup = self._setups.pop(symbol, None)
        if setup:
            setup.state = "EXPIRED"
            self._gate_stats["G5_EXPIRED"] += 1
            _log.info(
                f"{symbol}: [SETUP_EXPIRED] reason={reason} "
                f"waited={setup.candles_waited}/{setup.expires_after}"
            )

    def _log_gate_summary(self, symbol: str) -> None:
        _log.info(
            f"{symbol}: [GATE_SUMMARY] "
            f"G4_PASS={self._gate_stats['G4_PASS']} "
            f"G4_NEAR={self._gate_stats['G4_NEAR']} "
            f"G4_FAIL={self._gate_stats['G4_FAIL']} "
            f"G5_PASS={self._gate_stats['G5_PASS']} "
            f"G5_WAIT={self._gate_stats['G5_WAIT']} "
            f"G5_EXPIRED={self._gate_stats['G5_EXPIRED']}"
        )

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

    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str,
        h1_data: pd.DataFrame | None = None,
    ) -> Signal:

        if self.data_provider is None:
            return Signal(type=SignalType.HOLD, symbol=symbol, confidence=0.0)

        h4  = self._fetch_data(symbol, 'H4',  300)
        m15 = self._fetch_data(symbol, 'M15', 1000)
        m5  = self._fetch_data(symbol, 'M5',  500)
        # [FIX-ML-2] 200 H1 bars (was 100) to match training-time lookback so
        # rolling / lagged features are fully populated on the most recent row.
        h1  = h1_data if h1_data is not None else self._fetch_data(symbol, 'H1',  200)
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

        # ── Gate 4: Price in Zone / Near Zone ────────────────────────────────
        price = m5['close'].iloc[-1]
        m5_time = m5.index[-1]
        m5_atr = self._latest_atr(m5)
        recent_zones = zones[zones['t'] >= m15.index[-100]]   # ~25-hour window
        zone_pool = recent_zones if len(recent_zones) > 0 else zones

        nearest_zone = None
        nearest_distance = float("inf")
        for _, z in zone_pool.iterrows():
            distance = self._zone_distance(price, z)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_zone = z

        if nearest_zone is None:
            self._gate_stats["G4_FAIL"] += 1
            self._expire_setup(symbol, "no_valid_zone")
            _log.info(f"{symbol}: [G4-FAIL] No usable zone")
            self._log_gate_summary(symbol)
            return Signal(type=SignalType.HOLD, symbol=symbol, confidence=0.0)

        if nearest_distance == 0.0:
            zone_status = "PASS"
            self._gate_stats["G4_PASS"] += 1
            _log.info(
                f"{symbol}: [G4-PASS] Price {price:.5f} inside zone "
                f"top={nearest_zone['top']:.5f} bot={nearest_zone['bot']:.5f}"
            )
        elif nearest_distance <= m5_atr * NEAR_ZONE_ATR_MULTIPLIER:
            zone_status = "NEAR"
            self._gate_stats["G4_NEAR"] += 1
            _log.info(
                f"{symbol}: [G4-NEAR] Price {price:.5f} near zone "
                f"distance={nearest_distance:.5f} atr={m5_atr:.5f} "
                f"limit={m5_atr * NEAR_ZONE_ATR_MULTIPLIER:.5f}"
            )
        else:
            self._gate_stats["G4_FAIL"] += 1
            self._expire_setup(symbol, "price_too_far_from_zone")
            _log.info(
                f"{symbol}: [G4-FAIL] Price {price:.5f} too far from zone "
                f"distance={nearest_distance:.5f} atr={m5_atr:.5f} "
                f"recent_zones={len(recent_zones)}"
            )
            self._log_gate_summary(symbol)
            return Signal(type=SignalType.HOLD, symbol=symbol, confidence=0.0)

        setup = self._remember_setup(symbol, bias, zone_status, nearest_zone, m5_time)

        # ── Gate 5: M5 CHoCH confirmation with setup memory ──────────────────
        if setup.candles_waited > setup.expires_after:
            self._expire_setup(symbol, "choch_timeout")
            self._log_gate_summary(symbol)
            return Signal(type=SignalType.HOLD, symbol=symbol, confidence=0.0)

        if not detect_choch_m5(m5, bias):
            setup.state = "WAITING_FOR_CHOCH"
            self._gate_stats["G5_WAIT"] += 1
            _log.info(
                f"{symbol}: [CHOCH_WAITING] bias={bias} zone={zone_status} "
                f"waited={setup.candles_waited}/{setup.expires_after}"
            )
            self._log_gate_summary(symbol)
            return Signal(type=SignalType.HOLD, symbol=symbol, confidence=0.0)

        setup.state = "READY_TO_TRADE"
        self._gate_stats["G5_PASS"] += 1
        _log.info(
            f"{symbol}: [G5-PASS] CHoCH confirmed after "
            f"{setup.candles_waited}/{setup.expires_after} M5 candles"
        )

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

        setup = self._setups.get(symbol)
        if setup:
            setup.state = "EXECUTED"

        _log.info(
            f"{symbol}: [TRADE_EXECUTED] direction={direction.upper()} "
            f"entry={price:.5f} sl={sl:.5f} tp={tp:.5f} "
            f"ml_score={ml_score:.3f} confidence={confidence:.3f}"
        )
        self._log_gate_summary(symbol)

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
