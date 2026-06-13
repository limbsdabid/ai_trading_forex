import logging

import pandas as pd

from .base import Strategy, Signal, SignalType
from .smc import (
    MIN_RISK_REWARD_RATIO,
    SetupMemory,
    calculate_tp_sl,
    detect_choch_m5,
    find_zones,
    get_confluence,
    get_h4_bias,
)
from src.risk import RiskManager
from src.ml.filter import MLFilter

# Module-level logger — same name used throughout the bot
_log = logging.getLogger("trading_bot")

NEAR_ZONE_ATR_MULTIPLIER = 0.25

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
        tp_sl = calculate_tp_sl(
            m5=m5,
            bias=bias,
            price=price,
            symbol=symbol,
            min_rr=MIN_RISK_REWARD_RATIO,
        )

        sl = tp_sl['sl']
        tp = tp_sl['tp']
        sl_pips = tp_sl['sl_pips']

        _log.info(
            f"{symbol}: [TP_SL] sl={sl:.5f} tp={tp:.5f} "
            f"sl_pips={tp_sl['sl_pips']:.1f} tp_pips={tp_sl['tp_pips']:.1f} "
            f"rr={tp_sl['rr']:.2f} source={tp_sl['tp_source']}"
        )

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

        setup = self._setups.pop(symbol, None)
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
                'tp_pips':   tp_sl['tp_pips'],
                'rr':        tp_sl['rr'],
                'tp_source': tp_sl['tp_source'],
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
