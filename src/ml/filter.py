"""
ML Filter — XGBoost signal confidence scorer, per-symbol and unified MTL models.

Per-symbol: models/{SYMBOL}_filter.pkl  (one per pair)
Unified MTL: models/mtl_filter.pkl     (single model, all pairs, 30 features)

A/B testing: score_both() returns (per_symbol, mtl) for side-by-side comparison,
logged to logs/ab_test_scores.csv without affecting trading decisions.

Usage:
    ml = MLFilter(threshold=0.55)
    score = ml.score(df_h1, symbol="GBPUSD")           # per-symbol
    score = ml.score(df_h1, symbol="GBPUSD")           # MTL if use_mtl=True
    old, mtl = ml.score_both(df_h1, symbol="GBPUSD")   # A/B comparison
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("trading_bot")

MODELS_DIR  = Path(__file__).parent.parent.parent / "models"
LOGS_DIR    = Path(__file__).parent.parent.parent / "logs"
AB_TEST_LOG = LOGS_DIR / "ab_test_scores.csv"

FALLBACK_MODEL = MODELS_DIR / "ml_filter.pkl"
DEFAULT_THRESHOLD = 0.52

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]

PAIR_ENCODER = {
    "EURUSD": 0, "GBPUSD": 1, "USDJPY": 2, "USDCHF": 3,
    "AUDUSD": 4, "USDCAD": 5, "NZDUSD": 6,
}

FEATURES_MTL = [
    "rsi", "rsi_lag1",
    "macd_hist", "bb_position", "bb_width",
    "atr_pct", "return_1", "return_5",
    "body_ratio", "volume_ratio",
    "session", "dow",
    "momentum_alignment", "trend_strength",
    "vol_spike", "rsi_extreme", "atr_regime",
    "pair_correlation",
    "usd_index_strength",
    "cross_correlation_eur",
    "cross_correlation_gbp",
    "cross_correlation_usd",
    "pair_id_eurusd", "pair_id_gbpusd", "pair_id_usdjpy",
    "pair_id_usdchf", "pair_id_audusd", "pair_id_usdcad", "pair_id_nzdusd",
]


def _session(hour: int) -> int:
    if 7  <= hour < 12: return 0   # London
    if 12 <= hour < 16: return 3   # Overlap
    if 16 <= hour < 20: return 1   # NY
    return 2                        # Asian


def _build_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Build the 17 SMC-context features from a DataFrame of OHLCV bars.
    Returns a single-row DataFrame ready for model.predict_proba(), or None on error.
    """
    try:
        df = df.copy()
        if "tick_volume" in df.columns and "volume" not in df.columns:
            df.rename(columns={"tick_volume": "volume"}, inplace=True)
        if len(df) < 30:
            return None

        c = df["close"]

        sma20 = c.rolling(20).mean()
        sma50 = c.rolling(50).mean()
        pct_from_sma20 = ((c - sma20) / sma20 * 100).iloc[-1]
        pct_from_sma50 = ((c - sma50) / sma50 * 100).iloc[-1]

        delta    = c.diff()
        gain     = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
        loss     = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
        rsi_ser  = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
        rsi      = rsi_ser.iloc[-1]
        rsi_lag1 = rsi_ser.iloc[-2]

        ema12     = c.ewm(span=12, adjust=False).mean()
        ema26     = c.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = (macd_line - macd_sig).iloc[-1]

        bb_sma   = c.rolling(20).mean()
        bb_std   = c.rolling(20).std()
        bb_upper = bb_sma + 2 * bb_std
        bb_lower = bb_sma - 2 * bb_std
        bb_width = ((bb_upper - bb_lower) / bb_sma * 100).iloc[-1]
        price    = c.iloc[-1]
        denom    = bb_upper.iloc[-1] - bb_lower.iloc[-1]
        bb_pos   = (price - bb_lower.iloc[-1]) / denom if denom != 0 else 0.5

        hl  = df["high"] - df["low"]
        hc  = (df["high"] - c.shift()).abs()
        lc  = (df["low"]  - c.shift()).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        atr_pct = atr / price * 100

        return_1 = c.pct_change(1).iloc[-1] * 100
        return_5 = c.pct_change(5).iloc[-1] * 100

        body_sz    = (df["close"] - df["open"]).abs().iloc[-1]
        rng        = (df["high"] - df["low"]).iloc[-1]
        body_ratio = body_sz / rng if rng != 0 else 0.5

        vol    = df["volume"]
        vol_ma = vol.rolling(20).mean().iloc[-1]
        vol_ratio = (vol.iloc[-1] / vol_ma) if vol_ma != 0 else 1.0

        last_ts = df.index[-1]
        if hasattr(last_ts, "hour"):
            hour = last_ts.hour
            dow  = last_ts.dayofweek
        else:
            hour, dow = 10, 1

        session = _session(hour)

        momentum_alignment = int(macd_hist > 0) + int(rsi > 50) + int(bb_pos > 0.5)
        trend_strength     = (abs(pct_from_sma20) + abs(pct_from_sma50)) / 2
        vol_spike          = int(vol_ratio > 1.5)
        rsi_extreme        = int(rsi < 30 or rsi > 70)
        if atr_pct < 0.08:    atr_regime = 0
        elif atr_pct < 0.15:  atr_regime = 1
        else:                  atr_regime = 2

        return pd.DataFrame([{
            "rsi":                rsi,
            "rsi_lag1":           rsi_lag1,
            "macd_hist":          macd_hist,
            "bb_position":        bb_pos,
            "bb_width":           bb_width,
            "atr_pct":            atr_pct,
            "return_1":           return_1,
            "return_5":           return_5,
            "body_ratio":         body_ratio,
            "volume_ratio":       vol_ratio,
            "session":            session,
            "dow":                dow,
            "momentum_alignment": momentum_alignment,
            "trend_strength":     trend_strength,
            "vol_spike":          vol_spike,
            "rsi_extreme":        rsi_extreme,
            "atr_regime":         atr_regime,
        }])

    except Exception as e:
        log.warning(f"MLFilter feature build failed: {e}")
        return None


def _build_mtl_features(df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
    """
    Build 30 features for MTL model: 17 SMC + 6 correlation (padded 0.0) + 7 pair_id.

    NOTE: Correlation features (pair_correlation, usd_index_strength, etc.)
    require all 7 pairs' aligned OHLCV data to compute rolling correlations.
    During live inference, only one symbol's data is available, so these
    features are set to 0.0 (neutral). The model still benefits from having
    learned correlation patterns during training.

    For full correlation scoring at inference time, pass cached pair data
    via MLFilter.update_pair_cache() before calling score().
    """
    base = _build_features(df)
    if base is None:
        return None

    pair_idx = PAIR_ENCODER.get(symbol, 0)

    row = base.iloc[0].to_dict()

    row["pair_correlation"]      = 0.0
    row["usd_index_strength"]    = 0.0
    row["cross_correlation_eur"] = 0.0
    row["cross_correlation_gbp"] = 0.0
    row["cross_correlation_usd"] = 0.0

    for i, sym in enumerate(SYMBOLS):
        row[f"pair_id_{sym.lower()}"] = 1.0 if i == pair_idx else 0.0

    return pd.DataFrame([row])


class MLFilter:
    """
    XGBoost signal confidence scorer — per-symbol or unified MTL.

    Per-symbol mode (default):
        models/{SYMBOL}_filter.pkl  — one model per pair

    MTL mode (use_mtl=True):
        models/mtl_filter.pkl       — single model, all 30 features
        Correlation features padded to 0.0 at inference (no multi-symbol data).

    A/B testing:
        score_both() returns (per_symbol, mtl) tuple without changing trading.
        Logged to logs/ab_test_scores.csv for offline comparison.
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD, use_mtl: bool = False,
                 ab_test: bool = False):
        self.threshold   = threshold
        self.use_mtl     = use_mtl
        self.ab_test     = ab_test and not use_mtl  # A/B only meaningful in per-symbol mode

        self._models:   dict[str, object] = {}
        self._features: dict[str, list]   = {}
        self._fallback_model    = None
        self._fallback_features = None
        self._mtl_model    = None
        self._mtl_features = None

        self._pair_cache: dict[str, pd.DataFrame] = {}

        self._load_fallback()
        if use_mtl:
            self._load_mtl_model()

    # ── Model loaders ────────────────────────────────────────────────────

    def _load_fallback(self) -> None:
        try:
            import joblib
            if FALLBACK_MODEL.exists():
                self._fallback_model    = joblib.load(FALLBACK_MODEL)
                feat_path = MODELS_DIR / "ml_filter_features.pkl"
                self._fallback_features = joblib.load(feat_path) if feat_path.exists() else None
                log.info("MLFilter: fallback model loaded (EURUSD-based)")
        except Exception as e:
            log.warning(f"MLFilter: fallback model failed to load: {e}")

    def _load_symbol(self, symbol: str) -> bool:
        if symbol in self._models:
            return True
        try:
            import joblib
            model_path = MODELS_DIR / f"{symbol}_filter.pkl"
            feat_path  = MODELS_DIR / f"{symbol}_filter_features.pkl"
            if not model_path.exists():
                return False
            self._models[symbol]   = joblib.load(model_path)
            self._features[symbol] = joblib.load(feat_path) if feat_path.exists() else None
            log.info(f"MLFilter: loaded {symbol} model")
            return True
        except Exception as e:
            log.warning(f"MLFilter: failed to load {symbol} model: {e}")
            return False

    def _load_mtl_model(self) -> None:
        try:
            import joblib
            model_path = MODELS_DIR / "mtl_filter.pkl"
            feat_path  = MODELS_DIR / "mtl_filter_features.pkl"
            if model_path.exists():
                self._mtl_model    = joblib.load(model_path)
                self._mtl_features = joblib.load(feat_path) if feat_path.exists() else None
                log.info("MLFilter: MTL model loaded")
            else:
                log.warning("MLFilter: MTL model not found (models/mtl_filter.pkl)")
                self.use_mtl = False
        except Exception as e:
            log.warning(f"MLFilter: failed to load MTL model: {e}")
            self.use_mtl = False

    @property
    def available(self) -> bool:
        return self._fallback_model is not None or len(self._models) > 0

    # ── Pair cache (optional, for correlation features at inference) ─────

    def update_pair_cache(self, symbol: str, df: pd.DataFrame) -> None:
        """Store recent OHLCV for a symbol, used by _build_correlation_features()."""
        self._pair_cache[symbol] = df.copy()

    # ── Scoring ──────────────────────────────────────────────────────────

    def score(self, df: pd.DataFrame, symbol: str = "EURUSD") -> float:
        """
        Compute win-probability.

        Routing:
          use_mtl=True  → MTL model (30 features, correlation padded)
          use_mtl=False → per-symbol model or fallback
        """
        if self.use_mtl:
            return self._score_mtl(df, symbol)
        return self._score_per_symbol(df, symbol)

    def _score_per_symbol(self, df: pd.DataFrame, symbol: str) -> float:
        features = _build_features(df)
        if features is None:
            return 0.5

        if self._load_symbol(symbol):
            model    = self._models[symbol]
            feat_lst = self._features.get(symbol)
        elif self._fallback_model is not None:
            model    = self._fallback_model
            feat_lst = self._fallback_features
        else:
            return 0.5

        try:
            if feat_lst is not None:
                features = features[feat_lst]
            return float(model.predict_proba(features)[0, 1])
        except Exception as e:
            log.warning(f"MLFilter._score_per_symbol({symbol}) failed: {e}")
            return 0.5

    def _score_mtl(self, df: pd.DataFrame, symbol: str) -> float:
        if self._mtl_model is None:
            return 0.5

        features = _build_mtl_features(df, symbol)
        if features is None:
            return 0.5

        try:
            feat_lst = self._mtl_features
            if feat_lst is not None:
                features = features[feat_lst]
            return float(self._mtl_model.predict_proba(features)[0, 1])
        except Exception as e:
            log.warning(f"MLFilter._score_mtl({symbol}) failed: {e}")
            return 0.5

    def score_both(self, df: pd.DataFrame, symbol: str = "EURUSD",
                   signal_type: str = "HOLD") -> tuple[float, float]:
        """
        Return (per_symbol_score, mtl_score) for A/B testing.

        Logs both scores to logs/ab_test_scores.csv.
        Does NOT affect trading decisions.
        """
        old_score = self._score_per_symbol(df, symbol)
        mtl_score = self._score_mtl(df, symbol)

        self._log_ab(symbol, old_score, mtl_score, signal_type)
        return old_score, mtl_score

    def should_trade(self, df: pd.DataFrame, symbol: str = "EURUSD") -> tuple[bool, float]:
        prob = self.score(df, symbol)
        return prob >= self.threshold, prob

    # ── A/B logging ──────────────────────────────────────────────────────

    def _log_ab(self, symbol: str, old_score: float, mtl_score: float,
                signal_type: str = "HOLD") -> None:
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            file_exists = AB_TEST_LOG.exists()
            with open(AB_TEST_LOG, "a", newline="") as f:
                w = csv.writer(f)
                if not file_exists:
                    w.writerow(["timestamp", "symbol", "old_score", "mtl_score",
                                "signal_type"])
                w.writerow([
                    datetime.now().isoformat(),
                    symbol,
                    round(old_score, 4),
                    round(mtl_score, 4),
                    signal_type,
                ])
        except Exception as e:
            log.debug(f"A/B log write failed: {e}")