"""
ML Filter — XGBoost signal confidence scorer, one model per symbol.

Loads pre-trained model from models/{SYMBOL}_filter.pkl and computes
a win-probability score for each SMC signal before it is sent to
the broker. A score >= threshold is required to trade.

Falls back to the generic ml_filter.pkl if a symbol-specific model
is not yet trained.

Usage inside SMCStrategy:
    from src.ml.filter import MLFilter
    ml = MLFilter()
    score = ml.score(df_h1, symbol="GBPUSD")
    if score < threshold:
        return HOLD
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("trading_bot")

MODELS_DIR     = Path(__file__).parent.parent.parent / "models"
FALLBACK_MODEL = MODELS_DIR / "ml_filter.pkl"          # legacy single model
DEFAULT_THRESHOLD = 0.52


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


class MLFilter:
    """
    Wraps per-symbol XGBoost models.

    Each symbol has its own model: models/{SYMBOL}_filter.pkl
    Falls back to the legacy single model if the symbol model is missing.
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self._models:   dict[str, object] = {}   # symbol → model
        self._features: dict[str, list]   = {}   # symbol → feature list
        self._fallback_model    = None
        self._fallback_features = None
        self._load_fallback()

    def _load_fallback(self) -> None:
        """Load legacy single model as fallback."""
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
        """Lazy-load the model for a specific symbol. Returns True if successful."""
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

    @property
    def available(self) -> bool:
        """True if at least the fallback model is loaded."""
        return self._fallback_model is not None or len(self._models) > 0

    def score(self, df: pd.DataFrame, symbol: str = "EURUSD") -> float:
        """
        Compute win-probability for the current market context.

        Uses the symbol-specific model if available, otherwise falls back
        to the generic EURUSD model.

        Parameters
        ----------
        df     : Recent H1 OHLCV bars for the symbol
        symbol : e.g. "GBPUSD" — selects the correct trained model

        Returns
        -------
        float  Win probability in [0, 1]. Returns 0.5 if unavailable.
        """
        features = _build_features(df)
        if features is None:
            return 0.5

        # Try symbol-specific model first
        if self._load_symbol(symbol):
            model    = self._models[symbol]
            feat_lst = self._features.get(symbol)
        elif self._fallback_model is not None:
            model    = self._fallback_model
            feat_lst = self._fallback_features
            log.debug(f"{symbol}: using fallback (EURUSD) ML model")
        else:
            return 0.5   # no model at all — don't block trades

        try:
            if feat_lst is not None:
                features = features[feat_lst]
            return float(model.predict_proba(features)[0, 1])
        except Exception as e:
            log.warning(f"MLFilter.score() failed for {symbol}: {e}")
            return 0.5

    def should_trade(self, df: pd.DataFrame, symbol: str = "EURUSD") -> tuple[bool, float]:
        """Returns (allow_trade, confidence_score)."""
        prob = self.score(df, symbol)
        return prob >= self.threshold, prob