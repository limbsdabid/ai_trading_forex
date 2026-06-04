"""
ML Filter — XGBoost signal confidence scorer.

Loads pre-trained model from models/ml_filter.pkl and computes
a win-probability score for each SMC signal before it is sent to
the broker.  A score >= THRESHOLD is required to trade.

Usage inside SMCStrategy:
    from src.ml.filter import MLFilter
    ml = MLFilter()
    score = ml.score(df_h1)   # df_h1 = recent H1 bars for the symbol
    if score < 0.55:
        return HOLD
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("trading_bot")

MODEL_PATH    = Path(__file__).parent.parent.parent / "models" / "ml_filter.pkl"
FEATURES_PATH = Path(__file__).parent.parent.parent / "models" / "ml_filter_features.pkl"
MODELS_DIR    = Path(__file__).parent.parent.parent / "models"

# Default threshold — above this we allow the trade
DEFAULT_THRESHOLD = 0.55


def _model_paths(symbol: str) -> tuple[Path, Path]:
    """Return (model_path, features_path) for a given symbol.
    Falls back to the generic ml_filter.pkl if per-symbol file doesn't exist."""
    sym = symbol.upper()
    sym_model    = MODELS_DIR / f"{sym}_ml_filter.pkl"
    sym_features = MODELS_DIR / f"{sym}_ml_filter_features.pkl"
    if sym_model.exists() and sym_features.exists():
        return sym_model, sym_features
    # Fallback to generic (EURUSD-trained) model
    return MODEL_PATH, FEATURES_PATH


def _session(hour: int) -> int:
    """Encode trading session. London=0, Overlap=3, NY=1, Asian=2."""
    if 7 <= hour < 12:
        return 0   # London
    if 12 <= hour < 16:
        return 3   # London / NY overlap  (highest quality)
    if 16 <= hour < 20:
        return 1   # NY
    return 2       # Asian / off-hours


def _build_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Build the 17 SMC-context features from a DataFrame of OHLCV bars.
    df must have columns: open, high, low, close, volume (or tick_volume).
    Returns a single-row DataFrame ready for model.predict_proba(), or None on error.
    """
    try:
        df = df.copy()

        # Normalise volume column name
        if "tick_volume" in df.columns and "volume" not in df.columns:
            df.rename(columns={"tick_volume": "volume"}, inplace=True)

        if len(df) < 30:
            return None

        c = df["close"]

        # --- Trend indicators ---
        sma20 = c.rolling(20).mean()
        sma50 = c.rolling(50).mean()
        pct_from_sma20  = ((c - sma20) / sma20 * 100).iloc[-1]
        pct_from_sma50  = ((c - sma50) / sma50 * 100).iloc[-1]

        # --- Momentum ---
        delta     = c.diff()
        gain      = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
        loss      = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
        rs        = gain / loss.replace(0, np.nan)
        rsi_ser   = 100 - (100 / (1 + rs))
        rsi       = rsi_ser.iloc[-1]
        rsi_lag1  = rsi_ser.iloc[-2]

        ema12     = c.ewm(span=12, adjust=False).mean()
        ema26     = c.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = (macd_line - macd_sig).iloc[-1]

        # --- Bollinger ---
        bb_sma    = c.rolling(20).mean()
        bb_std    = c.rolling(20).std()
        bb_upper  = bb_sma + 2 * bb_std
        bb_lower  = bb_sma - 2 * bb_std
        bb_width  = ((bb_upper - bb_lower) / bb_sma * 100).iloc[-1]
        price     = c.iloc[-1]
        bb_pos    = ((price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
                     if (bb_upper.iloc[-1] - bb_lower.iloc[-1]) != 0 else 0.5)

        # --- Volatility ---
        hl   = df["high"] - df["low"]
        hc   = (df["high"] - c.shift()).abs()
        lc   = (df["low"]  - c.shift()).abs()
        tr   = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr  = tr.rolling(14).mean().iloc[-1]
        atr_pct = atr / price * 100

        # --- Returns ---
        return_1  = c.pct_change(1).iloc[-1]  * 100
        return_5  = c.pct_change(5).iloc[-1]  * 100

        # --- Candle structure ---
        body_sz   = (df["close"] - df["open"]).abs().iloc[-1]
        rng       = (df["high"] - df["low"]).iloc[-1]
        body_ratio = body_sz / rng if rng != 0 else 0.5

        # --- Volume ---
        vol       = df["volume"]
        vol_ma    = vol.rolling(20).mean().iloc[-1]
        vol_ratio = (vol.iloc[-1] / vol_ma) if vol_ma != 0 else 1.0

        # --- Time / session ---
        last_ts = df.index[-1]
        if hasattr(last_ts, "hour"):
            hour = last_ts.hour
            dow  = last_ts.dayofweek
        else:
            hour, dow = 10, 1   # fallback: London morning, Tuesday

        session = _session(hour)

        # --- Composite SMC features ---
        momentum_alignment = (
            int(macd_hist > 0) +
            int(rsi > 50) +
            int(bb_pos > 0.5)
        )
        trend_strength = (abs(pct_from_sma20) + abs(pct_from_sma50)) / 2
        vol_spike      = int(vol_ratio > 1.5)
        rsi_extreme    = int(rsi < 30 or rsi > 70)

        # ATR regime: 0=low, 1=med, 2=high  (rough quantile boundaries)
        if atr_pct < 0.08:
            atr_regime = 0
        elif atr_pct < 0.15:
            atr_regime = 1
        else:
            atr_regime = 2

        row = {
            "rsi":                  rsi,
            "rsi_lag1":             rsi_lag1,
            "macd_hist":            macd_hist,
            "bb_position":          bb_pos,
            "bb_width":             bb_width,
            "atr_pct":              atr_pct,
            "return_1":             return_1,
            "return_5":             return_5,
            "body_ratio":           body_ratio,
            "volume_ratio":         vol_ratio,
            "session":              session,
            "dow":                  dow,
            "momentum_alignment":   momentum_alignment,
            "trend_strength":       trend_strength,
            "vol_spike":            vol_spike,
            "rsi_extreme":          rsi_extreme,
            "atr_regime":           atr_regime,
        }

        return pd.DataFrame([row])

    except Exception as e:
        log.warning(f"MLFilter feature build failed: {e}")
        return None


class MLFilter:
    """
    Wraps the trained XGBoost model.

    Parameters
    ----------
    threshold : float
        Minimum win-probability required to allow a trade (default 0.55).
    symbol : str
        Currency pair (e.g. "EURUSD"). Loads models/{SYMBOL}_ml_filter.pkl
        if available, otherwise falls back to the generic ml_filter.pkl.
    model_path : Path | None
        Override model path directly (for testing). Skips symbol lookup.
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD,
                 symbol: str = "EURUSD",
                 model_path: Optional[Path] = None):
        self.threshold  = threshold
        self.symbol     = symbol.upper()
        self._model     = None
        self._features  = None
        self._available = False

        if model_path is not None:
            # Explicit override (e.g. tests)
            self._load(model_path, FEATURES_PATH)
        else:
            m_path, f_path = _model_paths(self.symbol)
            self._load(m_path, f_path)

    def _load(self, model_path: Path, features_path: Path) -> None:
        try:
            import joblib
            self._model    = joblib.load(model_path)
            self._features = joblib.load(features_path)
            self._available = True
            log.info(f"MLFilter [{self.symbol}] loaded from {model_path}")
        except FileNotFoundError:
            log.warning(
                f"MLFilter [{self.symbol}] model not found — running without ML filter. "
                "Train first: python src/ml/train.py"
            )
        except Exception as e:
            log.warning(f"MLFilter [{self.symbol}] failed to load: {e}")

    @property
    def available(self) -> bool:
        """True if the model file was loaded successfully."""
        return self._available

    def score(self, df: pd.DataFrame) -> float:
        """
        Compute win-probability for the current market context.

        Parameters
        ----------
        df : pd.DataFrame
            Recent H1 (or H4) OHLCV bars for the symbol.
            Must have open/high/low/close/volume columns.

        Returns
        -------
        float
            Win probability in [0, 1].  Returns 0.5 (neutral) if model
            is unavailable or feature extraction fails.
        """
        if not self._available:
            return 0.5   # neutral — don't block trades if model missing

        features = _build_features(df)
        if features is None:
            return 0.5

        try:
            # Ensure column order matches training
            features = features[self._features]
            prob = float(self._model.predict_proba(features)[0, 1])
            return prob
        except Exception as e:
            log.warning(f"MLFilter.score() failed: {e}")
            return 0.5

    def should_trade(self, df: pd.DataFrame) -> tuple[bool, float]:
        """
        Returns (allow_trade, confidence_score).

        Parameters
        ----------
        df : pd.DataFrame
            Recent OHLCV bars.

        Returns
        -------
        tuple[bool, float]
            (True if score >= threshold, raw score)
        """
        prob = self.score(df)
        return prob >= self.threshold, prob