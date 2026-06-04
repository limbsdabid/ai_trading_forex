"""
Train / retrain the ML signal filter.

Run from project root:
    python src/ml/train.py              # trains all available pairs
    python src/ml/train.py EURUSD       # trains one specific symbol
    python src/ml/train.py data/EURUSD_H1_ML.csv  # legacy single-file usage

Reads:  data/{SYMBOL}_H1_ML.csv  for each symbol
Writes: models/{SYMBOL}_ml_filter.pkl
        models/{SYMBOL}_ml_filter_features.pkl
        models/{SYMBOL}_ml_filter_report.txt
        models/ml_filter.pkl              (kept for backward compat — copy of EURUSD)
        models/ml_filter_features.pkl     (kept for backward compat)
        models/ml_filter_report.txt       (kept for backward compat)
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
DATA_PATH    = ROOT / "data" / "EURUSD_H1_ML.csv"   # kept for backward compat
MODELS_DIR   = ROOT / "models"
MODEL_PATH   = MODELS_DIR / "ml_filter.pkl"          # kept for backward compat
FEATURES_PATH= MODELS_DIR / "ml_filter_features.pkl" # kept for backward compat
REPORT_PATH  = MODELS_DIR / "ml_filter_report.txt"   # kept for backward compat

ALL_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]

FEATURES = [
    "rsi", "rsi_lag1",
    "macd_hist", "bb_position", "bb_width",
    "atr_pct", "return_1", "return_5",
    "body_ratio", "volume_ratio",
    "session", "dow",
    "momentum_alignment", "trend_strength",
    "vol_spike", "rsi_extreme", "atr_regime",
]


def _session(h: int) -> int:
    if 7 <= h < 12:  return 0   # London
    if 12 <= h < 16: return 3   # Overlap
    if 16 <= h < 20: return 1   # NY
    return 2                     # Asian


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]

    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    msig  = macd.ewm(span=9, adjust=False).mean()

    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rsi   = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    bb_sma = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    bb_top = bb_sma + 2 * bb_std
    bb_bot = bb_sma - 2 * bb_std

    hl = df["high"] - df["low"]
    hc = (df["high"] - c.shift()).abs()
    lc = (df["low"]  - c.shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    vol_col = "tick_volume" if "tick_volume" in df.columns else "volume"
    vol_ma  = df[vol_col].rolling(20).mean()

    df["pct_from_sma20"]    = (c - sma20) / sma20 * 100
    df["pct_from_sma50"]    = (c - sma50) / sma50 * 100
    df["rsi"]               = rsi
    df["rsi_lag1"]          = rsi.shift(1)
    df["macd_hist"]         = macd - msig
    df["bb_position"]       = (c - bb_bot) / (bb_top - bb_bot).replace(0, np.nan)
    df["bb_width"]          = (bb_top - bb_bot) / bb_sma * 100
    df["atr_pct"]           = atr / c * 100
    df["return_1"]          = c.pct_change(1) * 100
    df["return_5"]          = c.pct_change(5) * 100
    df["body_ratio"]        = (df["close"] - df["open"]).abs() / hl.replace(0, np.nan)
    df["volume_ratio"]      = df[vol_col] / vol_ma.replace(0, np.nan)

    if hasattr(df.index, "hour"):
        df["session"] = df.index.hour.map(_session)
        df["dow"]     = df.index.dayofweek
    else:
        df["session"] = 0
        df["dow"]     = 0

    df["momentum_alignment"] = (
        (df["macd_hist"] > 0).astype(int) +
        (df["rsi"] > 50).astype(int) +
        (df["bb_position"] > 0.5).astype(int)
    )
    df["trend_strength"]     = df[["pct_from_sma20","pct_from_sma50"]].abs().mean(axis=1)
    df["vol_spike"]          = (df["volume_ratio"] > 1.5).astype(int)
    df["rsi_extreme"]        = ((df["rsi"] < 30) | (df["rsi"] > 70)).astype(int)
    df["atr_regime"]         = pd.qcut(df["atr_pct"], 3, labels=[0,1,2], duplicates="drop").astype(float).fillna(1).astype(int)

    return df


def train(data_path: Path = DATA_PATH, symbol: str = "EURUSD") -> None:
    """Train one model for a single symbol and save as models/{SYMBOL}_ml_filter.pkl."""
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, classification_report

    log.info(f"[{symbol}] Loading data from {data_path}")
    df = pd.read_csv(data_path, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # If the CSV already has computed features (from notebook), use them directly
    # Otherwise recompute from OHLCV
    if "rsi" not in df.columns:
        if "time" in df.columns:
            df = df.set_index("time")
        df = add_features(df)
        df = df.reset_index()
    else:
        df["hour"] = pd.to_datetime(df["time"]).dt.hour
        df["dow"]  = pd.to_datetime(df["time"]).dt.dayofweek
        df["session"] = df["hour"].apply(_session)
        df["momentum_alignment"] = (
            (df.get("macd_hist", df.get("macd", 0)) > 0).astype(int) +
            (df["rsi"] > 50).astype(int) +
            (df.get("bb_position", 0.5) > 0.5).astype(int)
        )
        df["trend_strength"] = df[["pct_from_sma20","pct_from_sma50"]].abs().mean(axis=1)
        df["vol_spike"]      = (df.get("volume_ratio", 1) > 1.5).astype(int)
        df["rsi_extreme"]    = ((df["rsi"] < 30) | (df["rsi"] > 70)).astype(int)
        df["atr_regime"]     = pd.qcut(df["atr_pct"], 3, labels=[0,1,2], duplicates="drop").astype(float).fillna(1).astype(int)
        if "macd_hist" not in df.columns and "macd" in df.columns and "macd_signal" in df.columns:
            df["macd_hist"] = df["macd"] - df["macd_signal"]

    df = df.dropna(subset=FEATURES + ["Target"])
    log.info(f"[{symbol}] Samples after dropna: {len(df)}")

    X = df[FEATURES]
    y = df["Target"].astype(int)

    n = len(df)
    tr_end  = int(n * 0.70)
    val_end = int(n * 0.85)

    X_tr,  y_tr  = X.iloc[:tr_end],       y.iloc[:tr_end]
    X_val, y_val = X.iloc[tr_end:val_end], y.iloc[tr_end:val_end]
    X_te,  y_te  = X.iloc[val_end:],       y.iloc[val_end:]

    log.info(f"[{symbol}] Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_te)}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=15,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=2,
        scale_pos_weight=y_tr.value_counts()[0] / y_tr.value_counts()[1],
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=20,
        verbosity=0,
    )

    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    proba_te = model.predict_proba(X_te)[:, 1]
    auc_te   = roc_auc_score(y_te, proba_te)
    acc_te   = model.score(X_te, y_te)
    report   = classification_report(y_te, model.predict(X_te), target_names=["Loss","Win"])

    log.info(f"[{symbol}] Test AUC:      {auc_te:.3f}")
    log.info(f"[{symbol}] Test Accuracy: {acc_te:.3f}")

    lines = [
        f"=== ML Filter Training Report — {symbol} ===",
        f"Data: {data_path}",
        f"Samples: {len(df)} (train={len(X_tr)}, val={len(X_val)}, test={len(X_te)})",
        f"Test AUC:      {auc_te:.3f}",
        f"Test Accuracy: {acc_te:.3f}",
        "",
        "Confidence gate analysis (test set):",
    ]

    for t in [0.45, 0.50, 0.52, 0.55, 0.58, 0.60]:
        mask = proba_te >= t
        if mask.sum() == 0:
            lines.append(f"  >= {t}: 0 signals")
            continue
        wr = y_te.values[mask].mean() * 100
        lines.append(f"  >= {t}: {mask.sum():4d} signals | WR={wr:.1f}%")

    lines += ["", "Feature Importances:"]
    imp = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    for _, row in imp.iterrows():
        lines.append(f"  {row['feature']:25s} {row['importance']:.4f}")

    lines += ["", "Classification Report:", report]

    report_text = "\n".join(lines)
    log.info("\n" + report_text)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Per-symbol paths
    sym_model_path    = MODELS_DIR / f"{symbol}_ml_filter.pkl"
    sym_features_path = MODELS_DIR / f"{symbol}_ml_filter_features.pkl"
    sym_report_path   = MODELS_DIR / f"{symbol}_ml_filter_report.txt"

    joblib.dump(model,    sym_model_path)
    joblib.dump(FEATURES, sym_features_path)
    sym_report_path.write_text(report_text)

    log.info(f"[{symbol}] Model   → {sym_model_path}")
    log.info(f"[{symbol}] Features→ {sym_features_path}")
    log.info(f"[{symbol}] Report  → {sym_report_path}")

    # Backward compat: keep generic ml_filter.pkl in sync with EURUSD
    if symbol == "EURUSD":
        joblib.dump(model,    MODEL_PATH)
        joblib.dump(FEATURES, FEATURES_PATH)
        REPORT_PATH.write_text(report_text)
        log.info(f"[{symbol}] Backward-compat model → {MODEL_PATH}")


def train_all() -> None:
    """Train one model per symbol for all available CSV files."""
    results = []
    for sym in ALL_SYMBOLS:
        csv_path = ROOT / "data" / f"{sym}_H1_ML.csv"
        if not csv_path.exists():
            log.warning(f"[{sym}] CSV not found, skipping: {csv_path}")
            results.append((sym, "SKIPPED"))
            continue
        try:
            train(csv_path, sym)
            results.append((sym, "OK"))
        except Exception as e:
            log.error(f"[{sym}] Training failed: {e}")
            results.append((sym, f"ERROR: {e}"))

    log.info("\n=== Training Summary ===")
    for sym, status in results:
        log.info(f"  {sym:10s} {status}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    if arg is None:
        # No args → train all available symbols
        train_all()
    elif arg.upper() in ALL_SYMBOLS:
        # e.g. python src/ml/train.py GBPUSD
        sym = arg.upper()
        csv_path = ROOT / "data" / f"{sym}_H1_ML.csv"
        train(csv_path, sym)
    else:
        # Legacy: python src/ml/train.py data/EURUSD_H1_ML.csv
        data_path = Path(arg)
        sym = data_path.stem.split("_")[0].upper()
        train(data_path, sym)