"""
Train / retrain the ML signal filter — one model per symbol.

Run from project root:
    python src/ml/train.py                         # all 7 pairs
    python src/ml/train.py EURUSD                  # single pair
    python src/ml/train.py EURUSD GBPUSD USDJPY    # specific pairs

Reads:  data/{SYMBOL}_H1_ML.csv  per symbol
Writes: models/{SYMBOL}_filter.pkl
        models/{SYMBOL}_filter_features.pkl
        models/{SYMBOL}_filter_report.txt
        models/ml_filter_report.txt  (combined summary)
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ROOT       = Path(__file__).parent.parent.parent
DATA_DIR   = ROOT / "data"
MODELS_DIR = ROOT / "models"

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]

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
    if 7  <= h < 12: return 0   # London
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
    df["trend_strength"]  = df[["pct_from_sma20", "pct_from_sma50"]].abs().mean(axis=1)
    df["vol_spike"]       = (df["volume_ratio"] > 1.5).astype(int)
    df["rsi_extreme"]     = ((df["rsi"] < 30) | (df["rsi"] > 70)).astype(int)
    df["atr_regime"]      = (
        pd.qcut(df["atr_pct"], 3, labels=[0, 1, 2], duplicates="drop")
        .astype(float).astype(int)
    )
    return df


def _find_data(symbol: str) -> Path | None:
    """Look for CSV data file for a symbol in various naming conventions."""
    candidates = [
        DATA_DIR / f"{symbol}_H1_ML.csv",
        DATA_DIR / f"{symbol}_H1.csv",
        DATA_DIR / f"{symbol}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def train_symbol(symbol: str) -> dict | None:
    """
    Train one XGBoost model for a single symbol.
    Returns a summary dict, or None if no data found.
    """
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, classification_report

    data_path = _find_data(symbol)
    if data_path is None:
        log.warning(f"{symbol}: no CSV found in {DATA_DIR} — skipping")
        return None

    log.info(f"{symbol}: loading data from {data_path}")
    df = pd.read_csv(data_path, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Compute features if not already present
    if "rsi" not in df.columns:
        if "time" in df.columns:
            df = df.set_index("time")
        df = add_features(df)
        df = df.reset_index()
    else:
        df["hour"]    = pd.to_datetime(df["time"]).dt.hour
        df["dow"]     = pd.to_datetime(df["time"]).dt.dayofweek
        df["session"] = df["hour"].map(_session)
        df["momentum_alignment"] = (
            (df.get("macd_hist", df.get("macd", 0)) > 0).astype(int) +
            (df["rsi"] > 50).astype(int) +
            (df.get("bb_position", 0.5) > 0.5).astype(int)
        )
        df["trend_strength"] = df[["pct_from_sma20", "pct_from_sma50"]].abs().mean(axis=1)
        df["vol_spike"]      = (df.get("volume_ratio", 1) > 1.5).astype(int)
        df["rsi_extreme"]    = ((df["rsi"] < 30) | (df["rsi"] > 70)).astype(int)
        df["atr_regime"]     = (
            pd.qcut(df["atr_pct"], 3, labels=[0, 1, 2], duplicates="drop")
            .astype(float).astype(int)
        )
        if "macd_hist" not in df.columns and "macd" in df.columns and "macd_signal" in df.columns:
            df["macd_hist"] = df["macd"] - df["macd_signal"]

    df = df.dropna(subset=FEATURES + ["Target"])
    if len(df) < 200:
        log.warning(f"{symbol}: only {len(df)} samples after dropna — skipping (need >= 200)")
        return None

    log.info(f"{symbol}: {len(df)} samples")

    X = df[FEATURES]
    y = df["Target"].astype(int)

    n       = len(df)
    tr_end  = int(n * 0.70)
    val_end = int(n * 0.85)

    X_tr,  y_tr  = X.iloc[:tr_end],       y.iloc[:tr_end]
    X_val, y_val = X.iloc[tr_end:val_end], y.iloc[tr_end:val_end]
    X_te,  y_te  = X.iloc[val_end:],      y.iloc[val_end:]

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
    report   = classification_report(y_te, model.predict(X_te), target_names=["Loss", "Win"])

    log.info(f"{symbol}: AUC={auc_te:.3f} | Acc={acc_te:.3f}")

    # Build report text
    lines = [
        f"=== ML Filter Training Report — {symbol} ===",
        f"Data: {data_path}",
        f"Samples: {len(df)} (train={len(X_tr)}, val={len(X_val)}, test={len(X_te)})",
        f"Test AUC:      {auc_te:.3f}",
        f"Test Accuracy: {acc_te:.3f}",
        "",
        "Confidence gate analysis (test set):",
    ]
    gate_results = {}
    for t in [0.45, 0.50, 0.52, 0.55, 0.58, 0.60]:
        mask = proba_te >= t
        if mask.sum() == 0:
            lines.append(f"  >= {t}: 0 signals")
            gate_results[t] = (0, 0.0)
        else:
            wr = y_te.values[mask].mean() * 100
            lines.append(f"  >= {t}: {mask.sum():4d} signals | WR={wr:.1f}%")
            gate_results[t] = (mask.sum(), wr)

    lines += ["", "Feature Importances:"]
    imp = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    for _, row in imp.iterrows():
        lines.append(f"  {row['feature']:25s} {row['importance']:.4f}")

    lines += ["", "Classification Report:", report]
    report_text = "\n".join(lines)

    # Save model + features + report per symbol
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path    = MODELS_DIR / f"{symbol}_filter.pkl"
    features_path = MODELS_DIR / f"{symbol}_filter_features.pkl"
    report_path   = MODELS_DIR / f"{symbol}_filter_report.txt"

    joblib.dump(model,    model_path)
    joblib.dump(FEATURES, features_path)
    report_path.write_text(report_text)

    log.info(f"{symbol}: model saved → {model_path}")

    return {
        "symbol":    symbol,
        "samples":   len(df),
        "auc":       auc_te,
        "accuracy":  acc_te,
        "gate":      gate_results,
        "data_path": str(data_path),
    }


def train_all(symbols: list[str]) -> None:
    """Train models for all requested symbols and write a combined summary."""
    results = []
    skipped = []

    for sym in symbols:
        result = train_symbol(sym)
        if result:
            results.append(result)
        else:
            skipped.append(sym)

    # Combined summary report
    summary_lines = [
        "=== ML Filter Combined Training Summary ===",
        f"Trained : {len(results)} symbols",
        f"Skipped : {len(skipped)} symbols  {skipped if skipped else ''}",
        "",
        f"{'Symbol':<10} {'Samples':>8} {'AUC':>6} {'Acc':>6}  Best threshold (WR)",
        "-" * 65,
    ]

    for r in results:
        # Find best threshold with at least 10 signals
        best_t = best_wr = 0.0
        for t, (sigs, wr) in sorted(r["gate"].items()):
            if sigs >= 10 and wr > best_wr:
                best_wr = wr
                best_t  = t
        summary_lines.append(
            f"{r['symbol']:<10} {r['samples']:>8} {r['auc']:>6.3f} {r['accuracy']:>6.3f}"
            f"  >= {best_t} → {best_wr:.1f}% WR"
        )

    if skipped:
        summary_lines += [
            "",
            "Skipped symbols (no CSV data found):",
            f"  {', '.join(skipped)}",
            f"  → Add data files to {DATA_DIR}/  e.g. GBPUSD_H1_ML.csv",
        ]

    summary_text = "\n".join(summary_lines)
    log.info("\n" + summary_text)

    summary_path = MODELS_DIR / "ml_filter_report.txt"
    summary_path.write_text(summary_text)
    log.info(f"Summary → {summary_path}")


if __name__ == "__main__":
    # Allow running with specific symbols: python src/ml/train.py EURUSD GBPUSD
    requested = [s.upper() for s in sys.argv[1:]] if len(sys.argv) > 1 else SYMBOLS
    invalid   = [s for s in requested if s not in SYMBOLS]
    if invalid:
        log.error(f"Unknown symbols: {invalid}. Valid: {SYMBOLS}")
        sys.exit(1)

    log.info(f"Training models for: {requested}")
    train_all(requested)