"""
Train / retrain the ML signal filter — one model per symbol, or unified MTL model.

Run from project root:
    python src/ml/train.py                         # all 7 pairs (per-symbol)
    python src/ml/train.py EURUSD                  # single pair
    python src/ml/train.py EURUSD GBPUSD USDJPY    # specific pairs
    python src/ml/train.py --mtl                   # train unified MTL model

Reads:  data/{SYMBOL}_H1_ML.csv  per symbol
Writes: models/{SYMBOL}_filter.pkl (per-symbol)
        models/mtl_filter.pkl (unified MTL model)
        models/{SYMBOL}_filter_features.pkl
        models/{SYMBOL}_filter_report.txt
        models/ml_filter_report.txt  (combined summary)
"""

import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import joblib

from src.ml import correlations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

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

FEATURES_MTL = FEATURES + [
    "pair_correlation",
    "usd_index_strength",
    "cross_correlation_eur",
    "cross_correlation_gbp",
    "cross_correlation_usd",
    "pair_id_eurusd", "pair_id_gbpusd", "pair_id_usdjpy",
    "pair_id_usdchf", "pair_id_audusd", "pair_id_usdcad", "pair_id_nzdusd",
]

PAIR_ENCODER = {
    "EURUSD": 0,
    "GBPUSD": 1,
    "USDJPY": 2,
    "USDCHF": 3,
    "AUDUSD": 4,
    "USDCAD": 5,
    "NZDUSD": 6,
}

MTL_LABEL = "trade_outcome"
MTL_R_MULTIPLE = "forward_r"
MTL_FORWARD_BARS = 12
MTL_REWARD_R = 2.0


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
    try:
        df["atr_regime"] = (
            pd.qcut(df["atr_pct"], 3, labels=[0, 1, 2], duplicates="drop")
            .astype(float).fillna(1).astype(int)
        )
    except Exception:
        df["atr_regime"] = 1
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


def add_trade_outcome_labels(
    df: pd.DataFrame,
    horizon: int = MTL_FORWARD_BARS,
    reward_r: float = MTL_REWARD_R,
) -> pd.DataFrame:
    """Label rows by simulated SL/TP outcome and forward R-multiple.

    Direction is inferred from the local trend context. Long if close is above
    SMA20, otherwise short. Risk is ATR-based when available, falling back to
    the candle range. A win is first touch of +reward_r before -1R within the
    lookahead window. If neither level is touched, use terminal R at horizon.
    """
    df = df.copy()
    required_ohlc = {"open", "high", "low", "close"}
    if not required_ohlc.issubset(df.columns):
        missing = sorted(required_ohlc - set(df.columns))
        log.warning(f"Skipping trade-outcome labels: missing OHLC columns {missing}")
        df[MTL_R_MULTIPLE] = np.nan
        df[MTL_LABEL] = np.nan
        return df

    if "sma_20" in df.columns:
        direction = np.where(df["close"] >= df["sma_20"], 1, -1)
    elif "pct_from_sma20" in df.columns:
        direction = np.where(df["pct_from_sma20"] >= 0, 1, -1)
    else:
        direction = np.where(df["close"].diff().fillna(0) >= 0, 1, -1)

    risk = df.get("atr")
    if risk is None:
        risk = df["high"] - df["low"]
    risk = risk.fillna(df["high"] - df["low"]).replace(0, np.nan)
    risk = risk.fillna((df["high"] - df["low"]).rolling(14).mean())
    risk = risk.bfill().ffill()

    outcomes = np.full(len(df), np.nan)
    r_values = np.full(len(df), np.nan)
    closes = df["close"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    risks = risk.to_numpy()

    for i in range(len(df) - horizon):
        if not np.isfinite(risks[i]) or risks[i] <= 0:
            continue

        side = direction[i]
        entry = closes[i]
        stop = entry - side * risks[i]
        target = entry + side * reward_r * risks[i]
        final_r = side * (closes[i + horizon] - entry) / risks[i]

        result_r = final_r
        for j in range(i + 1, i + horizon + 1):
            hit_tp = highs[j] >= target if side == 1 else lows[j] <= target
            hit_sl = lows[j] <= stop if side == 1 else highs[j] >= stop
            if hit_tp and hit_sl:
                result_r = -1.0
                break
            if hit_sl:
                result_r = -1.0
                break
            if hit_tp:
                result_r = reward_r
                break

        r_values[i] = result_r
        outcomes[i] = 1 if result_r > 0 else 0

    df[MTL_R_MULTIPLE] = r_values
    df[MTL_LABEL] = outcomes
    return df


def expectancy_from_r(r_values: pd.Series) -> tuple[float, float, float, float]:
    """Return expectancy, win rate, average win R, and average loss R."""
    r_values = pd.Series(r_values).dropna()
    if r_values.empty:
        return 0.0, 0.0, 0.0, 0.0

    wins = r_values[r_values > 0]
    losses = r_values[r_values <= 0]
    win_rate = len(wins) / len(r_values)
    loss_rate = 1.0 - win_rate
    avg_win_r = wins.mean() if len(wins) else 0.0
    avg_loss_r = abs(losses.mean()) if len(losses) else 0.0
    expectancy = (win_rate * avg_win_r) - (loss_rate * avg_loss_r)
    return float(expectancy), float(win_rate), float(avg_win_r), float(avg_loss_r)


def load_all_symbols(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Load all symbol data into memory for correlation computation."""
    all_data = {}
    for sym in symbols:
        data_path = _find_data(sym)
        if data_path is None:
            log.warning(f"{sym}: no CSV found, skipping correlation computation for this pair")
            continue
        df = pd.read_csv(data_path, parse_dates=["time"])
        df = df.sort_values("time").reset_index(drop=True)
        if "time" in df.columns:
            df = df.set_index("time")
        all_data[sym] = df
    return all_data


def add_features_mtl(df: pd.DataFrame, symbol: str, all_symbols_data: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Add features including correlation features for MTL model."""
    df = add_features(df)

    if all_symbols_data is None or symbol not in all_symbols_data:
        # Fallback: use zeros for correlation features
        df["pair_correlation"] = 0.0
        df["usd_index_strength"] = 0.0
        df["cross_correlation_eur"] = 0.0
        df["cross_correlation_gbp"] = 0.0
        df["cross_correlation_usd"] = 0.0
    else:
        df["pair_correlation"] = correlations.pair_correlation(symbol, df, all_symbols_data)
        df["usd_index_strength"] = correlations.usd_index_strength(all_symbols_data)
        df["cross_correlation_eur"] = correlations.cross_correlation_eur(symbol, all_symbols_data)
        df["cross_correlation_gbp"] = correlations.cross_correlation_gbp(symbol, all_symbols_data)
        df["cross_correlation_usd"] = correlations.cross_correlation_usd(symbol, all_symbols_data)

    pair_idx = PAIR_ENCODER.get(symbol, 0)
    for i in range(len(SYMBOLS)):
        df[f"pair_id_{SYMBOLS[i].lower()}"] = 1 if i == pair_idx else 0

    return df


def _add_mtl_extra(df: pd.DataFrame, symbol: str,
                    all_symbols_data: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Add MTL features to a DataFrame that already has base SMC features.

    Computes:
      - Missing derived features (session, dow, momentum_alignment, etc.)
      - Correlation features (from all_symbols_data if close available)
      - Pair-id one-hot encoding
    """
    from src.ml import correlations

    # ── Ensure derived features the CSV may lack ────────────────────────
    if hasattr(df.index, "hour") and hasattr(df.index, "dayofweek"):
        if "session" not in df.columns:
            df["session"] = df.index.hour.map(_session)
        if "dow" not in df.columns:
            df["dow"] = df.index.dayofweek

    if "momentum_alignment" not in df.columns:
        df["momentum_alignment"] = (
            (df.get("macd_hist", df.get("macd", 0)) > 0).astype(int) +
            (df["rsi"] > 50).astype(int) +
            (df.get("bb_position", 0.5) > 0.5).astype(int)
        )
    if "trend_strength" not in df.columns:
        df["trend_strength"] = (
            df.get("pct_from_sma20", 0).abs() +
            df.get("pct_from_sma50", 0).abs()
        ) / 2
    if "vol_spike" not in df.columns:
        df["vol_spike"] = (df.get("volume_ratio", 1) > 1.5).astype(int)
    if "rsi_extreme" not in df.columns:
        df["rsi_extreme"] = ((df["rsi"] < 30) | (df["rsi"] > 70)).astype(int)
    if "atr_regime" not in df.columns:
        if "atr_pct" in df.columns:
            try:
                df["atr_regime"] = (
                    pd.qcut(df["atr_pct"], 3, labels=[0, 1, 2], duplicates="drop")
                    .astype(float).fillna(1).astype(int)
                )
            except Exception:
                df["atr_regime"] = 1
        else:
            df["atr_regime"] = 1

    # ── Correlation features — only if close prices are available ───────
    has_close = "close" in df.columns
    corr_data = None
    if all_symbols_data and has_close:
        corr_data = {s: d for s, d in all_symbols_data.items()
                     if "close" in d.columns}

    if corr_data and symbol in corr_data:
        df["pair_correlation"] = correlations.pair_correlation(symbol, df, corr_data)
        df["usd_index_strength"] = correlations.usd_index_strength(corr_data)
        df["cross_correlation_eur"] = correlations.cross_correlation_eur(symbol, corr_data)
        df["cross_correlation_gbp"] = correlations.cross_correlation_gbp(symbol, corr_data)
        df["cross_correlation_usd"] = correlations.cross_correlation_usd(symbol, corr_data)
    else:
        df["pair_correlation"] = 0.0
        df["usd_index_strength"] = 0.0
        df["cross_correlation_eur"] = 0.0
        df["cross_correlation_gbp"] = 0.0
        df["cross_correlation_usd"] = 0.0

    pair_idx = PAIR_ENCODER.get(symbol, 0)
    for i in range(len(SYMBOLS)):
        df[f"pair_id_{SYMBOLS[i].lower()}"] = 1 if i == pair_idx else 0

    return df


def train_unified_model(symbols: list[str]) -> dict | None:
    """Train a single MTL model on all symbols combined with stratified split."""
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, classification_report

    log.info("Loading all symbol data for MTL training...")
    all_data = load_all_symbols(symbols)

    if not all_data:
        log.error("No data loaded for any symbol")
        return None

    log.info(f"Loaded {len(all_data)} symbols: {list(all_data.keys())}")

    dfs = []
    for sym in symbols:
        if sym not in all_data:
            log.warning(f"{sym}: skipping (no data)")
            continue

        df = all_data[sym].copy()
        df = df.reset_index()  # time index -> column

        has_rsi = "rsi" in df.columns
        df = df.set_index("time")
        if has_rsi:
            # CSV has pre-computed SMC features; add MTL extras only
            df = _add_mtl_extra(df, sym, all_data)
        else:
            # Raw OHLCV data — compute all features from scratch
            df = add_features_mtl(df, sym, all_data)
        df = add_trade_outcome_labels(df)
        df = df.reset_index()

        df["symbol"] = sym
        dfs.append(df)

    if not dfs:
        log.error("No valid data after feature engineering")
        return None

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.dropna(subset=FEATURES_MTL + [MTL_LABEL, MTL_R_MULTIPLE])

    log.info(f"Combined dataset: {len(combined_df)} rows across {len(set(combined_df['symbol']))} symbols")

    X = combined_df[FEATURES_MTL]
    y = combined_df[MTL_LABEL].astype(int)

    train_test_splits = []
    for sym in set(combined_df["symbol"]):
        sym_idx = combined_df["symbol"] == sym
        sym_data = combined_df[sym_idx]
        n = len(sym_data)
        tr_end = int(n * 0.70)
        val_end = int(n * 0.85)

        train_idx = sym_data.index[:tr_end]
        val_idx = sym_data.index[tr_end:val_end]
        test_idx = sym_data.index[val_end:]

        train_test_splits.extend([
            ("train", train_idx),
            ("val", val_idx),
            ("test", test_idx),
        ])

    train_idx_list = [idx for phase, idx in train_test_splits if phase == "train"]
    val_idx_list   = [idx for phase, idx in train_test_splits if phase == "val"]
    test_idx_list  = [idx for phase, idx in train_test_splits if phase == "test"]

    def _merge_index(idxs: list[pd.Index]) -> pd.Index:
        combined = []
        for ix in idxs:
            combined.extend(ix.tolist())
        return pd.Index(combined)

    train_idx = _merge_index(train_idx_list)
    val_idx   = _merge_index(val_idx_list)
    test_idx  = _merge_index(test_idx_list)

    X_tr, y_tr = X.loc[train_idx], y.loc[train_idx]
    X_val, y_val = X.loc[val_idx], y.loc[val_idx]
    X_te, y_te = X.loc[test_idx], y.loc[test_idx]

    log.info(f"Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_te)}")

    scale_pos = y_tr.value_counts().get(0, 1) / y_tr.value_counts().get(1, 1)
    model = XGBClassifier(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.01,
        min_child_weight=20,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=2,
        scale_pos_weight=scale_pos,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=50,
        verbosity=0,
    )

    log.info("Training unified MTL model (n_est=1000, depth=8, lr=0.01, mw=20, esr=50)...")
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    proba_te = model.predict_proba(X_te)[:, 1]
    auc_te = roc_auc_score(y_te, proba_te)
    acc_te = model.score(X_te, y_te)
    expectancy_te, wr_te, avg_win_r_te, avg_loss_r_te = expectancy_from_r(
        combined_df.loc[test_idx, MTL_R_MULTIPLE]
    )

    best_iter = model.best_iteration + 1 if hasattr(model, 'best_iteration') else 600
    log.info(f"MTL Model: AUC={auc_te:.3f} | Acc={acc_te:.3f} | best_iter={best_iter}")

    lines = [
        "=== ML Filter Unified MTL Training Report ===",
        f"Symbols: {', '.join(set(combined_df['symbol']))}",
        f"Total samples: {len(combined_df)} (train={len(X_tr)}, val={len(X_val)}, test={len(X_te)})",
        f"Test AUC: {auc_te:.3f}",
        f"Test Accuracy: {acc_te:.3f}",
        f"Test Expectancy: {expectancy_te:.3f}R",
        f"Test WR: {wr_te * 100:.1f}% | Avg Win: {avg_win_r_te:.2f}R | Avg Loss: {avg_loss_r_te:.2f}R",
        "",
        "Per-symbol test performance:",
    ]

    for sym in symbols:
        sym_mask = (combined_df.loc[test_idx, "symbol"] == sym).values
        if sym_mask.sum() == 0:
            continue
        y_sym    = y_te.values[sym_mask]
        proba_sym = proba_te[sym_mask]
        auc_sym = roc_auc_score(y_sym, proba_sym) if len(np.unique(y_sym)) > 1 else 0.5
        exp_sym, wr_sym, avg_win_sym, avg_loss_sym = expectancy_from_r(
            combined_df.loc[test_idx, MTL_R_MULTIPLE].values[sym_mask]
        )
        lines.append(
            f"  {sym}: AUC={auc_sym:.3f} | WR={wr_sym * 100:.1f}% "
            f"| Exp={exp_sym:.3f}R | AvgW={avg_win_sym:.2f}R | AvgL={avg_loss_sym:.2f}R "
            f"({len(y_sym)} samples)"
        )

    lines += ["", "Feature Importances (Top 15):"]
    imp = pd.DataFrame({"feature": FEATURES_MTL, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).head(15)
    for _, row in imp.iterrows():
        lines.append(f"  {row['feature']:30s} {row['importance']:.4f}")

    report_text = "\n".join(lines)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "mtl_filter.pkl"
    features_path = MODELS_DIR / "mtl_filter_features.pkl"
    report_path = MODELS_DIR / "mtl_filter_report.txt"

    joblib.dump(model, model_path)
    joblib.dump(FEATURES_MTL, features_path)
    report_path.write_text(report_text)

    log.info(f"MTL model saved -> {model_path}")

    return {
        "model": "mtl",
        "symbols": list(set(combined_df["symbol"])),
        "samples": len(combined_df),
        "auc": auc_te,
        "accuracy": acc_te,
        "expectancy": expectancy_te,
        "test_samples": len(X_te),
    }


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
        try:
            df["atr_regime"] = (
                pd.qcut(df["atr_pct"], 3, labels=[0, 1, 2], duplicates="drop")
                .astype(float).fillna(1).astype(int)
            )
        except Exception:
            df["atr_regime"] = 1
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

    log.info(f"{symbol}: model saved -> {model_path}")

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
            f"  >= {best_t} -> {best_wr:.1f}% WR"
        )

    if skipped:
        summary_lines += [
            "",
            "Skipped symbols (no CSV data found):",
            f"  {', '.join(skipped)}",
            f"  -> Add data files to {DATA_DIR}/  e.g. GBPUSD_H1_ML.csv",
        ]

    summary_text = "\n".join(summary_lines)
    log.info("\n" + summary_text)

    summary_path = MODELS_DIR / "ml_filter_report.txt"
    summary_path.write_text(summary_text)
    log.info(f"Summary -> {summary_path}")


def tune_mtl(symbols: list[str]) -> None:
    """Grid search over MTL hyperparameters to maximise validation AUC."""
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score

    log.info("=== MTL Hyperparameter Tuning ===")
    all_data = load_all_symbols(symbols)
    if not all_data:
        log.error("No data loaded")
        return

    dfs = []
    for sym in symbols:
        if sym not in all_data:
            continue
        df = all_data[sym].copy()
        df = df.reset_index()
        has_rsi = "rsi" in df.columns
        df = df.set_index("time")
        df = _add_mtl_extra(df, sym, all_data) if has_rsi else add_features_mtl(df, sym, all_data)
        df = add_trade_outcome_labels(df)
        df = df.reset_index()
        df["symbol"] = sym
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True).dropna(subset=FEATURES_MTL + [MTL_LABEL, MTL_R_MULTIPLE])
    log.info(f"Combined: {len(combined)} rows")

    X, y = combined[FEATURES_MTL], combined[MTL_LABEL].astype(int)

    # Stratified split
    train_parts, val_parts, test_parts = [], [], []
    for sym in set(combined["symbol"]):
        idx = combined["symbol"] == sym
        sub = combined[idx]
        n = len(sub)
        tr_end, val_end = int(n * 0.70), int(n * 0.85)
        train_parts.append(sub.index[:tr_end])
        val_parts.append(sub.index[tr_end:val_end])
        test_parts.append(sub.index[val_end:])

    def _merge(idxs):
        c = []
        for ix in idxs:
            c.extend(ix.tolist())
        return pd.Index(c)

    train_idx = _merge(train_parts)
    val_idx   = _merge(val_parts)
    test_idx  = _merge(test_parts)

    X_tr, y_tr = X.loc[train_idx], y.loc[train_idx]
    X_val, y_val = X.loc[val_idx], y.loc[val_idx]
    X_te, y_te = X.loc[test_idx], y.loc[test_idx]

    scale_pos = y_tr.value_counts().get(0, 1) / y_tr.value_counts().get(1, 1)

    param_grid = [
        {"max_depth": d, "learning_rate": lr, "n_estimators": n, "min_child_weight": mw}
        for d in [4, 6, 8]
        for lr in [0.01, 0.015, 0.03]
        for n in [300, 600, 1000]
        for mw in [10, 15, 20]
    ]
    # Limit to a sensible subset
    param_grid = [
        {"max_depth": 4, "learning_rate": 0.03,  "n_estimators": 300,  "min_child_weight": 15},
        {"max_depth": 4, "learning_rate": 0.015, "n_estimators": 600,  "min_child_weight": 15},
        {"max_depth": 6, "learning_rate": 0.03,  "n_estimators": 300,  "min_child_weight": 15},
        {"max_depth": 6, "learning_rate": 0.015, "n_estimators": 600,  "min_child_weight": 15},
        {"max_depth": 6, "learning_rate": 0.01,  "n_estimators": 1000, "min_child_weight": 20},
        {"max_depth": 8, "learning_rate": 0.03,  "n_estimators": 300,  "min_child_weight": 20},
        {"max_depth": 8, "learning_rate": 0.015, "n_estimators": 600,  "min_child_weight": 20},
        {"max_depth": 8, "learning_rate": 0.01,  "n_estimators": 1000, "min_child_weight": 20},
    ]

    best_score, best_params = 0.0, {}

    for params in param_grid:
        log.info(f"Trying {params}")
        model = XGBClassifier(
            **params,
            subsample=0.7,
            colsample_bytree=0.7,
            gamma=2,
            scale_pos_weight=scale_pos,
            random_state=42,
            eval_metric="logloss",
            early_stopping_rounds=50,
            verbosity=0,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        val_proba = model.predict_proba(X_val)[:, 1]
        val_auc   = roc_auc_score(y_val, val_proba)
        te_proba  = model.predict_proba(X_te)[:, 1]
        te_auc    = roc_auc_score(y_te, te_proba)
        te_exp, te_wr, te_avg_win, te_avg_loss = expectancy_from_r(
            combined.loc[test_idx, MTL_R_MULTIPLE]
        )

        log.info(
            f"  val_auc={val_auc:.4f}  test_auc={te_auc:.4f}  "
            f"test_exp={te_exp:.3f}R  wr={te_wr * 100:.1f}%  "
            f"avgW={te_avg_win:.2f}R  avgL={te_avg_loss:.2f}R  "
            f"best_iter={model.best_iteration + 1}"
        )

        if val_auc > best_score:
            best_score = val_auc
            best_params = params

    log.info(f"=== Best: val_auc={best_score:.4f}  params={best_params} ===")

    # Train final model with best params
    log.info("Training final model with best params...")
    final = XGBClassifier(
        **best_params,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=2,
        scale_pos_weight=scale_pos,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=50,
        verbosity=0,
    )
    final.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    te_proba = final.predict_proba(X_te)[:, 1]
    te_auc   = roc_auc_score(y_te, te_proba)
    te_exp, te_wr, te_avg_win, te_avg_loss = expectancy_from_r(
        combined.loc[test_idx, MTL_R_MULTIPLE]
    )
    log.info(
        f"Final test AUC: {te_auc:.4f} | expectancy={te_exp:.3f}R | "
        f"WR={te_wr * 100:.1f}% | AvgW={te_avg_win:.2f}R | AvgL={te_avg_loss:.2f}R"
    )

    joblib.dump(final, MODELS_DIR / "mtl_filter.pkl")
    joblib.dump(FEATURES_MTL, MODELS_DIR / "mtl_filter_features.pkl")
    log.info(f"Tuned model saved -> {MODELS_DIR / 'mtl_filter.pkl'}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--mtl":
        log.info("Training unified MTL model for all symbols...")
        train_unified_model(SYMBOLS)
    elif len(sys.argv) > 1 and sys.argv[1] == "--tune":
        log.info("Running MTL hyperparameter tuning...")
        tune_mtl(SYMBOLS)
    else:
        # Allow running with specific symbols: python src/ml/train.py EURUSD GBPUSD
        requested = [s.upper() for s in sys.argv[1:]] if len(sys.argv) > 1 else SYMBOLS
        invalid = [s for s in requested if s not in SYMBOLS]
        if invalid:
            log.error(f"Unknown symbols: {invalid}. Valid: {SYMBOLS}")
            sys.exit(1)

        log.info(f"Training per-symbol models for: {requested}")
        train_all(requested)
