"""Export H1 ML datasets while preserving raw OHLC columns.

Use this when a pair CSV is missing open/high/low/close, for example:
    python src/ml/export_pair_data.py EURUSD --overwrite

By default the script requires a real MT5 connection. Pass --allow-simulated
only for pipeline testing, not for training production models.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.provider import DataProvider, FOREX_MAJORS
from src.config import Config
from src.ml.train import add_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = ROOT / "data"
REQUIRED_OHLC = ["open", "high", "low", "close"]
RAW_COLUMNS = ["open", "high", "low", "close", "volume", "spread", "real_volume"]
BASE_FEATURE_COLUMNS = [
    "pct_from_sma20",
    "pct_from_sma50",
    "rsi",
    "rsi_lag1",
    "macd_hist",
    "bb_position",
    "bb_width",
    "atr_pct",
    "return_1",
    "return_5",
    "body_ratio",
    "volume_ratio",
    "session",
    "dow",
    "momentum_alignment",
    "trend_strength",
    "vol_spike",
    "rsi_extreme",
    "atr_regime",
    "is_london_session",
    "is_ny_session",
    "hour_sin",
    "hour_cos",
]


def existing_file_has_ohlc(path: Path) -> bool:
    if not path.exists():
        return False
    columns = pd.read_csv(path, nrows=0).columns
    return set(REQUIRED_OHLC).issubset(columns)


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df.iloc[:-1]


def export_symbol(provider: DataProvider, symbol: str, timeframe: str, bars: int) -> int:
    price_data = provider.fetch_rates(symbol, timeframe, bars)
    if price_data is None or price_data.data.empty:
        raise RuntimeError(f"{symbol}: no price data returned")

    df = price_data.data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" not in df.columns:
            raise ValueError(f"{symbol}: missing DatetimeIndex or time column")
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
    df = df.sort_index()

    missing_ohlc = [col for col in REQUIRED_OHLC if col not in df.columns]
    if missing_ohlc:
        raise ValueError(f"{symbol}: fetched data missing OHLC columns {missing_ohlc}")

    df = add_features(df)
    df = add_target(df)

    output_columns = (
        [col for col in RAW_COLUMNS if col in df.columns]
        + BASE_FEATURE_COLUMNS
        + ["Target"]
    )
    output_columns = [col for col in output_columns if col in df.columns]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"{symbol}_{timeframe}_ML.csv"
    df[output_columns].dropna().to_csv(out_path, index_label="time")
    return len(df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export OHLC-preserving ML datasets.")
    parser.add_argument("symbols", nargs="*", default=FOREX_MAJORS)
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--bars", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-simulated", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbols = [symbol.upper() for symbol in args.symbols]
    invalid = [symbol for symbol in symbols if symbol not in FOREX_MAJORS]
    if invalid:
        raise ValueError(f"Unknown symbols: {invalid}. Valid symbols: {FOREX_MAJORS}")

    config = Config.from_env()
    provider = DataProvider(
        mt5_login=config.mt5_login,
        mt5_password=config.mt5_password,
        mt5_server=config.mt5_server,
    )
    connected = provider.connect_mt5()
    if not connected and not args.allow_simulated:
        log.error("MT5 is not connected. Refusing to export simulated training data.")
        log.error("Start MT5/login first, or pass --allow-simulated for pipeline testing only.")
        return 1

    try:
        for symbol in symbols:
            out_path = DATA_DIR / f"{symbol}_{args.timeframe}_ML.csv"
            has_ohlc = existing_file_has_ohlc(out_path)
            if out_path.exists() and has_ohlc and not args.overwrite:
                log.info(f"{symbol}: existing file already has OHLC, skipping")
                continue

            if out_path.exists() and not has_ohlc:
                log.info(f"{symbol}: existing file missing OHLC, regenerating")
            elif args.overwrite:
                log.info(f"{symbol}: overwriting existing dataset")
            else:
                log.info(f"{symbol}: exporting new dataset")

            rows = export_symbol(provider, symbol, args.timeframe, args.bars)
            log.info(f"{symbol}: saved {rows:,} rows -> {out_path}")
            time.sleep(0.3)
    finally:
        provider.disconnect_mt5()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
