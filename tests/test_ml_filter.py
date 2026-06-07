import csv
import sys
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from src.ml import filter as ml_filter
from src.ml.filter import MLFilter


def _sample_ohlcv(rows: int = 80) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=rows, freq="h")
    close = pd.Series([1.10 + i * 0.0001 for i in range(rows)], index=index)
    return pd.DataFrame(
        {
            "open": close - 0.00005,
            "high": close + 0.0002,
            "low": close - 0.0002,
            "close": close,
            "volume": 100 + pd.Series(range(rows), index=index),
        },
        index=index,
    )


def test_constructor_accepts_symbol():
    filt = MLFilter(symbol="gbpusd")

    assert filt.symbol == "GBPUSD"


def test_load_symbol_supports_filter_and_ml_filter_names(tmp_path, monkeypatch):
    class FakeJoblib:
        @staticmethod
        def load(path):
            return f"loaded:{Path(path).name}"

    monkeypatch.setitem(sys.modules, "joblib", FakeJoblib)
    monkeypatch.setattr(ml_filter, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(ml_filter, "FALLBACK_MODEL", tmp_path / "ml_filter.pkl")
    (tmp_path / "EURUSD_filter.pkl").write_text("model")
    (tmp_path / "EURUSD_filter_features.pkl").write_text("features")
    (tmp_path / "AUDUSD_ml_filter.pkl").write_text("model")
    (tmp_path / "AUDUSD_ml_filter_features.pkl").write_text("features")

    filt = MLFilter()

    assert filt._load_symbol("EURUSD") is True
    assert filt._models["EURUSD"] == "loaded:EURUSD_filter.pkl"
    assert filt._features["EURUSD"] == "loaded:EURUSD_filter_features.pkl"
    assert filt._load_symbol("AUDUSD") is True
    assert filt._models["AUDUSD"] == "loaded:AUDUSD_ml_filter.pkl"
    assert filt._features["AUDUSD"] == "loaded:AUDUSD_ml_filter_features.pkl"


def test_mtl_returns_neutral_when_pair_cache_incomplete():
    ml_filter.SHARED_PAIR_CACHE.clear()
    filt = MLFilter(use_mtl=True)
    filt._mtl_model = Mock()

    score = filt._score_mtl(_sample_ohlcv(), "EURUSD")

    assert score == 0.5
    filt._mtl_model.predict_proba.assert_not_called()


def test_ab_logging_writes_expected_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(ml_filter, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(ml_filter, "AB_TEST_LOG", tmp_path / "ab_test_scores.csv")
    notify = Mock()
    monkeypatch.setattr(ml_filter, "send_telegram_message", notify)

    filt = MLFilter(symbol="EURUSD")
    filt._log_ab("EURUSD", 0.61, 0.57, "BUY")

    with open(tmp_path / "ab_test_scores.csv", newline="") as f:
        rows = list(csv.reader(f))

    assert rows[0] == ["timestamp", "symbol", "old_score", "mtl_score", "signal_type"]
    assert rows[1][1:] == ["EURUSD", "0.61", "0.57", "BUY"]
    notify.assert_called_once_with(
        "🤖 MTL Shadow Signal: EURUSD | Action: BUY | Prob: 57.0%"
    )
