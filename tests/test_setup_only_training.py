import pandas as pd

from src.ml import train


def test_filter_setup_candidates_prefers_exact_ready_column(monkeypatch):
    monkeypatch.setattr(train, "SETUP_MIN_ROWS_PER_SYMBOL", 1)
    monkeypatch.setattr(train, "SETUP_FILTER_MODE", "ready")
    df = pd.DataFrame(
        {
            "close": [1.0, 1.1, 1.2],
            "setup_candidate": [1, 1, 1],
            "setup_ready_to_trade": [0, 1, 0],
        }
    )

    filtered = train.filter_setup_candidates(df, "EURUSD", "test")

    assert len(filtered) == 1
    assert filtered.iloc[0]["close"] == 1.1


def test_filter_setup_candidates_uses_exact_candidate_without_ready(monkeypatch):
    monkeypatch.setattr(train, "SETUP_MIN_ROWS_PER_SYMBOL", 1)
    monkeypatch.setattr(train, "SETUP_FILTER_MODE", "ready")
    df = pd.DataFrame(
        {
            "close": [1.0, 1.1, 1.2],
            "setup_candidate": [0, 1, 1],
        }
    )

    filtered = train.filter_setup_candidates(df, "EURUSD", "test")

    assert filtered["close"].tolist() == [1.1, 1.2]


def test_filter_setup_candidates_respects_choch_age(monkeypatch):
    monkeypatch.setattr(train, "SETUP_MIN_ROWS_PER_SYMBOL", 1)
    monkeypatch.setattr(train, "SETUP_FILTER_MODE", "ready")
    monkeypatch.setattr(train, "SETUP_MAX_CHOCH_AGE", 3)
    df = pd.DataFrame(
        {
            "close": [1.0, 1.1, 1.2],
            "setup_candidate": [1, 1, 1],
            "setup_ready_to_trade": [1, 1, 1],
            "setup_choch_age": [1, 4, 3],
        }
    )

    filtered = train.filter_setup_candidates(df, "EURUSD", "test")

    assert filtered["close"].tolist() == [1.0, 1.2]


def test_best_thresholds_disable_negative_expectancy_symbol(monkeypatch):
    monkeypatch.setattr(train, "DISABLE_WEAK_SYMBOLS", True)
    monkeypatch.setattr(train, "MIN_PAIR_VALIDATION_EXPECTANCY", 0.0)
    probabilities = pd.Series([0.6] * 12 + [0.6] * 12)
    r_values = pd.Series([-1.0] * 12 + [1.0] * 12)
    symbols = pd.Series(["BAD"] * 12 + ["GOOD"] * 12)

    thresholds, stats, _ = train.best_expectancy_thresholds_by_symbol(
        probabilities,
        r_values,
        symbols,
        thresholds=pd.Series([0.5]).to_numpy(),
        min_trades_per_symbol=1,
    )

    assert thresholds["BAD"] == train.DISABLED_SYMBOL_THRESHOLD
    assert stats["BAD"]["disabled"] is True
    assert thresholds["GOOD"] == 0.5
    assert stats["GOOD"]["disabled"] is False
