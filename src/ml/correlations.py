"""
Correlation features for MTL model — compute cross-pair relationships.

These features help the unified model understand how each pair moves relative
to others and to the USD strength basket.

Functions:
- pair_correlation(symbol, df, all_symbols_data) → correlation with average other pair
- usd_index_strength(all_symbols_data) → USD strength estimate
- cross_correlation_eur(symbol, all_symbols_data) → correlation with EURUSD
- cross_correlation_gbp(symbol, all_symbols_data) → correlation with GBPUSD
- cross_correlation_usd(symbol, all_symbols_data) → correlation with USD strength pairs
"""

import numpy as np
import pandas as pd


def pair_correlation(symbol: str, df: pd.DataFrame, all_symbols_data: dict[str, pd.DataFrame], window: int = 20) -> pd.Series:
    """
    Correlation between this symbol's returns and average of other pair returns.

    Parameters
    ----------
    symbol : str
        e.g. "EURUSD"
    df : pd.DataFrame
        This symbol's OHLCV data (must have 'close' column)
    all_symbols_data : dict[str, pd.DataFrame]
        Map of all loaded symbols: {"EURUSD": df1, "GBPUSD": df2, ...}
    window : int
        Rolling window for correlation (default 20)

    Returns
    -------
    pd.Series
        Correlation values [-1, 1], aligned with df.index
    """
    returns_this = df["close"].pct_change(fill_method=None)

    other_symbols = [s for s in all_symbols_data.keys() if s != symbol]
    if not other_symbols:
        return pd.Series(0.0, index=df.index)

    other_returns = pd.concat(
        [all_symbols_data[s]["close"].pct_change(fill_method=None) for s in other_symbols],
        axis=1
    )
    avg_other = other_returns.mean(axis=1)

    corr = returns_this.rolling(window).corr(avg_other)
    return corr.fillna(0.0)


def usd_index_strength(all_symbols_data: dict[str, pd.DataFrame]) -> pd.Series:
    """
    Estimate USD strength from major pairs using returns-based differential.

    Formula: avg_return(weak_usd_pairs) - avg_return(strong_usd_pairs)

    Interpretation:
    - Positive: USD weak (EUR, GBP, AUD, NZD up relative to JPY, CHF, CAD)
    - Negative: USD strong (JPY, CHF, CAD up relative to EUR, GBP, AUD, NZD)

    Returns-based to avoid price scale mismatch (EURUSD ~1.08 vs USDJPY ~150).

    Parameters
    ----------
    all_symbols_data : dict[str, pd.DataFrame]
        Map of all loaded symbols

    Returns
    -------
    pd.Series
        USD strength index, aligned to first symbol's index
    """
    pairs_weak_usd   = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"]
    pairs_strong_usd = ["USDJPY", "USDCHF", "USDCAD"]

    closes_w = pd.concat(
        [all_symbols_data.get(s, pd.DataFrame()).get("close", pd.Series(dtype=float))
         for s in pairs_weak_usd],
        axis=1
    )
    closes_s = pd.concat(
        [all_symbols_data.get(s, pd.DataFrame()).get("close", pd.Series(dtype=float))
         for s in pairs_strong_usd],
        axis=1
    )

    if closes_w.empty or closes_s.empty or closes_w.dropna(how="all").empty:
        idx = closes_w.index if not closes_w.empty else closes_s.index
        return pd.Series(0.0, index=idx)

    weak_rets   = closes_w.pct_change(fill_method=None)
    strong_rets = closes_s.pct_change(fill_method=None)

    index = weak_rets.mean(axis=1) - strong_rets.mean(axis=1)
    return index.fillna(0.0)


def cross_correlation_eur(symbol: str, all_symbols_data: dict[str, pd.DataFrame], window: int = 20) -> pd.Series:
    """Correlation with EURUSD returns."""
    if symbol == "EURUSD" or "EURUSD" not in all_symbols_data:
        return pd.Series(0.0, index=all_symbols_data.get(symbol, pd.DataFrame()).index)

    returns_this = all_symbols_data[symbol]["close"].pct_change(fill_method=None)
    returns_eur = all_symbols_data["EURUSD"]["close"].pct_change(fill_method=None)

    corr = returns_this.rolling(window).corr(returns_eur)
    return corr.fillna(0.0)


def cross_correlation_gbp(symbol: str, all_symbols_data: dict[str, pd.DataFrame], window: int = 20) -> pd.Series:
    """Correlation with GBPUSD returns."""
    if symbol == "GBPUSD" or "GBPUSD" not in all_symbols_data:
        return pd.Series(0.0, index=all_symbols_data.get(symbol, pd.DataFrame()).index)

    returns_this = all_symbols_data[symbol]["close"].pct_change(fill_method=None)
    returns_gbp = all_symbols_data["GBPUSD"]["close"].pct_change(fill_method=None)

    corr = returns_this.rolling(window).corr(returns_gbp)
    return corr.fillna(0.0)


def cross_correlation_usd(symbol: str, all_symbols_data: dict[str, pd.DataFrame], window: int = 20) -> pd.Series:
    """
    Correlation with USD strength pairs (USDJPY, USDCHF, USDCAD average).
    """
    usd_pairs = ["USDJPY", "USDCHF", "USDCAD"]
    returns_this = all_symbols_data.get(symbol, pd.DataFrame()).get("close", pd.Series()).pct_change(fill_method=None)

    usd_returns = pd.concat(
        [all_symbols_data.get(s, pd.DataFrame()).get("close", pd.Series()).pct_change(fill_method=None) for s in usd_pairs],
        axis=1
    )

    if returns_this.empty or usd_returns.empty:
        return pd.Series(0.0, index=returns_this.index if not returns_this.empty else usd_returns.index)

    avg_usd = usd_returns.mean(axis=1)
    corr = returns_this.rolling(window).corr(avg_usd)
    return corr.fillna(0.0)
