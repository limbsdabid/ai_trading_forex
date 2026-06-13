import pandas as pd

from .structure import find_swings, get_next_liquidity


MIN_RISK_REWARD_RATIO = 2.0
MIN_SL_PIPS = 10


def pip_size_for_symbol(symbol: str) -> float:
    return 0.01 if "JPY" in symbol.upper() else 0.0001


def calculate_tp_sl(
    m5: pd.DataFrame,
    bias: str,
    price: float,
    symbol: str,
    min_rr: float = MIN_RISK_REWARD_RATIO,
) -> dict:
    pip_size = pip_size_for_symbol(symbol)
    m5_highs, m5_lows = find_swings(m5)

    if bias == 'bullish' and len(m5_lows) > 0:
        raw_sl = m5_lows['p'].iloc[-1] - pip_size
    elif bias == 'bearish' and len(m5_highs) > 0:
        raw_sl = m5_highs['p'].iloc[-1] + pip_size
    else:
        raw_sl = (
            price - MIN_SL_PIPS * pip_size
            if bias == 'bullish'
            else price + MIN_SL_PIPS * pip_size
        )

    sl_pips = abs(price - raw_sl) / pip_size
    sl_pips = max(round(sl_pips) + 1, MIN_SL_PIPS)

    sl = (
        price - sl_pips * pip_size
        if bias == 'bullish'
        else price + sl_pips * pip_size
    )

    fallback_tp = (
        price + sl_pips * min_rr * pip_size
        if bias == 'bullish'
        else price - sl_pips * min_rr * pip_size
    )

    liquidity_price = get_next_liquidity(m5, bias)
    tp = fallback_tp
    tp_source = "fallback_min_rr"

    if liquidity_price is not None:
        valid_liquidity = (
            liquidity_price > price
            if bias == 'bullish'
            else liquidity_price < price
        )

        if valid_liquidity:
            tp_distance_pips = abs(liquidity_price - price) / pip_size
            liquidity_rr = tp_distance_pips / sl_pips if sl_pips > 0 else 0.0

            if liquidity_rr >= min_rr:
                tp = liquidity_price
                tp_source = "liquidity"
            else:
                tp = fallback_tp
                tp_source = "liquidity_too_close_adjusted"

    tp_pips = abs(tp - price) / pip_size
    rr = tp_pips / sl_pips if sl_pips > 0 else 0.0

    return {
        'sl': sl,
        'tp': tp,
        'sl_pips': sl_pips,
        'tp_pips': tp_pips,
        'rr': rr,
        'tp_source': tp_source,
    }
