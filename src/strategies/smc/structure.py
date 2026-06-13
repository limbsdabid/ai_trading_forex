import pandas as pd


def find_swings(df):
    highs, lows = [], []
    for i in range(2, len(df) - 2):
        if df['high'].iloc[i] == df['high'].iloc[i-2:i+3].max():
            highs.append({'t': df.index[i], 'p': df['high'].iloc[i]})
        if df['low'].iloc[i] == df['low'].iloc[i-2:i+3].min():
            lows.append({'t': df.index[i], 'p': df['low'].iloc[i]})
    return (
        pd.DataFrame(highs) if highs else pd.DataFrame(),
        pd.DataFrame(lows) if lows else pd.DataFrame(),
    )


def get_h4_bias(df):
    if len(df) < 10:
        return 'neutral'
    highs, lows = find_swings(df)
    if len(highs) < 2 or len(lows) < 2:
        return 'neutral'
    close = df['close'].iloc[-1]
    prev_high = highs['p'].iloc[-2]
    prev_low = lows['p'].iloc[-2]
    if close > prev_high:
        return 'bullish'
    if close < prev_low:
        return 'bearish'
    return 'neutral'


def find_zones(df, min_gap=0.00005, impulse_min=0.0010):
    fvgs, obs = [], []

    for i in range(2, len(df)):
        c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
        if c3['low'] > c1['high']:
            fvgs.append({
                't': df.index[i], 'd': 'bullish',
                'top': c3['low'], 'bot': c1['high'],
                'mid': (c3['low'] + c1['high']) / 2,
            })
        if c3['high'] < c1['low']:
            fvgs.append({
                't': df.index[i], 'd': 'bearish',
                'top': c1['low'], 'bot': c3['high'],
                'mid': (c1['low'] + c3['high']) / 2,
            })

    for i in range(1, len(df)):
        prev, curr = df.iloc[i-1], df.iloc[i]
        if prev['close'] < prev['open'] and curr['close'] - curr['open'] >= impulse_min:
            obs.append({
                't': df.index[i], 'd': 'bullish',
                'top': max(prev['open'], prev['close']),
                'bot': min(prev['open'], prev['close']),
                'mid': (max(prev['open'], prev['close']) + min(prev['open'], prev['close'])) / 2,
            })
        if prev['close'] > prev['open'] and curr['open'] - curr['close'] >= impulse_min:
            obs.append({
                't': df.index[i], 'd': 'bearish',
                'top': max(prev['open'], prev['close']),
                'bot': min(prev['open'], prev['close']),
                'mid': (max(prev['open'], prev['close']) + min(prev['open'], prev['close'])) / 2,
            })

    df_o = pd.DataFrame(obs) if obs else pd.DataFrame()
    df_f = pd.DataFrame(fvgs) if fvgs else pd.DataFrame()
    return df_o, df_f


def get_confluence(obs, fvg, bias, max_dist=0.0020):
    """
    Find zones where Order Blocks and FVGs align.
    Falls back to OB-only zones if no OB+FVG confluence is found.
    """
    if bias != 'neutral':
        obs = obs[obs['d'] == bias].copy() if len(obs) > 0 else pd.DataFrame()
        fvg = fvg[fvg['d'] == bias].copy() if len(fvg) > 0 else pd.DataFrame()

    zones = []

    if len(obs) > 0 and len(fvg) > 0:
        for _, o in obs.iterrows():
            for _, f in fvg.iterrows():
                if o['d'] != f['d']:
                    continue
                if abs(o['mid'] - f['mid']) <= max_dist:
                    zones.append({
                        't': max(o['t'], f['t']),
                        'd': o['d'],
                        'top': max(o['top'], f['top']),
                        'bot': min(o['bot'], f['bot']),
                        'mid': (max(o['top'], f['top']) + min(o['bot'], f['bot'])) / 2,
                        'type': 'confluence',
                    })

    if len(zones) == 0 and len(obs) > 0:
        for _, o in obs.iterrows():
            zones.append({
                't': o['t'],
                'd': o['d'],
                'top': o['top'],
                'bot': o['bot'],
                'mid': o['mid'],
                'type': 'ob_only',
            })

    return pd.DataFrame(zones).sort_values('t') if zones else pd.DataFrame()


def detect_choch_m5(m5_avail, bias):
    if len(m5_avail) < 20:
        return False

    highs, lows = find_swings(m5_avail)

    if bias == 'bullish':
        if len(lows) >= 3:
            hl1, hl2, hl3 = lows['p'].iloc[-3], lows['p'].iloc[-2], lows['p'].iloc[-1]
            if hl2 > hl1 and hl3 > hl2:
                recent_highs = highs[highs['t'] > lows['t'].iloc[-2]]
                if len(recent_highs) > 0 and m5_avail['high'].iloc[-1] > recent_highs['p'].iloc[-1]:
                    return True
        if len(highs) >= 2 and len(lows) >= 1:
            last_high = highs['p'].iloc[-1]
            if highs['t'].iloc[-1] > lows['t'].iloc[-1]:
                if m5_avail['high'].iloc[-1] > last_high:
                    return True

    elif bias == 'bearish':
        if len(highs) >= 3:
            lh1, lh2, lh3 = highs['p'].iloc[-3], highs['p'].iloc[-2], highs['p'].iloc[-1]
            if lh2 < lh1 and lh3 < lh2:
                recent_lows = lows[lows['t'] > highs['t'].iloc[-2]]
                if len(recent_lows) > 0 and m5_avail['low'].iloc[-1] < recent_lows['p'].iloc[-1]:
                    return True
        if len(lows) >= 2 and len(highs) >= 1:
            last_low = lows['p'].iloc[-1]
            if lows['t'].iloc[-1] > highs['t'].iloc[-1]:
                if m5_avail['low'].iloc[-1] < last_low:
                    return True

    return False


def get_next_liquidity(m5_avail, bias):
    if len(m5_avail) < 10:
        return None
    highs, lows = find_swings(m5_avail)
    if bias == 'bullish' and len(highs) > 0:
        return highs['p'].iloc[-1]
    if bias == 'bearish' and len(lows) > 0:
        return lows['p'].iloc[-1]
    return None
