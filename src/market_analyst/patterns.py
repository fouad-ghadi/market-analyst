"""Candlestick patterns: doji, hammer, shooting star, bullish/bearish engulfing.

Rulebook: "A pattern only matters at a level. A hammer in the middle of
nowhere is noise." So this module has two steps:
    detect()            which candles have which shape (pure geometry)
    signals_at_zones()  keep only the shapes that happen AT a zone, on the
                        correct side (bullish shapes at support, bearish at resistance)

Candle anatomy:   range = high - low        body = |close - open|
                  upper wick = high - max(open, close)
                  lower wick = min(open, close) - low
Each pattern uses only the candle itself and the one before it (no look-ahead).
"""

from __future__ import annotations

import pandas as pd

from .levels import Zone

BULLISH = ("hammer", "bullish_engulfing")
BEARISH = ("shooting_star", "bearish_engulfing")
LABELS = {
    "doji": "doji", "hammer": "hammer", "shooting_star": "shooting star",
    "bullish_engulfing": "bullish engulfing", "bearish_engulfing": "bearish engulfing",
}


def detect(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return a table of True/False columns, one per pattern, same index as df."""
    p = cfg["patterns"]
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = h - l
    body = (c - o).abs()
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    has_range = rng > 0

    out = pd.DataFrame(index=df.index)
    # Doji: almost no body, the market could not decide.
    out["doji"] = has_range & (body <= p["doji_max_body_ratio"] * rng)
    # Hammer: long lower wick (sellers pushed down, buyers pushed back up),
    # small upper wick.
    out["hammer"] = (
        has_range
        & (lower >= p["hammer_min_wick_body_ratio"] * body)
        & (upper <= p["hammer_max_opposite_wick_ratio"] * rng)
        & (lower > upper)
    )
    # Shooting star: the mirror image, long upper wick.
    out["shooting_star"] = (
        has_range
        & (upper >= p["hammer_min_wick_body_ratio"] * body)
        & (lower <= p["hammer_max_opposite_wick_ratio"] * rng)
        & (upper > lower)
    )
    # Engulfing: this candle's body fully covers the previous candle's body,
    # in the opposite direction.
    po, pc = o.shift(1), c.shift(1)
    prev_body = (pc - po).abs()
    out["bullish_engulfing"] = (
        (pc < po) & (c > o) & (o <= pc) & (c >= po) & (body > prev_body)
    )
    out["bearish_engulfing"] = (
        (pc > po) & (c < o) & (o >= pc) & (c <= po) & (body > prev_body)
    )
    return out.fillna(False).astype(bool)


def touches_from_above(candle, zone: Zone, margin: float) -> bool:
    """The candle dipped into (or very near) the zone and closed back above its
    bottom: the zone is acting as support."""
    return candle["low"] <= zone.high + margin and candle["close"] >= zone.low


def touches_from_below(candle, zone: Zone, margin: float) -> bool:
    """The candle poked into (or very near) the zone and closed back below its
    top: the zone is acting as resistance."""
    return candle["high"] >= zone.low - margin and candle["close"] <= zone.high


def zone_signals(low: float, high: float, close: float, flags: dict, zones: list[Zone],
                 margin: float) -> list[tuple[str, str, Zone]]:
    """Which patterns of ONE candle count, and at which zone.

    `flags` says which shapes the candle has ({'hammer': True, ...}).
    Returns (pattern, direction, zone) tuples. Used by both the live analysis
    and the backtest, so they can never disagree.
    """
    candle = {"low": low, "high": high, "close": close}
    found = []
    for zone in zones:
        from_above = touches_from_above(candle, zone, margin)
        from_below = touches_from_below(candle, zone, margin)
        for name in BULLISH:
            if flags[name] and from_above:
                found.append((name, "bullish", zone))
        for name in BEARISH:
            if flags[name] and from_below:
                found.append((name, "bearish", zone))
        if flags["doji"] and (from_above or from_below):
            found.append(("doji", "neutral", zone))
    return found


def signals_at_zones(df: pd.DataFrame, zones: list[Zone], atr_value: float,
                     cfg: dict) -> list[dict]:
    """Scan the last few closed candles for patterns that happen at a zone.

    Returns a list of dicts: time, pattern, direction ('bullish', 'bearish' or
    'neutral'), zone. Newest first.
    """
    p = cfg["patterns"]
    margin = p["near_level_atr_mult"] * atr_value
    flags = detect(df, cfg)
    found = []
    for pos in range(len(df) - 1, max(len(df) - 1 - p["lookback_bars"], -1), -1):
        candle = df.iloc[pos]
        hits = zone_signals(candle["low"], candle["high"], candle["close"],
                            flags.iloc[pos].to_dict(), zones, margin)
        found += [_signal(df, pos, name, direction, zone) for name, direction, zone in hits]
    return found


def _signal(df, pos, name, direction, zone) -> dict:
    return {
        "time": df.index[pos], "bars_ago": len(df) - 1 - pos, "pattern": name,
        "label": LABELS[name], "direction": direction, "zone": zone,
    }
