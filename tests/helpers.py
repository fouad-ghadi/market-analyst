"""Builders for hand-made candles, so each test knows the right answer in advance."""

from __future__ import annotations

import numpy as np
import pandas as pd


def candles(rows, start="2026-03-02 00:00", freq="h") -> pd.DataFrame:
    """rows = list of (open, high, low, close). Returns a frame shaped like real data."""
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)
    df.index.name = "time_utc"
    df.insert(0, "time_server", idx.tz_localize(None))
    df["tick_volume"], df["spread"], df["real_volume"] = 100, 0, 0
    return df


def from_highs_lows(highs, lows) -> pd.DataFrame:
    """Candles where only the wick tips matter (open = close = the middle)."""
    return candles([((h + l) / 2, h, l, (h + l) / 2) for h, l in zip(highs, lows)])


def path(points, bars_per_leg=10, wick=0.05) -> pd.DataFrame:
    """A zigzag: price walks in straight legs between `points`.
    Example: path([105, 110, 100]) rises from 105 to 110, then falls to 100.
    Every candle has a small wick on both sides."""
    closes = [points[0]]
    for a, b in zip(points, points[1:]):
        closes += list(np.linspace(a, b, bars_per_leg + 1)[1:])
    rows = []
    for prev, cur in zip(closes, closes[1:]):
        rows.append((prev, max(prev, cur) + wick, min(prev, cur) - wick, cur))
    return candles(rows)


def append(df: pd.DataFrame, rows) -> pd.DataFrame:
    """Add candles (open, high, low, close) after the end of df."""
    old = [tuple(r) for r in df[["open", "high", "low", "close"]].to_numpy()]
    return candles(old + list(rows))
