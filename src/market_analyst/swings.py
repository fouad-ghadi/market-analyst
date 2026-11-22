"""Swing highs and swing lows (fractals).

A swing high is a candle whose HIGH (wick tip) is above the highs of the N
candles before it and not below the highs of the N candles after it.
A swing low is the mirror image with LOWS.

No look-ahead: we cannot know a candle is a swing high until N more candles
have closed. So each swing has two positions:
    pos            the candle where the swing happened
    confirmed_pos  = pos + N, the candle whose close CONFIRMS it
Anything that makes a decision at candle t may only use swings with
confirmed_pos <= t. The simplest way to guarantee that: give these functions
only the candles up to t. The last N candles can never be swings yet.
"""

from __future__ import annotations

import pandas as pd

COLUMNS = ["time", "kind", "price", "pos", "confirmed_pos", "confirmed_time"]


def find_swings(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Return one row per confirmed swing, oldest first.

    Columns: time (of the swing candle), kind ('high' or 'low'), price (the
    wick tip), pos, confirmed_pos, confirmed_time.
    """
    high, low = df["high"], df["low"]

    # Highest high of the n candles BEFORE each candle, and of the n candles AFTER it.
    left_max = high.shift(1).rolling(n).max()
    right_max = high[::-1].shift(1).rolling(n).max()[::-1]
    left_min = low.shift(1).rolling(n).min()
    right_min = low[::-1].shift(1).rolling(n).min()[::-1]

    # Strictly above the left side, not below the right side. With two equal
    # highs next to each other, only the first one counts (no duplicates).
    # Near the edges the rolling windows are NaN, comparisons are False, so the
    # first and last n candles are never swings.
    is_high = (high > left_max) & (high >= right_max)
    is_low = (low < left_min) & (low <= right_min)

    rows = []
    for pos in range(len(df)):
        for flag, kind, price in ((is_high.iloc[pos], "high", high.iloc[pos]),
                                  (is_low.iloc[pos], "low", low.iloc[pos])):
            if flag:
                rows.append({
                    "time": df.index[pos], "kind": kind, "price": float(price), "pos": pos,
                    "confirmed_pos": pos + n, "confirmed_time": df.index[pos + n],
                })
    return pd.DataFrame(rows, columns=COLUMNS)
