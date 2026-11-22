"""Small indicator helpers. Everything here is causal: the value at bar t
only uses bars up to t."""

from __future__ import annotations

import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    """How far price really travelled during each candle, including any gap
    from the previous close. It is the largest of:
        high - low,  |high - previous close|,  |low - previous close|
    """
    prev_close = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range: the average candle size over the last `period` candles.
    We use it as a ruler, so "close to a level" means the same thing on EURUSD
    (moves of 0.0010) and on GOLD (moves of 10.00)."""
    return true_range(df).rolling(period).mean()


def last_atr(df: pd.DataFrame, period: int = 14) -> float:
    """ATR at the most recent candle (NaN-safe: falls back to the mean candle range)."""
    value = atr(df, period).iloc[-1]
    if pd.isna(value):
        value = (df["high"] - df["low"]).mean()
    return float(value)


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average: an average that gives more weight to recent
    candles. Causal: the value at candle t only uses candles up to t."""
    return series.ewm(span=period, adjust=False).mean()
