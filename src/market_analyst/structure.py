"""Market structure: uptrend, downtrend or range.

Rulebook:
    Uptrend   = Higher Highs + Higher Lows   (HH + HL)
    Downtrend = Lower Highs  + Lower Lows    (LH + LL)
    Otherwise = range
    The trend is "in question" when structure breaks: price makes a lower low
    in an uptrend, or a higher high in a downtrend.

We compare the last two CONFIRMED swing highs with each other, and the last
two confirmed swing lows with each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .swings import find_swings


@dataclass
class Structure:
    timeframe: str
    trend: str                  # 'up', 'down', 'range' or 'unknown' (not enough swings)
    highs_label: str            # 'HH', 'LH' or '-'
    lows_label: str             # 'HL', 'LL' or '-'
    last_swing_high: float | None
    last_swing_low: float | None
    broken: bool                # True = structure broke after the last swings
    note: str

    @property
    def summary(self) -> str:
        names = {"up": "UPTREND", "down": "DOWNTREND", "range": "RANGE", "unknown": "UNKNOWN"}
        text = f"{names[self.trend]} ({self.highs_label} + {self.lows_label})"
        return f"{text}, {self.note}" if self.note else text


def classify(df: pd.DataFrame, cfg: dict, timeframe: str,
             swings: pd.DataFrame | None = None) -> Structure:
    """Classify the structure visible on `df` (candles up to "now")."""
    if swings is None:
        swings = find_swings(df, cfg["swings"]["fractal_n"])
    highs = swings[swings["kind"] == "high"].tail(2)
    lows = swings[swings["kind"] == "low"].tail(2)
    if len(highs) < 2 or len(lows) < 2:
        return Structure(timeframe, "unknown", "-", "-", None, None, False,
                         "not enough swings yet")

    prev_high, last_high = highs["price"].tolist()
    prev_low, last_low = lows["price"].tolist()
    highs_label = "HH" if last_high > prev_high else "LH"
    lows_label = "HL" if last_low > prev_low else "LL"

    if highs_label == "HH" and lows_label == "HL":
        trend = "up"
    elif highs_label == "LH" and lows_label == "LL":
        trend = "down"
    else:
        trend = "range"

    # Structure break: look only at candles AFTER the swing that could be broken.
    broken, note = False, ""
    if trend == "up":
        after = df.iloc[int(lows["pos"].iloc[-1]) + 1:]
        if not after.empty and after["low"].min() < last_low:
            broken, note = True, "structure broken: price made a lower low, trend in question"
    elif trend == "down":
        after = df.iloc[int(highs["pos"].iloc[-1]) + 1:]
        if not after.empty and after["high"].max() > last_high:
            broken, note = True, "structure broken: price made a higher high, trend in question"

    return Structure(timeframe, trend, highs_label, lows_label, last_high, last_low, broken, note)


def trend_series(n_bars: int, swings: pd.DataFrame) -> list[str]:
    """The trend label at every candle, using only swings confirmed by then.

    trend_series(len(df), swings)[t] equals classify(df[:t+1]).trend. It exists
    so the backtest does not have to re-classify from scratch at every candle.
    """
    out = ["unknown"] * n_bars
    highs: list[float] = []
    lows: list[float] = []
    current, start = "unknown", 0
    for swing in swings.sort_values("confirmed_pos").itertuples():
        pos = int(swing.confirmed_pos)
        for i in range(start, min(pos, n_bars)):
            out[i] = current
        start = pos
        (highs if swing.kind == "high" else lows).append(swing.price)
        if len(highs) >= 2 and len(lows) >= 2:
            hh, hl = highs[-1] > highs[-2], lows[-1] > lows[-2]
            current = "up" if hh and hl else "down" if not hh and not hl else "range"
    for i in range(start, n_bars):
        out[i] = current
    return out
