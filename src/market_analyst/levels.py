"""Support / Resistance zones, built from swing wick tips.

Rulebook -> code:
  * "Draw levels on wick tips"            -> zones are built from swing prices (highs/lows)
  * "Levels are zones, not exact lines"   -> each zone has a low and a high edge
  * "2 touches with a clear reaction"     -> a touch = a confirmed swing inside the zone
                                             (a swing IS a reaction: price turned there)
  * "1 touch = level to watch"            -> status 'watch'
  * "Recent touches matter more"          -> recency weight in the score
  * "Higher timeframe levels are stronger"-> timeframe weight in the score
  * "Broken resistance becomes support"   -> a zone's role depends on which side of it
                                             price last CLOSED (a wick through is not a break)

How zones are found: sort all swing prices, then walk up the list. A price
joins the current group while it is within `tolerance` (= k x ATR) of the
group's lowest price. Otherwise it starts a new group. Each group is a zone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .indicators import atr, last_atr
from .swings import find_swings


@dataclass
class Zone:
    low: float                 # bottom edge
    high: float                # top edge
    timeframe: str             # timeframe it was found on
    touches: int               # confirmed swings inside the zone
    touch_highs: int           # ... of which swing highs (zone acted as resistance)
    touch_lows: int            # ... of which swing lows  (zone acted as support)
    first_touch: pd.Timestamp
    last_touch: pd.Timestamp
    bars_since_last_touch: int
    score: float
    status: str                # 'validated' (>= 2 touches) or 'watch' (1 touch)
    role: str                  # 'support', 'resistance' or 'inside' (price is in the zone)
    confirmed_by: list[str] = field(default_factory=list)  # other timeframes with a zone here

    @property
    def center(self) -> float:
        return (self.low + self.high) / 2

    @property
    def role_reversal(self) -> bool:
        """True if the zone has acted as both support and resistance."""
        return self.touch_highs > 0 and self.touch_lows > 0

    @property
    def untested_flip(self) -> bool:
        """True if the zone changed role and has not been touched from the new
        side yet: e.g. only swing highs (it was resistance) but price is now
        above it. The rulebook's "break and retest": the retest has not happened."""
        return (self.role == "support" and self.touch_lows == 0) or (
            self.role == "resistance" and self.touch_highs == 0)

    def overlaps(self, other: "Zone") -> bool:
        return self.low <= other.high and other.low <= self.high

    def distance_to(self, price: float) -> float:
        """0 if price is inside the zone, else the distance to the nearest edge."""
        if price > self.high:
            return price - self.high
        if price < self.low:
            return self.low - price
        return 0.0


def role_for(low: float, high: float, last_close: float) -> str:
    """Role reversal rule: the zone is support if price closed above it,
    resistance if price closed below it."""
    if last_close > high:
        return "support"
    if last_close < low:
        return "resistance"
    return "inside"


def reaction_positions(df: pd.DataFrame, swings: pd.DataFrame, cfg: dict) -> list[int]:
    """For each swing: the candle position from which it counts as a touch
    "with a clear reaction", or -1 if it does not (yet).

    For a swing high: how far did price FALL from the wick tip during the next
    `reaction_bars` candles? For a swing low: how far did it RISE? The swing
    counts from the first candle where that move reaches
    `min_reaction_atr_mult` x ATR (ATR measured at the swing candle), and never
    before the swing itself is confirmed.

    No look-ahead: at candle t, a swing is a touch only if its position here is
    <= t. The backtest uses exactly this to replay history.
    """
    lcfg = cfg["levels"]
    mult, window = lcfg.get("min_reaction_atr_mult", 0), lcfg.get("reaction_bars", 0)
    if swings.empty:
        return []
    if mult <= 0 or window <= 0:
        return [int(p) for p in swings["confirmed_pos"]]
    atr_values = atr(df, cfg["atr"]["period"]).bfill().to_numpy()
    highs, lows = df["high"].to_numpy(), df["low"].to_numpy()
    out = []
    for swing in swings.itertuples():
        needed = mult * atr_values[swing.pos]
        found = -1
        for j in range(swing.pos + 1, min(swing.pos + 1 + window, len(df))):
            move = swing.price - lows[j] if swing.kind == "high" else highs[j] - swing.price
            if move >= needed:
                found = max(j, int(swing.confirmed_pos))
                break
        out.append(found)
    return out


def clear_reactions(df: pd.DataFrame, swings: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Keep only the swings where price clearly reacted (rulebook: a touch needs
    "a clear reaction"), judged on the candles present in `df`."""
    if swings.empty:
        return swings
    return swings[[p >= 0 for p in reaction_positions(df, swings, cfg)]]


def zones_from_touches(touches: list[dict], tolerance: float, last_close: float,
                       last_pos: int, cfg: dict, timeframe: str) -> list[Zone]:
    """Group touches into zones. `touches` are swing records (price, kind, pos,
    time) that already passed the lookback and clear-reaction filters."""
    lcfg = cfg["levels"]
    half_life = lcfg["recency_half_life_bars"]
    tf_weight = lcfg["timeframe_weight"].get(timeframe, 1.0)

    # --- group swing prices that are within `tolerance` of each other ---
    groups: list[list[dict]] = []
    for swing in sorted(touches, key=lambda t: t["price"]):
        if groups and swing["price"] - groups[-1][0]["price"] <= tolerance:
            groups[-1].append(swing)
        else:
            groups.append([swing])

    zones = []
    for members in groups:
        prices = [m["price"] for m in members]
        center = sum(prices) / len(prices)
        # A zone is never thinner than `tolerance` (price can overshoot a little).
        low = min(min(prices), center - tolerance / 2)
        high = max(max(prices), center + tolerance / 2)
        count = len(members)
        ages = [last_pos - m["pos"] for m in members]
        # Each touch adds between 0 and 1: 1 if it just happened, 0.5 after
        # `half_life` bars, 0.25 after twice that, and so on.
        recency = sum(0.5 ** (age / half_life) for age in ages)
        zones.append(Zone(
            low=low, high=high, timeframe=timeframe, touches=count,
            touch_highs=sum(m["kind"] == "high" for m in members),
            touch_lows=sum(m["kind"] == "low" for m in members),
            first_touch=min(m["time"] for m in members),
            last_touch=max(m["time"] for m in members),
            bars_since_last_touch=min(ages),
            score=round(count + recency + tf_weight, 2),
            status="validated" if count >= lcfg["min_touches_validated"] else "watch",
            role=role_for(low, high, last_close),
        ))
    return zones


def build_zones(df: pd.DataFrame, cfg: dict, timeframe: str,
                swings: pd.DataFrame | None = None) -> list[Zone]:
    """Find the zones visible on `df` (candles up to "now"), sorted by price."""
    lcfg = cfg["levels"]
    if swings is None:
        swings = find_swings(df, cfg["swings"]["fractal_n"])

    # Only recent swings build zones (very old ones are no longer relevant),
    # and only the ones with a clear reaction.
    last_pos = len(df) - 1
    swings = swings[swings["pos"] >= last_pos - lcfg["lookback_bars"]]
    swings = clear_reactions(df, swings, cfg)
    if swings.empty:
        return []
    tolerance = lcfg["tolerance_atr_mult"] * last_atr(df, cfg["atr"]["period"])
    return zones_from_touches(swings.to_dict("records"), tolerance,
                              float(df["close"].iloc[-1]), last_pos, cfg, timeframe)


def combine_timeframes(zones_by_tf: dict[str, list[Zone]], cfg: dict) -> list[Zone]:
    """Put zones from several timeframes in one list, sorted by price.

    When a lower-timeframe zone overlaps a higher-timeframe zone, they are the
    same area seen twice: we keep the higher-timeframe zone, note that the
    other timeframe confirms it, and raise its score.
    """
    weights = cfg["levels"]["timeframe_weight"]
    ordered = sorted(zones_by_tf, key=lambda tf: -weights.get(tf, 1.0))  # strongest first
    combined: list[Zone] = []
    for tf in ordered:
        for zone in zones_by_tf[tf]:
            match = next((z for z in combined if z.overlaps(zone)), None)
            if match is None:
                combined.append(zone)
                continue
            if tf not in match.confirmed_by and tf != match.timeframe:
                match.confirmed_by.append(tf)
                match.score = round(match.score + weights.get(tf, 1.0), 2)
            # The same turning points show up on both timeframes, so we do not
            # add the counts (that would count one touch twice). We keep the
            # larger count: the lower timeframe often sees touches that are
            # too small to be a swing on the higher one.
            if zone.touches > match.touches:
                match.touches = zone.touches
                match.touch_highs, match.touch_lows = zone.touch_highs, zone.touch_lows
            match.first_touch = min(match.first_touch, zone.first_touch)
            match.last_touch = max(match.last_touch, zone.last_touch)
            if zone.status == "validated":
                match.status = "validated"
    return sorted(combined, key=lambda z: z.center)


def nearest_zones(zones: list[Zone], price: float, validated_only: bool = True
                  ) -> tuple[Zone | None, Zone | None, Zone | None]:
    """Return (support below price, zone price is inside, resistance above price)."""
    pool = [z for z in zones if z.status == "validated"] if validated_only else zones
    below = [z for z in pool if z.high < price]
    above = [z for z in pool if z.low > price]
    inside = [z for z in pool if z.low <= price <= z.high]
    return (
        max(below, key=lambda z: z.high) if below else None,
        max(inside, key=lambda z: z.score) if inside else None,
        min(above, key=lambda z: z.low) if above else None,
    )
