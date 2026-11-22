"""Backtest: replay history candle by candle and see what a strategy would have done.

How it works (and why it cannot cheat):
  * Decisions are taken at the CLOSE of each H1 candle, using only candles that
    were closed at that moment. Higher-timeframe candles (H4, D1) are only used
    once they have fully closed too.
  * An order decided at the close of candle i can fill on candle i+1 at the
    earliest.
  * The spread is paid on every trade (a buy fills at chart price + spread;
    a sell is closed at chart price + spread).
  * If the stop and the target are both inside the same candle, we assume the
    STOP was hit first. That is the pessimistic choice.
  * On the candle where a limit/stop order fills, only the stop can be hit
    (we do not know the path inside the candle, so no "instant win").
  * After a weekend or holiday, price can open beyond the stop: the trade then
    exits at that worse price ("gap").
  * One trade at a time.

Results are in R: +2R means the trade won twice what it risked, -1R is a
full stop loss. Expectancy = average R per trade. Above 0 = made money in
the test, below 0 = lost money. A backtest describes the past; it does not
promise the future.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import patterns
from .analysis import Location, locate, trend_alignment
from .indicators import atr, ema
from .levels import (
    Zone,
    combine_timeframes,
    reaction_positions,
    role_for,
    zones_from_touches,
)
from .structure import trend_series
from .swings import find_swings

TF_HOURS = {"M5": 1 / 12, "M15": 0.25, "M30": 0.5, "H1": 1, "H4": 4, "D1": 24}


# =============================================================================
# 1. History: "what was known at candle i?"
# =============================================================================

class _LevelFrame:
    """One level timeframe (H4 or D1), prepared once so zones can be rebuilt
    quickly as of any candle."""

    def __init__(self, df: pd.DataFrame, cfg: dict, tf: str):
        self.tf, self.cfg = tf, cfg
        self.close = df["close"].to_numpy()
        self.atr = atr(df, cfg["atr"]["period"]).bfill().to_numpy()
        swings = find_swings(df, cfg["swings"]["fractal_n"])
        self.trend = trend_series(len(df), swings)
        # From which candle does each swing count as a touch? (-1 = never)
        self.touches = [dict(rec, usable_from=pos) for rec, pos in
                        zip(swings.to_dict("records"), reaction_positions(df, swings, cfg))
                        if pos >= 0]

    def zones_at(self, i: int) -> list[Zone]:
        lcfg = self.cfg["levels"]
        known = [t for t in self.touches
                 if t["usable_from"] <= i and t["pos"] >= i - lcfg["lookback_bars"]]
        tolerance = lcfg["tolerance_atr_mult"] * self.atr[i]
        return zones_from_touches(known, tolerance, self.close[i], i, self.cfg, self.tf)


@dataclass
class Snapshot:
    """Everything the level strategy needs at one candle."""
    zones: list[Zone]
    loc: Location
    hits: list[tuple[str, str, Zone]]     # reversal candles of THIS candle, at zones
    margin: float


class History:
    """Candles of one symbol plus fast "as of candle i" lookups."""

    def __init__(self, symbol: str, cfg: dict, frames: dict[str, pd.DataFrame]):
        self.symbol, self.cfg = symbol, cfg
        bcfg = cfg["backtest"]
        self.entry_tf = bcfg["entry_timeframe"]
        df = frames[self.entry_tf]
        self.n = len(df)
        self.time = df.index
        self.open, self.high = df["open"].to_numpy(), df["high"].to_numpy()
        self.low, self.close = df["low"].to_numpy(), df["close"].to_numpy()
        # True where the market was closed before this candle (weekend, holiday):
        # only there can price open beyond a stop. A single missing hour in the
        # data feed is not a real gap.
        hours_since_prev = np.diff(self.time.asi8, prepend=self.time.asi8[0]) / 3.6e12
        self.after_closure = hours_since_prev > cfg["backtest"]["gap_after_hours"]
        self.spread = float(cfg["symbols"][symbol].get("typical_spread", 0.0))
        self.tick = cfg["symbols"][symbol]["pip_size"] / 10

        self.atr = atr(df, cfg["atr"]["period"]).bfill().to_numpy()
        # "Is today wilder than usual?": ATR against its own median of the last 20 days.
        median = pd.Series(self.atr).rolling(480, min_periods=100).median()
        self.atr_ratio = (pd.Series(self.atr) / median).fillna(1.0).to_numpy()
        self.ema_fast = ema(df["close"], cfg["strategies"]["h2l2"]["ema_period"]).to_numpy()
        self.ema50 = ema(df["close"], 50).to_numpy()
        self.ema200 = ema(df["close"], 200).to_numpy()
        self.flags = {name: col.to_numpy() for name, col in patterns.detect(df, cfg).items()}
        self.trend_entry = trend_series(self.n, find_swings(df, cfg["swings"]["fractal_n"]))

        # Higher timeframes: for each entry candle, the index of the last
        # higher-timeframe candle that had CLOSED by the entry candle's close.
        closes_at = self.time + pd.Timedelta(hours=TF_HOURS[self.entry_tf])
        self.level_tfs = [tf for tf in cfg["strategy"]["level_timeframes"] if tf in frames]
        self.levels: dict[str, _LevelFrame] = {}
        self.idx: dict[str, np.ndarray] = {}
        for tf in self.level_tfs:
            self.levels[tf] = _LevelFrame(frames[tf], cfg, tf)
            tf_closes_at = frames[tf].index + pd.Timedelta(hours=TF_HOURS[tf])
            self.idx[tf] = np.searchsorted(tf_closes_at.asi8, closes_at.asi8, side="right") - 1

        need = bcfg["min_history_bars"]
        ready = np.ones(self.n, dtype=bool)
        for tf in self.level_tfs:
            ready &= self.idx[tf] >= need.get(tf, 0)
        self.first_bar = int(np.argmax(ready)) if ready.any() else self.n
        self._zone_key, self._zones = None, []

    # --- lookups ------------------------------------------------------------
    def trend(self, tf: str, i: int) -> str:
        if tf == self.entry_tf:
            return self.trend_entry[i]
        j = self.idx[tf][i]
        return self.levels[tf].trend[j] if j >= 0 else "unknown"

    def zones(self, i: int) -> list[Zone]:
        """Combined zones as of candle i. Rebuilt only when a new H4/D1 candle closed."""
        key = tuple(int(self.idx[tf][i]) for tf in self.level_tfs)
        if key != self._zone_key:
            per_tf = {tf: self.levels[tf].zones_at(j)
                      for tf, j in zip(self.level_tfs, key) if j >= 0}
            self._zones, self._zone_key = combine_timeframes(per_tf, self.cfg), key
        price = self.close[i]
        for z in self._zones:
            z.role = role_for(z.low, z.high, price)
        return self._zones

    def snapshot(self, i: int) -> Snapshot:
        cfg = self.cfg
        zones = self.zones(i)
        margin = cfg["patterns"]["near_level_atr_mult"] * self.atr[i]
        loc = locate(self.close[i], self.low[i], self.high[i], zones, margin, cfg)
        pool = ([z for z in zones if z.status == "validated"]
                if cfg["strategy"]["require_validated_level"] else zones)
        flags = {name: bool(arr[i]) for name, arr in self.flags.items()}
        hits = (patterns.zone_signals(self.low[i], self.high[i], self.close[i], flags, pool,
                                      margin) if any(flags.values()) else [])
        return Snapshot(zones, loc, hits, margin)

    def features(self, i: int, direction: str) -> dict:
        """Facts known at candle i, stored with each trade so we can study
        afterwards which conditions helped. All of them are causal."""
        close_time = self.time[i] + pd.Timedelta(hours=TF_HOURS[self.entry_tf])
        hour = close_time.hour
        session = ("asia" if hour < 7 else "london" if hour < 13
                   else "overlap" if hour < 17 else "late")
        sign = 1 if direction == "buy" else -1
        out = {
            "hour": hour, "weekday": close_time.day_of_week, "session": session,
            "atr_ratio": round(float(self.atr_ratio[i]), 2),
            "ema200_side": bool(sign * (self.close[i] - self.ema200[i]) > 0),
            "ema50_slope": bool(sign * (self.ema50[i] - self.ema50[max(i - 10, 0)]) > 0),
            "trend_H1": trend_alignment(self.trend_entry[i], direction),
        }
        for tf in self.level_tfs:
            out[f"trend_{tf}"] = trend_alignment(self.trend(tf, i), direction)
            out[f"regime_{tf}"] = self.trend(tf, i)      # up / down / range / unknown
        return out


# =============================================================================
# 2. Orders and trades
# =============================================================================

@dataclass
class Order:
    kind: str                 # 'market', 'limit' or 'stop'
    direction: str            # 'buy' or 'sell'
    sl: float
    tp: float
    price: float | None = None    # trigger price for limit/stop (buy: ask, sell: bid)
    info: dict = field(default_factory=dict)


@dataclass
class Trade:
    symbol: str
    strategy: str
    direction: str
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry: float
    sl: float
    tp: float
    exit: float
    outcome: str              # 'tp', 'sl', 'gap' or 'timeout'
    r: float                  # result in units of risk
    bars_held: int
    info: dict = field(default_factory=dict)


class Strategy:
    """Base class. A strategy looks at history up to candle i and may return
    the order it wants working during candle i+1 (or None)."""

    name = "strategy"

    def __init__(self, cfg: dict, accept=None):
        self.cfg = cfg
        # Optional filter: a function(info dict) -> bool. Used by the research
        # study to re-test a strategy with extra conditions.
        self.accept = accept

    def reset(self) -> None:
        pass

    def on_close(self, h: History, i: int, flat: bool) -> Order | None:
        raise NotImplementedError

    def on_fill(self) -> None:
        """Called when the strategy's order fills, so it can clear its setup."""
        self.reset()

    def _ok(self, info: dict) -> bool:
        return self.accept is None or bool(self.accept(info))


def try_fill(order: Order, o: float, h: float, l: float, spread: float) -> float | None:
    """Fill price if the order fills on a candle with open o, high h, low l."""
    buy = order.direction == "buy"
    if order.kind == "market":
        return o + spread if buy else o
    p = order.price
    if order.kind == "limit":        # better price or nothing
        if buy:
            return min(p, o + spread) if l + spread <= p else None
        return max(p, o) if h >= p else None
    if buy:                          # stop: breakout entry
        return max(p, o + spread) if h + spread >= p else None
    return min(p, o) if l <= p else None


def check_exit(direction: str, sl: float, tp: float, o: float, h: float, l: float,
               spread: float, first_bar: bool, market_entry: bool, can_gap: bool = True
               ) -> tuple[str, float] | None:
    """Did the trade end on this candle? Returns (outcome, exit price) or None.
    A buy closes at the chart (bid) price; a sell closes at chart + spread (ask).
    `can_gap`: the market was closed before this candle, so it may open beyond the stop."""
    if direction == "buy":
        if can_gap and not first_bar and o <= sl:
            return "gap", o
        if l <= sl:
            return "sl", sl
        if h >= tp and (not first_bar or market_entry):
            return "tp", tp
    else:
        if can_gap and not first_bar and o + spread >= sl:
            return "gap", o + spread
        if h + spread >= sl:
            return "sl", sl
        if l + spread <= tp and (not first_bar or market_entry):
            return "tp", tp
    return None


# =============================================================================
# 3. The replay loop
# =============================================================================

def run(h: History, strategy: Strategy, start: int | None = None, end: int | None = None
        ) -> list[Trade]:
    """Replay candles [start, end) and return the finished trades."""
    strategy.reset()
    max_hold = h.cfg["backtest"]["max_hold_bars"]
    start = h.first_bar if start is None else max(start, h.first_bar)
    end = h.n if end is None else min(end, h.n)
    trades: list[Trade] = []
    pending: Order | None = None
    pending_from = 0
    pos = None        # the open trade: dict(order, entry, entry_bar, signal_bar)

    for j in range(start, end):
        o, hi, lo, c = h.open[j], h.high[j], h.low[j], h.close[j]

        # 1. an order decided on an earlier close may fill on this candle
        if pos is None and pending is not None:
            fill = try_fill(pending, o, hi, lo, h.spread)
            if fill is not None:
                buy = pending.direction == "buy"
                valid = pending.sl < fill < pending.tp if buy else pending.tp < fill < pending.sl
                if valid:
                    pos = {"order": pending, "entry": fill, "entry_bar": j,
                           "signal_bar": pending_from, "sl": pending.sl,
                           "risk": abs(fill - pending.sl), "banked": 0.0, "size": 1.0,
                           "managed": False}
                    strategy.on_fill()
                pending = None

        # 2. manage the open trade
        if pos is not None:
            order = pos["order"]
            first = j == pos["entry_bar"]
            buy = order.direction == "buy"
            result = check_exit(order.direction, pos["sl"], order.tp, o, hi, lo, h.spread,
                                first, order.kind == "market", bool(h.after_closure[j]))
            if result is None and j - pos["entry_bar"] >= max_hold:
                result = ("timeout", c if buy else c + h.spread)
            if result is not None:
                outcome, exit_price = result
                if outcome == "sl" and pos["managed"]:
                    outcome = "breakeven"
                gain = exit_price - pos["entry"] if buy else pos["entry"] - exit_price
                trades.append(Trade(
                    symbol=h.symbol, strategy=strategy.name, direction=order.direction,
                    signal_time=h.time[pos["signal_bar"]], entry_time=h.time[pos["entry_bar"]],
                    exit_time=h.time[j], entry=pos["entry"], sl=order.sl, tp=order.tp,
                    exit=exit_price, outcome=outcome,
                    r=pos["banked"] + pos["size"] * gain / pos["risk"],
                    bars_held=j - pos["entry_bar"], info=order.info))
                pos = None
            elif not pos["managed"] and (not first or order.kind == "market"):
                # Trade management, applied AFTER the exit check so that a candle
                # which reaches +1R and also the stop still counts as a full loss.
                mcfg = strategy.cfg["stops"]
                trigger = mcfg.get("manage_at_r", 0)
                best = (hi - pos["entry"] if buy else pos["entry"] - (lo + h.spread))
                if trigger > 0 and best >= trigger * pos["risk"]:
                    pos["managed"] = True
                    part = mcfg.get("partial_fraction", 0.0)
                    pos["banked"] = part * trigger          # part closed at +trigger R
                    pos["size"] = 1.0 - part
                    if mcfg.get("breakeven_after_manage", True):
                        pos["sl"] = pos["entry"]            # rest cannot lose any more

        # 3. the candle has closed: the strategy decides what to do next
        order = strategy.on_close(h, j, flat=pos is None)
        if pos is None:
            pending, pending_from = order, j
    return trades


def latest_order(h: History, strategy: Strategy, warmup: int = 400) -> Order | None:
    """The order a strategy wants working for the NEXT candle, as of the last
    closed candle. The last `warmup` candles are replayed first so strategies
    that follow a setup over several candles (retest, h2l2) are up to date.
    Uses the same strategy code as the backtest."""
    strategy.reset()
    order = None
    for i in range(max(h.first_bar, h.n - warmup, 1), h.n):
        if order is not None and try_fill(order, h.open[i], h.high[i], h.low[i],
                                          h.spread) is not None:
            strategy.on_fill()          # that order would have filled: setup is used up
        order = strategy.on_close(h, i, flat=True)
    return order


# =============================================================================
# 4. Statistics
# =============================================================================

def summarize(trades: list[Trade], seed: int = 1) -> dict:
    """Numbers that describe a list of trades."""
    n = len(trades)
    if n == 0:
        return {"trades": 0, "win_rate": None, "expectancy_r": None, "total_r": 0.0,
                "profit_factor": None, "max_drawdown_r": 0.0, "avg_bars": None,
                "ci_low": None, "ci_high": None}
    r = np.array([t.r for t in trades])
    wins, losses = r[r > 0], r[r <= 0]
    equity = np.cumsum(r)
    drawdown = float((np.maximum.accumulate(np.maximum(equity, 0)) - equity).max())
    # Bootstrap: redraw the trades many times to see how much the average
    # could move by luck alone. If the 90% range includes 0, the result is not
    # distinguishable from "no edge".
    rng = np.random.default_rng(seed)
    means = rng.choice(r, size=(4000, n), replace=True).mean(axis=1)
    return {
        "trades": n,
        "win_rate": float(len(wins) / n),
        "expectancy_r": float(r.mean()),
        "total_r": float(r.sum()),
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else None,
        "max_drawdown_r": drawdown,
        "avg_bars": float(np.mean([t.bars_held for t in trades])),
        "ci_low": float(np.percentile(means, 5)),
        "ci_high": float(np.percentile(means, 95)),
    }


def trades_frame(trades: list[Trade]) -> pd.DataFrame:
    """Trades as a table (one row per trade, features as columns)."""
    rows = []
    for t in trades:
        row = {k: v for k, v in t.__dict__.items() if k != "info"}
        row.update(t.info)
        rows.append(row)
    return pd.DataFrame(rows)


def format_summary(label: str, s: dict) -> str:
    if s["trades"] == 0:
        return f"{label:34s}   0 trades"
    pf = f"{s['profit_factor']:.2f}" if s["profit_factor"] is not None else "  - "
    return (f"{label:34s} {s['trades']:4d} trades  win {s['win_rate'] * 100:4.0f}%  "
            f"avg {s['expectancy_r']:+.2f}R  (90% range {s['ci_low']:+.2f} to "
            f"{s['ci_high']:+.2f})  total {s['total_r']:+6.1f}R  PF {pf}  "
            f"max DD {s['max_drawdown_r']:.1f}R")
