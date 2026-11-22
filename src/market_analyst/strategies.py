"""The strategies the backtest can compare.

    level     Baseline (your rulebook): reversal candle at a validated level,
              enter at the close of that candle.
    retest    Second entry, rulebook meaning: same signal, but wait for price
              to leave the level and come back, and enter on the return.
    h2l2      Second entry, Al Brooks meaning (High 2 / Low 2): in a trend, the
              second attempt of a pullback to resume the trend.

Each strategy only sees candles up to the one that just closed, and answers
one question: "which order do I want working during the next candle?"
"""

from __future__ import annotations

from .analysis import setup_at_level, trend_alignment
from .backtest import History, Order, Strategy
from .planner import plan_at_level, plan_fixed_rr, sl_buffer


def level_info(h: History, i: int, direction: str, level, pattern: str, where: str,
               rr: float) -> dict:
    """Facts about a level setup at candle i (stored with the trade, used by filters)."""
    info = h.features(i, direction)
    info.update({
        "pattern": pattern,
        "location": where,
        "zone_touches": level.touches,
        "zone_score": level.score,
        "zone_d1": level.timeframe == "D1" or "D1" in level.confirmed_by,
        "zone_role_reversal": level.role_reversal,
        "planned_rr": round(rr, 2),
    })
    return info


class LevelConfirmation(Strategy):
    """Reversal candle at a validated level -> market entry at the candle close."""

    name = "level"

    def on_close(self, h: History, i: int, flat: bool) -> Order | None:
        if not flat:
            return None
        snap = h.snapshot(i)
        setup = setup_at_level(snap.loc, snap.hits)
        if setup is None:
            return None
        direction, level, target, pattern = setup
        plan, _ = plan_at_level(direction, h.close[i], level, target, h.atr[i], h.spread, self.cfg)
        if plan is None:
            return None            # no target or risk:reward below the minimum
        info = level_info(h, i, direction, level, pattern, snap.loc.where, plan.rr)
        if not self._ok(info):
            return None
        return Order("market", direction, sl=plan.sl, tp=plan.tp, info=info)


class RetestEntry(Strategy):
    """Same signal as the baseline, but the entry is a limit order at the edge
    of the level, placed only after price has moved away from it.

    Life of a setup:
      1. signal candle at the level            -> remember the level ("armed")
      2. price moves away by min_departure_atr -> the limit order becomes active
      3. price returns to the zone edge        -> filled
    Cancelled if: it takes more than expiry_bars, a candle closes through the
    level (the level failed), or price reaches the target first (move is over).
    """

    name = "retest"

    def reset(self) -> None:
        self.armed = None

    def on_close(self, h: History, i: int, flat: bool) -> Order | None:
        if not flat:
            self.armed = None
            return None
        p = self.cfg["strategies"]["retest"]
        a = self.armed
        if a is not None:
            buy = a["direction"] == "buy"
            failed = h.close[i] < a["level_low"] if buy else h.close[i] > a["level_high"]
            done = h.high[i] >= a["tp"] if buy else h.low[i] + h.spread <= a["tp"]
            if failed or done or i - a["signal_bar"] > p["expiry_bars"]:
                self.armed = a = None
            elif not a["departed"]:
                a["departed"] = (h.high[i] >= a["depart_at"] if buy
                                 else h.low[i] <= a["depart_at"])

        # A fresh signal (re)arms the setup.
        snap = h.snapshot(i)
        setup = setup_at_level(snap.loc, snap.hits)
        if setup is not None:
            direction, level, target, pattern = setup
            edge = level.high if direction == "buy" else level.low
            plan, _ = plan_at_level(direction, edge, level, target, h.atr[i], h.spread, self.cfg)
            if plan is not None:
                info = level_info(h, i, direction, level, pattern, snap.loc.where, plan.rr)
                if self._ok(info):
                    away = p["min_departure_atr"] * h.atr[i]
                    self.armed = a = {
                        "direction": direction, "signal_bar": i, "departed": False,
                        "level_low": level.low, "level_high": level.high,
                        "limit": plan.entry, "sl": plan.sl, "tp": plan.tp, "info": info,
                        "depart_at": edge + away if direction == "buy" else edge - away,
                    }
        if a is not None and a["departed"]:
            return Order("limit", a["direction"], sl=a["sl"], tp=a["tp"], price=a["limit"],
                         info=a["info"])
        return None


class H2L2(Strategy):
    """Al Brooks-style second entry.

    For buys (High 2), with the H4 trend up:
      phase 1  pullback starts: a candle with a lower high
      phase 2  "High 1": a candle goes above the previous candle's high.
               This first attempt to resume is NOT traded.
      phase 3  the attempt fails: another candle with a lower high
      entry    "High 2": buy stop just above the high of the last candle. It
               fills when the next candle goes above it (second attempt).
    Stop below the pullback low, target at target_rr x risk.
    Sells (Low 2) are the mirror image in an H4 downtrend.

    To use one piece of code for both directions, prices are flipped for
    sells (hi = -low, lo = -high), so "up" always means "with the trend".
    """

    name = "h2l2"

    def reset(self) -> None:
        self.side = None       # 'buy' or 'sell' machine currently running
        self.phase = 0
        self.peak = self.pb_low = 0.0
        self.start = 0

    def on_close(self, h: History, i: int, flat: bool) -> Order | None:
        p = self.cfg["strategies"]["h2l2"]
        trend = h.trend(p["trend_timeframe"], i)
        side = {"up": "buy", "down": "sell"}.get(trend)
        if not flat or side is None or i == 0:
            self.side, self.phase = side, 0
            return None
        if side != self.side:
            self.side, self.phase = side, 0

        s = 1 if side == "buy" else -1
        # "with-trend" view of the last two candles
        hi, prev_hi = (h.high[i], h.high[i - 1]) if s == 1 else (-h.low[i], -h.low[i - 1])
        lo = h.low[i] if s == 1 else -h.high[i]
        lower_high, higher_high = hi < prev_hi, hi > prev_hi

        if self.phase == 0:
            if lower_high:
                self.phase, self.start, self.peak, self.pb_low = 1, i, prev_hi, lo
            return None

        self.pb_low = min(self.pb_low, lo)
        if i - self.start > p["max_pullback_bars"]:
            self.phase = 0
            return None
        if self.phase == 1:
            if higher_high:
                self.phase = 2                    # High 1: first attempt, not traded
            return None
        if self.phase == 2:
            if hi > self.peak:
                self.phase = 0                    # the first attempt worked: no second entry
            elif lower_high:
                self.phase = 3
            if self.phase != 3:
                return None

        # phase 3: second leg of the pullback. Order above the last candle.
        if hi > self.peak:
            self.phase = 0
            return None
        if self.peak - self.pb_low < p["min_pullback_atr"] * h.atr[i]:
            return None                           # pullback too shallow to matter
        buffer = sl_buffer(h.spread, h.atr[i], self.cfg)
        if side == "buy":
            entry = h.high[i] + h.tick + h.spread          # buy stop (ask)
            sl = self.pb_low - buffer
        else:
            entry = h.low[i] - h.tick                      # sell stop (bid)
            sl = -self.pb_low + buffer
        plan = plan_fixed_rr(side, entry, sl, p["target_rr"])
        if plan.risk <= 0:
            return None
        info = h.features(i, side)
        # Did the pullback reach the 20 EMA? (self.pb_low is the deepest point,
        # flipped for sells, so s * pb_low is the real price.)
        ema_gap = s * (h.ema_fast[i] - s * self.pb_low)
        info.update({
            "pattern": "H2" if side == "buy" else "L2",
            "pullback_bars": i - self.start,
            "pullback_atr": round((self.peak - self.pb_low) / h.atr[i], 2),
            "touched_ema": bool(ema_gap >= 0),     # the pullback reached the 20 EMA
            "planned_rr": p["target_rr"],
        })
        if not self._ok(info):
            return None
        return Order("stop", side, sl=plan.sl, tp=plan.tp, price=plan.entry, info=info)


STRATEGIES = {cls.name: cls for cls in (LevelConfirmation, RetestEntry, H2L2)}

__all__ = ["STRATEGIES", "LevelConfirmation", "RetestEntry", "H2L2", "trend_alignment"]
