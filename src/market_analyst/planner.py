"""Trade plan: entry, stop loss, take profit, risk:reward.

Rulebook -> code:
    SL  a little BEYOND the level       -> zone edge -/+ buffer
    TP  just BEFORE the next level      -> opposite zone edge -/+ buffer
    buffer = max(2 x spread, 0.1 x ATR)
    R:R below 1:2                       -> no plan (the caller says NO TRADE)

Prices are the ones you would type into MT5:
  * a BUY opens at the ask (chart price + spread) and closes at the bid (chart price)
  * a SELL opens at the bid (chart price) and closes at the ask (chart price + spread)
The spread is a cost in both cases, and it is counted here.

This module only does arithmetic. It never talks to the broker.
"""

from __future__ import annotations

from dataclasses import dataclass

from .levels import Zone


@dataclass
class TradePlan:
    direction: str      # 'buy' or 'sell'
    entry: float        # expected fill price
    sl: float
    tp: float

    @property
    def risk(self) -> float:
        """Price distance lost if the stop is hit."""
        return self.entry - self.sl if self.direction == "buy" else self.sl - self.entry

    @property
    def reward(self) -> float:
        """Price distance won if the target is hit."""
        return self.tp - self.entry if self.direction == "buy" else self.entry - self.tp

    @property
    def rr(self) -> float:
        return self.reward / self.risk if self.risk > 0 else 0.0


def sl_buffer(spread: float, atr_value: float, cfg: dict) -> float:
    s = cfg["stops"]
    return max(s["sl_buffer_spread_mult"] * spread, s["sl_buffer_atr_mult"] * atr_value)


def fill_price(direction: str, chart_price: float, spread: float) -> float:
    """What you actually pay: a buy fills at the ask."""
    return chart_price + spread if direction == "buy" else chart_price


def plan_at_level(direction: str, chart_price: float, level: Zone, target: Zone | None,
                  atr_value: float, spread: float, cfg: dict) -> tuple[TradePlan | None, str]:
    """Plan a trade taken at `level`, aiming at `target` (the next opposite zone).

    Returns (plan, "") when the plan passes the rules, or (None, reason).
    """
    scfg = cfg["stops"]
    min_rr = scfg["min_risk_reward"]
    fixed = scfg.get("tp_mode", "next_level") == "fixed_rr"
    if target is None and not fixed:
        return None, ("There is no validated opposite level to place the take profit before, "
                      f"so the 1:{min_rr:g} minimum cannot be checked.")
    buffer = sl_buffer(spread, atr_value, cfg)
    entry = fill_price(direction, chart_price, spread)
    # Optional: never let the stop sit closer than k x ATR to the entry. A stop
    # inside the normal noise of one candle gets hit for no reason.
    room = scfg.get("min_risk_atr_mult", 0.0) * atr_value
    if direction == "buy":
        sl = min(level.low - buffer, entry - room)
    else:
        sl = max(level.high + buffer, entry + room)
    if fixed:
        plan = plan_fixed_rr(direction, entry, sl, scfg["fixed_rr"])
    elif direction == "buy":
        plan = TradePlan("buy", entry, sl=sl, tp=target.low - buffer)
    else:
        plan = TradePlan("sell", entry, sl=sl, tp=target.high + buffer)
    if plan.risk <= 0 or plan.reward <= 0:
        return None, "Price is already beyond the level or the target: no valid plan."
    if plan.rr < min_rr:
        return None, (f"Risk:reward is 1:{plan.rr:.1f}, below the 1:{min_rr:g} minimum "
                      "(the next opposite level is too close).")
    return plan, ""


def plan_fixed_rr(direction: str, entry: float, sl: float, rr: float) -> TradePlan:
    """Plan with a target at a fixed multiple of the risk (used by strategies
    that do not aim at a level)."""
    risk = entry - sl if direction == "buy" else sl - entry
    tp = entry + rr * risk if direction == "buy" else entry - rr * risk
    return TradePlan(direction, entry, sl, tp)
