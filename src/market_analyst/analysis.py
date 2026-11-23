"""Put the pieces together: structure + zones + location + candles -> verdict.

This follows the strategy steps of the rulebook, in order:
    1. mark validated levels on H4 and D1
    2. where is price compared to those levels?   (middle of range = NO TRADE)
    3. is there a reversal candle AT the level, on the last closed H1 candle?
    4. plan entry / SL / TP: is the risk:reward at least 1:2?
    5. is the trade with the H4 trend?

The result is one of:
    NO TRADE         a rule fails (middle of range, R:R too small, no target)
    WAIT             nothing wrong, but a step is not there yet (price not at a
                     level, or no confirmation candle)
    SETUP CANDIDATE  the chart conditions are met and a plan passes the rules.
                     It is not a prediction: the backtest tells how often such
                     setups worked in the past, and the pre-trade checklist
                     (are you calm? is it your own analysis?) is still yours.

`locate` and `setup_at_level` are also used by the backtest, so the live
analysis and the historical test apply exactly the same rules.

`frames` is {timeframe: candles up to "now"}. Nothing here looks past the
last closed candle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import patterns
from .indicators import last_atr
from .levels import Zone, build_zones, combine_timeframes, nearest_zones, role_for
from .planner import TradePlan, plan_at_level
from .structure import Structure, classify
from .swings import find_swings

AT_LEVEL = ("inside_zone", "at_support", "at_resistance")


@dataclass
class Location:
    support: Zone | None
    inside: Zone | None
    resistance: Zone | None
    pct: float | None          # 0 = at support, 100 = at resistance
    where: str                 # inside_zone, middle, at_support, at_resistance, no_range, between

    @property
    def level(self) -> Zone | None:
        """The zone price is at, if any."""
        if self.where == "inside_zone":
            return self.inside
        if self.where == "at_support":
            return self.support
        if self.where == "at_resistance":
            return self.resistance
        return None


def locate(price: float, last_low: float, last_high: float, zones: list[Zone],
           margin: float, cfg: dict) -> Location:
    """Where is price compared to the zones? `price` is the last close;
    last_low / last_high are the wick tips of that same candle."""
    scfg = cfg["strategy"]
    support, inside, resistance = nearest_zones(zones, price, scfg["require_validated_level"])
    pct = None
    if support and resistance:
        pct = (price - support.high) / (resistance.low - support.high) * 100
    half = scfg["middle_of_range_pct"] / 2

    # "At a level" is judged on the last closed candle's WICK, not only its
    # close: a hammer dips into support and closes back above it, so its close
    # is always a little away from the level.
    near_support = support is not None and support.distance_to(last_low) <= margin
    near_resistance = resistance is not None and resistance.distance_to(last_high) <= margin
    if near_support and near_resistance:    # one huge candle touched both: use the nearer one
        near_support = support.distance_to(price) <= resistance.distance_to(price)
        near_resistance = not near_support
    in_middle = pct is not None and 50 - half <= pct <= 50 + half

    if inside:
        where = "inside_zone"
    elif in_middle:
        # Rule 5 is absolute, so it is checked BEFORE "near a level". In a
        # narrow range the two can overlap, and the middle must win.
        where = "middle"
    elif near_support:
        where = "at_support"
    elif near_resistance:
        where = "at_resistance"
    elif pct is None:
        where = "no_range"
    else:
        where = "between"
    return Location(support, inside, resistance, pct, where)


def setup_at_level(loc: Location, hits: list[tuple[str, str, Zone]]
                   ) -> tuple[str, Zone, Zone | None, str] | None:
    """Is there a tradable reversal candle at the level price is at?

    `hits` are the (pattern, direction, zone) of ONE candle (see
    patterns.zone_signals). Returns (direction, level, target, pattern) or None.
    Bullish candles count at support, bearish at resistance, either inside a zone.
    """
    level = loc.level
    if level is None:
        return None
    allowed = {"at_support": ("bullish",), "at_resistance": ("bearish",),
               "inside_zone": ("bullish", "bearish")}[loc.where]
    for name, direction, zone in hits:
        if zone is level and direction in allowed:
            trade = "buy" if direction == "bullish" else "sell"
            return trade, level, (loc.resistance if trade == "buy" else loc.support), name
    return None


def trend_alignment(trend: str, direction: str) -> str:
    if trend in ("range", "unknown"):
        return "no clear trend"
    return "with trend" if (trend == "up") == (direction == "buy") else "counter-trend"


@dataclass
class Analysis:
    symbol: str
    as_of: pd.Timestamp                  # open time of the last closed confirmation candle
    price: float                         # its close
    structures: dict[str, Structure]
    zones: list[Zone]
    support: Zone | None
    inside: Zone | None
    resistance: Zone | None
    range_position_pct: float | None
    location: str
    signals: list[dict]
    verdict: str
    direction: str | None                # 'buy', 'sell' or None
    plan: TradePlan | None
    trend_alignment: str | None          # 'with trend', 'counter-trend', 'no clear trend'
    reasons: list[str] = field(default_factory=list)
    swings: dict[str, pd.DataFrame] = field(default_factory=dict)


def analysis_zones(cfg: dict, frames: dict[str, pd.DataFrame],
                   swings: dict[str, pd.DataFrame] | None = None) -> list[Zone]:
    """THE zones the verdict uses: built on each level timeframe (D1, H4),
    merged into one list, and labelled support/resistance from the last close
    of the confirmation timeframe (H1). The `chart` command, the dashboard and
    `analyze` all call this, so the picture and the verdict always show the
    same levels."""
    scfg = cfg["strategy"]
    level_tfs = [tf for tf in scfg["level_timeframes"] if tf in frames]
    n = cfg["swings"]["fractal_n"]
    swings = swings or {}
    zones = combine_timeframes(
        {tf: build_zones(frames[tf], cfg, tf,
                         swings[tf] if tf in swings else find_swings(frames[tf], n))
         for tf in level_tfs}, cfg)
    price = float(frames[scfg["confirmation_timeframe"]]["close"].iloc[-1])
    for z in zones:   # role reversal: role depends on which side price last closed
        z.role = role_for(z.low, z.high, price)
    return zones


def analyze(symbol: str, cfg: dict, frames: dict[str, pd.DataFrame],
            spread: float | None = None) -> Analysis:
    scfg = cfg["strategy"]
    conf_tf = scfg["confirmation_timeframe"]
    level_tfs = [tf for tf in scfg["level_timeframes"] if tf in frames]
    n = cfg["swings"]["fractal_n"]
    if spread is None:
        spread = cfg["symbols"].get(symbol, {}).get("typical_spread", 0.0)

    # --- structure on every timeframe we have ---
    swings = {tf: find_swings(df, n) for tf, df in frames.items()}
    structures = {tf: classify(frames[tf], cfg, tf, swings[tf]) for tf in frames}

    zones = analysis_zones(cfg, frames, swings)
    conf = frames[conf_tf]
    last = conf.iloc[-1]
    price = float(last["close"])

    conf_atr = last_atr(conf, cfg["atr"]["period"])
    margin = cfg["patterns"]["near_level_atr_mult"] * conf_atr
    loc = locate(price, float(last["low"]), float(last["high"]), zones, margin, cfg)

    # --- reversal candles at validated zones, on the confirmation timeframe ---
    pool = ([z for z in zones if z.status == "validated"]
            if scfg["require_validated_level"] else zones)
    signals = patterns.signals_at_zones(conf, pool, conf_atr, cfg)

    a = Analysis(
        symbol=symbol, as_of=conf.index[-1], price=price, structures=structures, zones=zones,
        support=loc.support, inside=loc.inside, resistance=loc.resistance,
        range_position_pct=loc.pct, location=loc.where, signals=signals, verdict="WAIT",
        direction=None, plan=None, trend_alignment=None, swings=swings,
    )
    _decide(a, loc, cfg, conf_tf, level_tfs, conf_atr, spread, frames)
    return a


def _decide(a: Analysis, loc: Location, cfg: dict, conf_tf: str, level_tfs: list[str],
            conf_atr: float, spread: float, frames: dict | None = None) -> None:
    """Fill verdict, direction, plan and the reasons."""
    scfg = cfg["strategy"]

    if a.location == "middle":
        a.verdict = "NO TRADE"
        a.reasons.append(
            f"Price is in the middle of the range ({a.range_position_pct:.0f}% of the way from "
            f"support to resistance). Rule: middle {scfg['middle_of_range_pct']}% = no trade.")
        return
    if a.location == "between":
        nearer = "support" if a.range_position_pct < 50 else "resistance"
        a.reasons.append(
            f"Price is not at a level ({a.range_position_pct:.0f}% of the range, nearer to "
            f"{nearer}). Wait for price to reach a validated level.")
        return
    if a.location == "no_range":
        missing = "resistance above" if a.support else "support below"
        if not a.support and not a.resistance:
            missing = "level on either side"
        a.reasons.append(
            f"No validated {missing} price in the data, so the range cannot be measured. "
            "Wait for price to reach a validated level.")
        return

    # --- price is at a level ---
    level = loc.level
    where = {"at_support": "at validated support", "at_resistance": "at validated resistance",
             "inside_zone": "inside a validated zone"}[a.location]
    wanted = {"at_support": "a hammer or bullish engulfing",
              "at_resistance": "a shooting star or bearish engulfing",
              "inside_zone": "a reversal candle (hammer / shooting star / engulfing)"}[a.location]

    # Only the candle that JUST closed can trigger an entry "on the close of
    # the confirmation candle".
    now_hits = [(s["pattern"], s["direction"], s["zone"]) for s in a.signals
                if s["bars_ago"] == 0]
    setup = setup_at_level(loc, now_hits)
    if setup is None:
        older = [s for s in a.signals if s["zone"] is level and s["direction"] != "neutral"
                 and s["bars_ago"] > 0]
        if older:
            s = older[0]
            a.reasons.append(
                f"Price is {where} ({_fmt_zone(level)}). A {s['label']} printed "
                f"{s['bars_ago']} candle(s) ago: the entry at its close has passed. Wait for a "
                "new confirmation candle or a retest of the level.")
        else:
            a.reasons.append(
                f"Price is {where} ({_fmt_zone(level)}), but there is no confirmation candle "
                f"yet. Wait for {wanted} to CLOSE on {conf_tf}.")
        return

    a.direction, level, target, pattern = setup
    a.reasons.append(
        f"Price is {where} ({_fmt_zone(level)}) and the last closed {conf_tf} candle is a "
        f"{patterns.LABELS[pattern]}.")

    plan, problem = plan_at_level(a.direction, a.price, level, target, conf_atr, spread, cfg)
    if plan is None:
        a.verdict = "NO TRADE"
        a.reasons.append(problem)
        return

    # Extra conditions from config.yaml (filters.level), checked with exactly
    # the code the backtest used.
    spec = cfg.get("filters", {}).get("level") or {}
    if spec and frames is not None:
        from .backtest import History          # imported here to avoid an import loop
        from .filters import failed_conditions
        from .strategies import level_info
        h = History(a.symbol, cfg, frames)
        info = level_info(h, h.n - 1, a.direction, level, pattern, a.location, plan.rr)
        failed = failed_conditions(spec, info)
        if failed:
            a.verdict = "NO TRADE"
            a.reasons.append("The setup fails the extra conditions that the backtest needed "
                             "to stop losing: " + "; ".join(failed) + ".")
            return

    a.plan = plan
    trend_tf = "H4" if "H4" in a.structures else level_tfs[-1]
    trend = a.structures[trend_tf].trend
    a.trend_alignment = trend_alignment(trend, a.direction)
    if a.trend_alignment == "counter-trend" and not scfg["allow_counter_trend"]:
        a.plan = None
        a.verdict = "NO TRADE"
        a.reasons.append(f"The {a.direction} is AGAINST the {trend_tf} trend, and "
                         "strategy.allow_counter_trend is false.")
        return
    a.verdict = "SETUP CANDIDATE"
    a.reasons.append(f"Planned risk:reward is 1:{plan.rr:.1f} (minimum "
                     f"1:{cfg['stops']['min_risk_reward']:g}).")
    a.reasons.append({
        "with trend": f"The {a.direction} is WITH the {trend_tf} trend.",
        "counter-trend": f"The {a.direction} is AGAINST the {trend_tf} trend: lower conviction.",
        "no clear trend": f"{trend_tf} has no clear trend: medium conviction.",
    }[a.trend_alignment])


def _fmt_zone(z: Zone) -> str:
    plural = "touch" if z.touches == 1 else "touches"
    return f"{z.timeframe} zone {z.low:g} to {z.high:g}, {z.touches} {plural}"
