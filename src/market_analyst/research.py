"""Research: look for settings and extra conditions that improve a strategy,
without fooling ourselves.

The danger: try enough combinations and one of them will look great on past
data by pure luck ("overfitting"). The protection used here:

  1. History is cut in two. The older part is IN-SAMPLE: we are allowed to
     search on it. The newest part is OUT-OF-SAMPLE: it is never used to
     choose anything.
  2. The search is small and fixed in advance (a short list of settings and
     conditions), and the report says how many combinations were tried.
  3. One setting is chosen for BOTH symbols together, not one per symbol.
  4. The chosen setting is then scored on the out-of-sample part, once.
     Only that score says anything about the future, and even it is one
     sample of a few dozen trades.

A setting "holds up" only if it is positive in-sample AND out-of-sample.
"""

from __future__ import annotations

import copy

import pandas as pd

from .backtest import History, Trade, run, summarize
from .filters import make_accept
from .strategies import STRATEGIES

MIN_TRADES = 40          # a candidate needs at least this many in-sample trades
MIN_GAIN = 0.05          # ... and must add at least 0.05 R per trade to be kept
MAX_CONDITIONS = 3       # at most three extra conditions (more = more overfitting)

# --- settings that change how trades are built (need a full replay) -------------
LEVEL_GRID = [
    {"stops.min_risk_atr_mult": room, "stops.tp_mode": mode, "stops.fixed_rr": rr}
    for room in (0.0, 0.5, 1.0, 1.5)
    for mode, rr in (("next_level", 2.0), ("fixed_rr", 2.0), ("fixed_rr", 3.0))
]
STRUCTURE_GRID = {
    "level": LEVEL_GRID,
    "retest": LEVEL_GRID,
    "h2l2": [{"strategies.h2l2.target_rr": rr, "strategies.h2l2.min_pullback_atr": depth}
             for rr in (1.0, 1.5, 2.0, 3.0) for depth in (1.0, 2.0)],
}

# --- what to do with a trade that is in profit (also needs a full replay) --------
MANAGEMENT_GRID = [
    {"stops.manage_at_r": 1.0, "stops.partial_fraction": 0.0},   # stop to breakeven at +1R
    {"stops.manage_at_r": 1.0, "stops.partial_fraction": 0.5},   # close half at +1R, rest at BE
    {"stops.manage_at_r": 1.5, "stops.partial_fraction": 0.0},
]

# --- extra entry conditions (checked on facts known when the signal appeared) ----
COMMON = [
    ("with the H4 trend", {"trend_H4": ["with trend"]}),
    ("not against the H4 trend", {"trend_H4": ["with trend", "no clear trend"]}),
    ("with the D1 trend", {"trend_D1": ["with trend"]}),
    ("not against the D1 trend", {"trend_D1": ["with trend", "no clear trend"]}),
    ("London or overlap session only", {"session": ["london", "overlap"]}),
    ("not during Asia", {"session": ["london", "overlap", "late"]}),
    ("on the right side of the 200 EMA", {"ema200_side": True}),
    ("50 EMA sloping with the trade", {"ema50_slope": True}),
    ("not a wild market (ATR <= 1.3 x usual)", {"max_atr_ratio": 1.3}),
    ("not a dead market (ATR >= 0.8 x usual)", {"min_atr_ratio": 0.8}),
]
AT_LEVEL = [
    ("zone has 3+ touches", {"min_zone_touches": 3}),
    ("zone also seen on D1", {"zone_d1": True}),
    ("zone had a role reversal", {"zone_role_reversal": True}),
    ("engulfing candles only", {"pattern": ["bullish_engulfing", "bearish_engulfing"]}),
    ("hammer / shooting star only", {"pattern": ["hammer", "shooting_star"]}),
    ("not inside the zone", {"location": ["at_support", "at_resistance"]}),
    ("only when D1 is ranging (regime switch)", {"regime_D1": ["range"]}),
]
CONDITIONS = {
    "level": COMMON + AT_LEVEL,
    "retest": COMMON + AT_LEVEL,
    "h2l2": COMMON + [
        ("pullback reached the 20 EMA", {"touched_ema": True}),
        ("pullback at least 2 x ATR deep", {"min_pullback_atr": 2.0}),
        ("pullback of 10 candles or fewer", {"max_pullback_bars": 10}),
        ("only when D1 is trending (regime switch)", {"regime_D1": ["up", "down"]}),
    ],
}


def with_overrides(cfg: dict, overrides: dict) -> dict:
    """Copy of cfg with dotted keys replaced: {"stops.fixed_rr": 3} etc."""
    out = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        node = out
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    return out


def split_time(h: History) -> pd.Timestamp:
    """Start of the out-of-sample part for one symbol."""
    usable = h.n - h.first_bar
    return h.time[h.first_bar + int(usable * (1 - h.cfg["backtest"]["out_of_sample_fraction"]))]


def run_all(histories: dict[str, History], name: str, cfg: dict, spec: dict | None = None
            ) -> list[Trade]:
    """Replay one strategy on every symbol; trades of all symbols in one list."""
    trades: list[Trade] = []
    for h in histories.values():
        trades += run(h, STRATEGIES[name](cfg, accept=make_accept(spec or {})))
    return trades


def split(trades: list[Trade], cuts: dict[str, pd.Timestamp]
          ) -> tuple[list[Trade], list[Trade]]:
    """(in-sample, out-of-sample) by the time the signal appeared."""
    inside = [t for t in trades if t.signal_time < cuts[t.symbol]]
    outside = [t for t in trades if t.signal_time >= cuts[t.symbol]]
    return inside, outside


def study(histories: dict[str, History], name: str, cfg: dict) -> dict:
    """Search on the in-sample part, then score the choice out-of-sample."""
    cuts = {sym: split_time(h) for sym, h in histories.items()}
    tried = 0

    def score(trades):
        s = summarize(trades)
        return s["expectancy_r"] if s["trades"] >= MIN_TRADES else None

    # --- baseline: the strategy exactly as configured ---
    base_in, base_out = split(run_all(histories, name, cfg), cuts)

    # --- step 1: structure (stop distance, target) chosen in-sample ---
    best_over, best_score, best_trades = {}, score(base_in), base_in
    for overrides in STRUCTURE_GRID[name]:
        tried += 1
        inside, _ = split(run_all(histories, name, with_overrides(cfg, overrides)), cuts)
        value = score(inside)
        if value is not None and (best_score is None or value > best_score):
            best_over, best_score, best_trades = overrides, value, inside

    # --- step 1b: trade management, on top of the chosen structure ---
    structure = dict(best_over)
    for extra in MANAGEMENT_GRID:
        tried += 1
        overrides = {**structure, **extra}
        inside, _ = split(run_all(histories, name, with_overrides(cfg, overrides)), cuts)
        value = score(inside)
        if value is not None and best_score is not None and value > best_score + MIN_GAIN:
            best_over, best_score, best_trades = overrides, value, inside

    # --- step 2: add conditions one at a time while they help, in-sample ---
    spec: dict = {}
    labels: list[str] = []
    current = best_trades
    for _ in range(MAX_CONDITIONS):
        options = []
        for label, cond in CONDITIONS[name]:
            if any(key in spec for key in cond):
                continue
            tried += 1
            keep = make_accept({**spec, **cond})
            kept = [t for t in current if keep(t.info)]
            value = score(kept)
            if value is not None:
                options.append((value, label, cond))
        if not options:
            break
        value, label, cond = max(options, key=lambda o: o[0])
        if best_score is not None and value < best_score + MIN_GAIN:
            break
        spec.update(cond)
        labels.append(label)
        best_score = value
        keep = make_accept(spec)
        current = [t for t in current if keep(t.info)]

    # --- step 3: replay the chosen setting properly and score both parts ---
    tuned_cfg = with_overrides(cfg, best_over)
    tuned_in, tuned_out = split(run_all(histories, name, tuned_cfg, spec), cuts)
    per_symbol = {
        sym: summarize([t for t in tuned_out if t.symbol == sym]) for sym in histories}
    return {
        "strategy": name, "combinations_tried": tried, "cuts": cuts,
        "overrides": best_over, "filters": spec, "labels": labels,
        "baseline_in": summarize(base_in), "baseline_out": summarize(base_out),
        "tuned_in": summarize(tuned_in), "tuned_out": summarize(tuned_out),
        "tuned_out_by_symbol": per_symbol,
    }


def verdict(result: dict) -> tuple[str, str]:
    """Judge the tuned setting on the UNSEEN part only. Returns (label, explanation).

    EDGE SHOWN      positive, and even the low end of the 90% range is above 0
    BREAK-EVEN      positive on average, but the range includes 0: not
                    distinguishable from no edge
    NO EDGE         zero or negative on the unseen part
    """
    out = result["tuned_out"]
    if out["trades"] < 20:
        return "TOO FEW TRADES", "Fewer than 20 unseen trades: nothing can be concluded."
    if out["expectancy_r"] <= 0:
        return "NO EDGE", ("The tuned version is not positive on the unseen part. "
                           "Do not trade it.")
    if out["ci_low"] <= 0:
        return "BREAK-EVEN", ("Slightly positive on the unseen part, but the 90% range "
                              "includes 0: this is not distinguishable from no edge.")
    return "EDGE SHOWN", ("Positive on the unseen part even at the low end of the range. "
                          "Still one sample: keep testing on demo before trusting it.")
