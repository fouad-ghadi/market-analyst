"""The analysis desk: a dark, live-updating web page for EURUSD and GOLD.

`market-analyst desk` builds reports/desk.html. With `--watch` it keeps
running, refreshes the data every `desk.refresh_seconds`, and the open page
reloads itself, so it works like a trading terminal panel.

What makes it different from a "signals" site:
  * Every number on the page comes from this tool's own code (the same
    functions `analyze`, `deals` and `backtest` use). Nothing is invented,
    and there are NO confidence percentages: the research found no proven
    edge (RESEARCH.md), so the page shows rule checks, not predictions.
  * The event feed is point-in-time: each event is computed with only the
    candles that existed when it happened (no look-ahead).
  * Analysis only. There are no order buttons and no code that sends orders.

How the page is built:
  * Python computes everything and writes the panels as HTML (this file).
  * A small script inside the page only DRAWS: the candle chart (as SVG),
    the clocks, and the symbol / timeframe switches. It does no analysis.
  * The page is one self-contained file: nothing is loaded from the internet.

Main functions (each one is tested on its own):
    session_status()   is the market open? is it the London-New York overlap?
    checklist()        the rulebook strategy steps, each pass / fail / wait / n/a
    recent_events()    what happened on the last H1 candles, as known at the time
    snapshot()         everything about one symbol, as plain data
    build_page()       the HTML page from the snapshots
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timedelta

import pandas as pd

from . import patterns, report
from .analysis import AT_LEVEL, Analysis
from .backtest import TF_HOURS, History
from .deals import STRATEGY_TEXT, Deal
from .indicators import last_atr
from .strategies import STRATEGIES
from .swings import find_swings

TREND_WORD = {"up": "Up", "down": "Down", "range": "Range", "unknown": "n/a"}
TREND_ICON = {"up": "&#9650;", "down": "&#9660;", "range": "&#9670;", "unknown": "&ndash;"}
STATE_ICON = {"pass": "&#10003;", "fail": "&#10005;", "wait": "&#9711;", "na": "&ndash;",
              "warn": "!", "you": "?"}
STATE_WORD = {"pass": "yes", "fail": "no", "wait": "not yet", "na": "n/a", "warn": "caution",
              "you": "your call"}
VERDICT_CLASS = {"NO TRADE": "fail", "WAIT": "wait", "SETUP CANDIDATE": "pass"}


def esc(value) -> str:
    return html.escape(str(value))


# =============================================================================
# 1. Session: is the market open, are we in the best hours?
# =============================================================================

def _hhmm(text: str) -> tuple[int, int]:
    h, m = text.split(":")
    return int(h), int(m)


def session_status(cfg: dict, now: datetime) -> dict:
    """Market open/closed (weekend) and the London-New York overlap window.

    Weekend = Friday 22:00 UTC to Sunday 21:00 UTC (roughly, broker dependent).
    Returns {"market": "open"/"closed", "overlap": bool, "text": ..., "next": ...}.
    """
    wd, hour = now.weekday(), now.hour          # Monday = 0 ... Sunday = 6
    closed = wd == 5 or (wd == 4 and hour >= 22) or (wd == 6 and hour < 21)
    win = cfg["time"]["best_session_utc"]
    (sh, sm), (eh, em) = _hhmm(win["start"]), _hhmm(win["end"])
    start = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end = now.replace(hour=eh, minute=em, second=0, microsecond=0)
    in_overlap = not closed and wd < 5 and start <= now < end
    if closed:
        text, nxt = "Market closed (weekend)", "Opens Sunday about 21:00 UTC"
    elif in_overlap:
        mins = int((end - now).total_seconds() // 60)
        text, nxt = "London-New York overlap: best hours", f"Ends in {mins // 60} h {mins % 60:02d} min"
    else:
        day = start if now < start else start + timedelta(days=1)
        while day.weekday() >= 5:
            day += timedelta(days=1)
        mins = int((day - now).total_seconds() // 60)
        text = "Market open, outside the best hours"
        nxt = f"Overlap starts in {mins // 60} h {mins % 60:02d} min"
    return {"market": "closed" if closed else "open", "overlap": in_overlap,
            "text": text, "next": nxt, "window": f"{win['start']}-{win['end']} UTC"}


# =============================================================================
# 2. Rulebook checklist (the strategy steps, from the analysis)
# =============================================================================

def checklist(a: Analysis, cfg: dict) -> list[dict]:
    """Turn the verdict into the rulebook steps, each with a state:
    pass / fail / wait (not there yet) / na (does not apply yet) / warn."""
    scfg = cfg["strategy"]
    at_level = a.location in AT_LEVEL
    last = a.reasons[-1] if a.reasons else ""
    steps = []

    # 1. At a validated level, not the middle of the range
    if at_level:
        steps.append(("At a validated level", "pass", "Price is at a D1/H4 zone with 2+ touches."))
    elif a.location == "middle":
        steps.append(("At a validated level", "fail",
                      f"Middle {scfg['middle_of_range_pct']}% of the range: no trade."))
    else:
        steps.append(("At a validated level", "wait", "Price is between levels."))

    # 2. Reversal candle closed at the level
    if a.direction:
        steps.append(("Confirmation candle at the level", "pass", a.reasons[0]))
    elif at_level:
        steps.append(("Confirmation candle at the level", "wait",
                      "No hammer / shooting star / engulfing has closed at the level yet."))
    else:
        steps.append(("Confirmation candle at the level", "na", "Only checked at a level."))

    # 3. R:R at least the minimum, 4. extra tested conditions
    min_rr = cfg["stops"]["min_risk_reward"]
    extra = "extra conditions" in last
    counter_refused = "allow_counter_trend" in last
    if a.plan is not None:
        steps.append((f"Risk:reward at least 1:{min_rr:g}", "pass", f"Planned 1:{a.plan.rr:.1f}."))
    elif a.direction and a.verdict == "NO TRADE" and not extra and not counter_refused:
        steps.append((f"Risk:reward at least 1:{min_rr:g}", "fail", last))
    else:
        steps.append((f"Risk:reward at least 1:{min_rr:g}", "na", "Needs a setup first."))
    if cfg.get("filters", {}).get("level"):
        state = "fail" if extra else "pass" if a.plan is not None else "na"
        steps.append(("Tested extra conditions (config filters)", state,
                      last if extra else "Conditions from the research, see config.yaml."))

    # 5. With the H4 trend
    h4 = a.structures.get("H4")
    trend_text = f"H4 is {TREND_WORD[h4.trend].lower()}." if h4 else ""
    if a.trend_alignment == "with trend":
        steps.append(("With the H4 trend", "pass", trend_text))
    elif a.trend_alignment in ("counter-trend", "no clear trend") or counter_refused:
        steps.append(("With the H4 trend", "fail" if counter_refused else "warn",
                      f"{trend_text} Lower conviction."))
    else:
        steps.append(("With the H4 trend", "na", trend_text or "Needs a setup first."))
    return [{"step": s, "state": st, "detail": d} for s, st, d in steps]


def manual_checks(cfg: dict) -> list[str]:
    """The checklist boxes only you can tick: the tool does not read your mind
    or your account."""
    r = cfg["risk"]
    return [
        "Can I explain WHY with my own analysis (not a Telegram signal)?",
        "Am I calm, not chasing a loss or a move?",
        f"Risking {r['risk_pct_per_trade']:g}% at most ({r['fixed_lot']} lot while learning)?",
        f"At most {r['max_open_trades']} open trade; stop after {r['max_losses_per_day']} "
        f"losses or -{r['max_daily_loss_pct']:g}% today.",
        f"Pending order expiry set to \"{cfg['orders']['pending_expiration']}\"?",
    ]


# =============================================================================
# 3. Event feed: what happened on the recent H1 candles (point in time)
# =============================================================================

def _zone_text(z, f) -> str:
    return f"{z.timeframe} zone {f(z.low)}-{f(z.high)} ({z.touches}x)"


def recent_events(h: History, cfg: dict, f, hours: int) -> list[dict]:
    """Events on the last `hours` H1 candles, newest first.

    For candle i, only what was known at its close is used (h.snapshot(i)
    rebuilds the zones as of that candle), so the feed never shows a level
    that did not exist yet. Events:
      * a reversal candle at a validated zone (the rulebook's confirmation)
      * a close through a validated zone (break: role reversal candidate)
      * a change of trend on H1 / H4 / D1 (structure)
      * price entering the middle of the range (no-trade area)
    """
    out = []
    start = max(h.first_bar + 1, h.n - hours)
    prev_where = None
    for i in range(start, h.n):
        t = h.time[i] + pd.Timedelta(hours=TF_HOURS[h.entry_tf])     # candle close time
        snap = h.snapshot(i)
        for name, direction, zone in snap.hits:
            if direction == "neutral":
                continue
            out.append({"time": t, "kind": "pattern", "tone": direction,
                        "text": f"{patterns.LABELS[name].capitalize()} closed at "
                                f"{_zone_text(zone, f)}"})
        for z in snap.zones:
            if z.status != "validated":
                continue
            before, now = h.close[i - 1], h.close[i]
            if before <= z.high < now:
                out.append({"time": t, "kind": "break", "tone": "bullish",
                            "text": f"Closed ABOVE {_zone_text(z, f)}: resistance may become support"})
            elif before >= z.low > now:
                out.append({"time": t, "kind": "break", "tone": "bearish",
                            "text": f"Closed BELOW {_zone_text(z, f)}: support may become resistance"})
        for tf in [h.entry_tf] + h.level_tfs:
            old, new = h.trend(tf, i - 1), h.trend(tf, i)
            if old != new and new != "unknown":
                tone = {"up": "bullish", "down": "bearish"}.get(new, "neutral")
                out.append({"time": t, "kind": "structure", "tone": tone,
                            "text": f"{tf} structure: {TREND_WORD[old].lower()} -> "
                                    f"{TREND_WORD[new].lower()}"})
        where = snap.loc.where
        if where == "middle" and prev_where not in (None, "middle"):
            out.append({"time": t, "kind": "location", "tone": "neutral",
                        "text": "Price entered the middle of the range (no-trade area)"})
        prev_where = where
    out.reverse()
    return out


# =============================================================================
# 4. One symbol as plain data
# =============================================================================

def _candles(df: pd.DataFrame, n: int) -> list[list]:
    view = df.iloc[-n:]
    return [[int(t.timestamp()), float(o), float(hi), float(lo), float(c)]
            for t, o, hi, lo, c in zip(view.index, view["open"], view["high"],
                                       view["low"], view["close"])]


def _swings(df: pd.DataFrame, n: int, cfg: dict) -> list[list]:
    s = find_swings(df, cfg["swings"]["fractal_n"])
    first = len(df) - n
    s = s[s["pos"] >= first]
    return [[int(r.pos - first), r.kind, float(r.price)] for r in s.itertuples()]


def snapshot(cfg: dict, symbol: str, frames: dict[str, pd.DataFrame], a: Analysis,
             deals: dict[str, Deal | None], history: History | None,
             news_lines: list[str] | None, live: dict | None) -> dict:
    """Everything the page shows about one symbol. Numbers only come from the
    analysis, the deals and the backtest; nothing is estimated here."""
    dcfg = cfg["desk"]
    dec = cfg["symbols"][symbol].get("price_decimals", 5)
    f = lambda v: f"{v:.{dec}f}"  # noqa: E731
    conf = frames[cfg["strategy"]["confirmation_timeframe"]]
    atr_h1 = last_atr(conf, cfg["atr"]["period"])
    shown_price = live["bid"] if live else a.price
    prev_day = float(frames["D1"]["close"].iloc[-1]) if "D1" in frames else None

    near = {id(z) for z in report.select_zones(a.zones, a.price, cfg["report"]["levels_each_side"])}
    zones = []
    for z in a.zones:
        dist = z.distance_to(a.price)
        zones.append({
            "low": z.low, "high": z.high, "tf": "+".join([z.timeframe] + z.confirmed_by),
            "touches": z.touches, "status": z.status, "role": z.role,
            "note": "role reversal" if z.role_reversal else
                    "broken, not retested" if z.untested_flip else "",
            "dist_atr": round(dist / atr_h1, 1) if atr_h1 else None,
            "inside": z.low <= a.price <= z.high,
            "near": id(z) in near,          # the chart draws only the zones nearest to price
        })

    plan = None
    if a.plan is not None:
        plan = {"entry": a.plan.entry, "sl": a.plan.sl, "tp": a.plan.tp, "rr": a.plan.rr,
                "direction": a.direction}
    deal_rows = []
    for name in STRATEGIES:
        d = deals.get(name)
        row = {"strategy": name, "why": STRATEGY_TEXT[name], "deal": None, "record": None}
        if d is not None:
            row["deal"] = {"order": d.order_text, "kind": d.kind, "direction": d.direction,
                           "entry": d.entry, "sl": d.sl, "tp": d.tp, "rr": d.rr,
                           "risk": None if d.risk_dollars is None else round(d.risk_dollars, 2),
                           "pattern": d.pattern}
            row["record"] = {"trades": d.record.trades, "avg_r": d.record.avg_r,
                             "new_trades": d.record.new_trades, "new_avg_r": d.record.new_avg_r,
                             "weak": d.record.weak}
        deal_rows.append(row)

    charts, swings = {}, {}
    for tf in dcfg["chart_timeframes"]:
        if tf in frames:
            n = min(dcfg["chart_bars"], len(frames[tf]))
            charts[tf] = _candles(frames[tf], n)
            swings[tf] = _swings(frames[tf], n, cfg)

    events = recent_events(history, cfg, f, dcfg["feed_hours"]) if history is not None else []

    return {
        "symbol": symbol, "decimals": dec, "price": a.price, "shown_price": shown_price,
        "live": live, "prev_day_close": prev_day,
        "day_change": None if prev_day is None else shown_price - prev_day,
        "as_of": a.as_of + pd.Timedelta(hours=TF_HOURS[cfg["strategy"]["confirmation_timeframe"]]),
        "atr_h1": atr_h1,
        "verdict": a.verdict, "direction": a.direction, "reasons": list(a.reasons),
        "alignment": a.trend_alignment, "plan": plan,
        "checklist": checklist(a, cfg),
        "structure": {tf: {"trend": s.trend, "labels": f"{s.highs_label} + {s.lows_label}",
                           "broken": s.broken, "note": s.note}
                      for tf, s in a.structures.items()},
        "location": a.location, "range_pct": a.range_position_pct,
        "support": None if a.support is None else [a.support.low, a.support.high],
        "resistance": None if a.resistance is None else [a.resistance.low, a.resistance.high],
        "zones": zones, "deals": deal_rows, "events": events[:dcfg["feed_max_items"]],
        "news": news_lines, "charts": charts, "swings": swings,
    }


# =============================================================================
# 5. The page
# =============================================================================

CSS = """
:root { color-scheme: dark;
  --bg:#070b10; --panel:#0d141c; --panel-2:#111b25; --line:#1b2834; --line-2:#26384a;
  --ink:#dbe6f0; --ink-2:#9fb0c0; --ink-3:#6c7f91; --accent:#38c6d4; --accent-2:#1f7f8a;
  --up:#2fbf71; --down:#ef5a5f; --warn:#f2b33d; --zone:#38c6d4; }
* { box-sizing:border-box; }
html,body { margin:0; background:var(--bg); color:var(--ink); }
body { font:13px/1.45 "Segoe UI", Inter, Roboto, Helvetica, Arial, sans-serif; }
.mono, .num { font-family: "Cascadia Mono", "JetBrains Mono", Consolas, ui-monospace, monospace;
  font-variant-numeric: tabular-nums; }
.app { display:grid; grid-template-rows:auto auto 1fr auto; min-height:100vh; }
header.bar { display:flex; align-items:center; gap:18px; padding:10px 18px;
  border-bottom:1px solid var(--line); background:linear-gradient(180deg,#0b131b,#080d13); }
.brand { font-weight:700; letter-spacing:.14em; font-size:14px; color:var(--accent); }
.brand small { display:block; letter-spacing:.24em; font-size:9px; color:var(--ink-3); font-weight:600; }
.pill { border:1px solid var(--line-2); border-radius:999px; padding:3px 10px; font-size:11px;
  color:var(--ink-2); white-space:nowrap; display:inline-flex; align-items:center; gap:6px; }
.pill .dot { width:7px; height:7px; border-radius:50%; background:var(--ink-3); }
.pill.on .dot { background:var(--up); box-shadow:0 0 8px var(--up); }
.pill.off .dot { background:var(--down); }
.pill.ro { border-color:var(--accent-2); color:var(--accent); }
.spacer { flex:1; }
.clock { text-align:right; color:var(--ink-2); font-size:11px; }
.clock b { color:var(--ink); font-size:15px; }
.ticker { display:flex; gap:0; border-bottom:1px solid var(--line); overflow-x:auto; }
.tick { display:flex; gap:12px; align-items:baseline; padding:8px 18px; border-right:1px solid var(--line);
  cursor:pointer; background:none; border-top:0; border-bottom:2px solid transparent; border-left:0;
  color:var(--ink); font:inherit; text-align:left; }
.tick.sel { border-bottom-color:var(--accent); background:var(--panel); }
.tick .sym { font-weight:700; letter-spacing:.06em; }
.tick .px { font-size:15px; }
.up { color:var(--up); } .down { color:var(--down); } .muted { color:var(--ink-3); }
.honest { padding:7px 18px; font-size:12px; color:var(--ink-2); border-bottom:1px solid var(--line);
  background:#0a1016; }
.honest b { color:var(--warn); }
main { display:grid; gap:10px; padding:10px; grid-template-columns: 290px minmax(0,1fr) 360px; }
@media (max-width:1280px) { main { grid-template-columns: 260px minmax(0,1fr); }
  .col.right { grid-column: 1 / -1; }
  .col.right .sym-view.sel { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); } }
@media (max-width:860px) { main { grid-template-columns: 1fr; } }
.col { display:flex; flex-direction:column; gap:10px; min-width:0; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px 14px;
  position:relative; }
.panel::before { content:""; position:absolute; left:14px; top:-1px; width:28px; height:2px;
  background:var(--accent); border-radius:2px; }
.panel h3 { margin:0 0 10px; font-size:10.5px; letter-spacing:.16em; text-transform:uppercase;
  color:var(--accent); font-weight:700; display:flex; justify-content:space-between; gap:8px; }
.panel h3 .hint { color:var(--ink-3); letter-spacing:.06em; font-weight:600; }
table { width:100%; border-collapse:collapse; }
td, th { padding:5px 4px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
th { font-size:10px; letter-spacing:.1em; text-transform:uppercase; color:var(--ink-3); font-weight:600; }
td.r, th.r { text-align:right; }
tr:last-child td { border-bottom:0; }
.mtf td.c { text-align:center; }
.trend { font-weight:700; white-space:nowrap; }
.trend.up { color:var(--up); } .trend.down { color:var(--down); } .trend.range { color:var(--warn); }
.trend.unknown { color:var(--ink-3); }
.sym-view { display:none; } .sym-view.sel { display:flex; flex-direction:column; gap:10px; }
.verdict { display:flex; align-items:center; gap:12px; margin-bottom:8px; }
.vbadge { font-weight:800; letter-spacing:.08em; font-size:16px; padding:6px 12px; border-radius:6px;
  border:1px solid; display:inline-flex; gap:8px; align-items:center; }
.vbadge.pass { color:var(--up); border-color:var(--up); background:rgba(47,191,113,.08); }
.vbadge.fail { color:var(--down); border-color:var(--down); background:rgba(239,90,95,.08); }
.vbadge.wait { color:var(--warn); border-color:var(--warn); background:rgba(242,179,61,.08); }
ul.steps { list-style:none; padding:0; margin:0; }
ul.steps li { display:grid; grid-template-columns:22px 1fr auto; gap:8px; padding:6px 0;
  border-bottom:1px solid var(--line); align-items:start; }
ul.steps li:last-child { border-bottom:0; }
.si { width:18px; height:18px; border-radius:50%; display:grid; place-items:center; font-size:11px;
  font-weight:800; border:1px solid var(--ink-3); color:var(--ink-3); }
.si.pass { border-color:var(--up); color:var(--up); } .si.fail { border-color:var(--down); color:var(--down); }
.si.wait { border-color:var(--warn); color:var(--warn); } .si.warn { border-color:var(--warn); color:var(--warn); }
.si.you { border-color:var(--accent); color:var(--accent); }
.sw { font-size:10px; text-transform:uppercase; letter-spacing:.08em; color:var(--ink-3); white-space:nowrap; }
.detail { color:var(--ink-3); font-size:11.5px; }
.reasons { margin:6px 0 0; padding-left:16px; color:var(--ink-2); }
.reasons li { margin:3px 0; }
.levels-grid { display:grid; grid-template-columns:repeat(4,auto); gap:2px 16px; justify-content:start; margin-top:8px; }
.levels-grid .lab { font-size:10px; color:var(--ink-3); letter-spacing:.08em; text-transform:uppercase; display:block; }
.chart-head { display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:6px; }
.seg { display:inline-flex; border:1px solid var(--line-2); border-radius:6px; overflow:hidden; }
.seg button { background:none; border:0; color:var(--ink-2); padding:4px 10px; font:inherit; font-size:11px; cursor:pointer; }
.seg button.sel { background:var(--accent-2); color:#fff; }
.chart { width:100%; height:440px; position:relative; }
.chart svg { width:100%; height:100%; display:block; }
.tip { position:absolute; pointer-events:none; background:#0b1219ee; border:1px solid var(--line-2);
  border-radius:6px; padding:6px 8px; font-size:11px; display:none; white-space:nowrap; z-index:2; }
.legend { display:flex; gap:14px; flex-wrap:wrap; color:var(--ink-3); font-size:11px; margin-top:6px; }
.legend i { display:inline-block; width:14px; height:8px; margin-right:5px; vertical-align:middle; border-radius:2px; }
.gauge { position:relative; height:26px; margin:16px 4px 6px; border-radius:4px;
  background:linear-gradient(90deg, rgba(47,191,113,.25), rgba(47,191,113,.05) 30%, rgba(255,255,255,.02) 50%, rgba(239,90,95,.05) 70%, rgba(239,90,95,.25)); border:1px solid var(--line-2); }
.gauge .mid { position:absolute; top:0; bottom:0; background:repeating-linear-gradient(45deg, rgba(242,179,61,.18) 0 6px, transparent 6px 12px);
  border-left:1px dashed var(--warn); border-right:1px dashed var(--warn); }
.gauge .mark { position:absolute; top:-7px; bottom:-7px; width:3px; background:var(--ink); border-radius:2px;
  box-shadow:0 0 10px var(--accent); }
.gauge-labels { display:flex; justify-content:space-between; font-size:11px; color:var(--ink-2); }
.ladder td { font-size:12px; }
.ladder tr.price td { color:var(--accent); font-weight:700; border-bottom:1px dashed var(--accent-2); }
.ladder tr.in td { color:var(--accent); }
.tag { font-size:10px; padding:1px 6px; border-radius:4px; border:1px solid var(--line-2); color:var(--ink-2); white-space:nowrap; }
.tag.val { border-color:var(--accent-2); color:var(--accent); }
.deal { border:1px solid var(--line); border-radius:6px; padding:8px 10px; margin-bottom:8px; background:var(--panel-2); cursor:default; }
.deal:hover { border-color:var(--accent-2); }
.deal .t { font-weight:700; }
.deal.none { color:var(--ink-3); padding:6px 10px; background:none; }
.rec { display:flex; align-items:center; gap:8px; margin-top:6px; font-size:11px; color:var(--ink-3); }
.bar0 { position:relative; display:inline-block; vertical-align:middle; width:110px; height:8px; background:var(--line); border-radius:4px; flex:none; }
.bar0::after { content:""; position:absolute; left:50%; top:-2px; bottom:-2px; width:1px; background:var(--ink-3); }
.bar0 span { position:absolute; top:0; bottom:0; border-radius:4px; }
.feed { list-style:none; margin:0; padding:0; max-height:330px; overflow-y:auto; }
.feed li { padding:6px 0 6px 12px; border-bottom:1px solid var(--line); position:relative; }
.feed li::before { content:""; position:absolute; left:0; top:11px; width:6px; height:6px; border-radius:50%; background:var(--ink-3); }
.feed li.bullish::before { background:var(--up); } .feed li.bearish::before { background:var(--down); }
.feed li.neutral::before { background:var(--warn); }
.feed .when { font-size:10.5px; color:var(--ink-3); }
.feed .k { font-size:9.5px; letter-spacing:.12em; text-transform:uppercase; color:var(--ink-3); margin-left:6px; }
.news li { margin:4px 0; }
footer { padding:8px 18px; border-top:1px solid var(--line); color:var(--ink-3); font-size:11px;
  display:flex; gap:16px; flex-wrap:wrap; justify-content:space-between; }
"""

# The script only draws. All numbers it shows come from the JSON written by Python.
JS = r"""
(function(){
const DATA = JSON.parse(document.getElementById('desk-data').textContent);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const state = Object.assign({sym: DATA.symbols[0].symbol, tf: DATA.default_tf}, readHash());
function readHash(){ const o={}; location.hash.slice(1).split('&').forEach(p=>{const [k,v]=p.split('='); if(k&&v) o[k]=decodeURIComponent(v);}); return o; }
function writeHash(){ history.replaceState(null,'','#sym='+state.sym+'&tf='+state.tf); }
function snap(){ return DATA.symbols.find(s=>s.symbol===state.sym) || DATA.symbols[0]; }
function fmt(v, d){ return v==null ? '-' : Number(v).toFixed(d); }

function select(sym){ state.sym=sym; document.querySelectorAll('[data-sym]').forEach(el=>{
  el.classList.toggle('sel', el.dataset.sym===sym); }); writeHash(); draw(); }
function selectTf(tf){ state.tf=tf; document.querySelectorAll('.seg [data-tf]').forEach(b=>b.classList.toggle('sel', b.dataset.tf===tf)); writeHash(); draw(); }
document.querySelectorAll('.tick[data-sym], .watch-row[data-sym]').forEach(el=>el.addEventListener('click',()=>select(el.dataset.sym)));
document.querySelectorAll('.seg [data-tf]').forEach(b=>b.addEventListener('click',()=>selectTf(b.dataset.tf)));
let overlay = null;
document.querySelectorAll('.deal[data-entry]').forEach(el=>{
  el.addEventListener('mouseenter',()=>{ overlay={entry:+el.dataset.entry, sl:+el.dataset.sl, tp:+el.dataset.tp, label:el.dataset.label}; draw(); });
  el.addEventListener('mouseleave',()=>{ overlay=null; draw(); });
});

const NS='http:/'+'/www.w3.org/2000/svg';   // split so the page holds no web address
function el(tag, attrs, parent){ const e=document.createElementNS(NS, tag); for(const k in attrs) e.setAttribute(k, attrs[k]); if(parent) parent.appendChild(e); return e; }

function draw(){
  const s = snap(); const box = document.getElementById('chart'); const tip = document.getElementById('tip');
  const tfs = Object.keys(s.charts); if(!s.charts[state.tf]) state.tf = tfs[tfs.length>1?1:0];
  document.querySelectorAll('.seg [data-tf]').forEach(b=>b.classList.toggle('sel', b.dataset.tf===state.tf));
  const bars = s.charts[state.tf] || []; const sw = s.swings[state.tf] || [];
  box.querySelectorAll('svg').forEach(n=>n.remove());
  const W = box.clientWidth, H = box.clientHeight, padR = 74, padB = 22, padT = 8;
  const svg = el('svg', {viewBox:`0 0 ${W} ${H}`}, null); box.insertBefore(svg, tip);
  if(!bars.length) return;
  let lo = Math.min(...bars.map(b=>b[3])), hi = Math.max(...bars.map(b=>b[2]));
  const extra = [s.shown_price]; const plan = overlay || s.plan;
  if(plan) extra.push(plan.sl, plan.tp, plan.entry);
  extra.forEach(v=>{ if(v!=null){ lo=Math.min(lo,v); hi=Math.max(hi,v);} });
  const pad=(hi-lo)*0.06; lo-=pad; hi+=pad;
  const cw = (W-padR)/bars.length; const y = v => padT + (hi-v)/(hi-lo)*(H-padT-padB);
  const grid = css('--line'), ink3 = css('--ink-3'), up = css('--up'), down = css('--down'), zc = css('--zone');
  // grid + price axis
  for(let k=0;k<=6;k++){ const v = lo + (hi-lo)*k/6; const yy=y(v);
    el('line',{x1:0,x2:W-padR,y1:yy,y2:yy,stroke:grid,'stroke-width':1},svg);
    const t=el('text',{x:W-padR+6,y:yy+4,fill:ink3,'font-size':10,'font-family':'Consolas,monospace'},svg); t.textContent=fmt(v,s.decimals); }
  // time axis
  const step = Math.max(1, Math.floor(bars.length/7));
  for(let i=0;i<bars.length;i+=step){ const d=new Date(bars[i][0]*1000);
    const t=el('text',{x:i*cw+cw/2,y:H-6,fill:ink3,'font-size':10,'text-anchor':'middle'},svg);
    t.textContent = state.tf==='D1' ? d.toISOString().slice(5,10) : d.toISOString().slice(5,10)+' '+d.toISOString().slice(11,16); }
  // zones (a label too close to one already written is skipped; the band is still drawn)
  const labelled=[];
  s.zones.forEach(z=>{ if(!z.near||z.high<lo||z.low>hi) return; const v=z.status==='validated';
    el('rect',{x:0,width:W-padR,y:y(z.high),height:Math.max(1,y(z.low)-y(z.high)),fill:zc,'fill-opacity':v?0.16:0.06,
      stroke:v?zc:'none','stroke-opacity':0.35,'stroke-dasharray':v?'':'3 3'},svg);
    if(v && labelled.every(l=>Math.abs(l-y(z.high))>13)){ labelled.push(y(z.high)); const t=el('text',{x:4,y:y(z.high)-3,fill:zc,'font-size':10,'fill-opacity':0.85},svg);
      t.textContent=`${z.role} ${z.touches}x ${z.tf}${z.note?' · '+z.note:''}`; } });
  // candles
  bars.forEach((b,i)=>{ const [t,o,h,l,c]=b; const col = c>=o?up:down; const x=i*cw+cw/2;
    el('line',{x1:x,x2:x,y1:y(h),y2:y(l),stroke:col,'stroke-width':1},svg);
    const top=y(Math.max(o,c)), bot=y(Math.min(o,c));
    el('rect',{x:x-Math.max(1,cw*0.34),width:Math.max(1.5,cw*0.68),y:top,height:Math.max(1,bot-top),fill:col},svg); });
  // swings
  sw.forEach(([i,k,p])=>{ const x=i*cw+cw/2, yy=y(p)+(k==='high'?-6:6);
    el('path',{d:k==='high'?`M${x-3},${yy-3}L${x+3},${yy-3}L${x},${yy+1}Z`:`M${x-3},${yy+3}L${x+3},${yy+3}L${x},${yy-1}Z`,fill:ink3},svg); });
  // plan / hovered deal
  function hline(v,col,label,dash){ if(v==null) return; const yy=y(v);
    el('line',{x1:0,x2:W-padR,y1:yy,y2:yy,stroke:col,'stroke-width':1.2,'stroke-dasharray':dash||''},svg);
    el('rect',{x:W-padR,y:yy-8,width:padR,height:16,fill:col,rx:2},svg);
    const t=el('text',{x:W-padR+4,y:yy+4,fill:'#06090d','font-size':10,'font-weight':700,'font-family':'Consolas,monospace'},svg); t.textContent=fmt(v,s.decimals);
    if(label){ const u=el('text',{x:W-padR-6,y:yy-4,fill:col,'font-size':10,'text-anchor':'end'},svg); u.textContent=label; } }
  if(plan){ hline(plan.tp, up, (overlay?overlay.label+' ':'')+'TP', '6 4'); hline(plan.sl, down, 'SL', '6 4'); hline(plan.entry, css('--warn'), 'entry', '2 3'); }
  hline(s.shown_price, css('--accent'), s.live ? 'live bid' : 'last close');
  // crosshair + tooltip
  const cross = el('line',{y1:0,y2:H-padB,stroke:ink3,'stroke-dasharray':'2 3',visibility:'hidden'},svg);
  svg.addEventListener('mousemove',ev=>{ const r=svg.getBoundingClientRect(); const i=Math.floor((ev.clientX-r.left)/cw);
    if(i<0||i>=bars.length){ tip.style.display='none'; cross.setAttribute('visibility','hidden'); return; }
    const [t,o,h,l,c]=bars[i]; const x=i*cw+cw/2; cross.setAttribute('x1',x); cross.setAttribute('x2',x); cross.setAttribute('visibility','visible');
    tip.innerHTML = `<span class="muted">${new Date(t*1000).toISOString().slice(0,16).replace('T',' ')} UTC</span><br>`+
      `O <b>${fmt(o,s.decimals)}</b> H <b>${fmt(h,s.decimals)}</b> L <b>${fmt(l,s.decimals)}</b> C <b class="${c>=o?'up':'down'}">${fmt(c,s.decimals)}</b>`;
    tip.style.display='block'; tip.style.left=Math.min(ev.clientX-r.left+14, W-230)+'px'; tip.style.top='10px'; });
  svg.addEventListener('mouseleave',()=>{ tip.style.display='none'; cross.setAttribute('visibility','hidden'); });
}

// clocks: UTC, local, next H1 close
function pad2(n){ return String(n).padStart(2,'0'); }
function tickClock(){ const n=new Date();
  ['utc','utc2'].forEach(id=>{ const u=document.getElementById(id); if(u) u.textContent=pad2(n.getUTCHours())+':'+pad2(n.getUTCMinutes())+':'+pad2(n.getUTCSeconds()); });
  const l=document.getElementById('local'); if(l) l.textContent=n.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
  const left = 3600 - (n.getUTCMinutes()*60+n.getUTCSeconds());
  const h=document.getElementById('h1close'); if(h) h.textContent=pad2(Math.floor(left/60))+':'+pad2(left%60);
  const age=document.getElementById('age'); if(age){ const s=Math.max(0,Math.round((n-new Date(DATA.generated))/1000)); age.textContent = s<120 ? s+' s' : Math.round(s/60)+' min'; } }
setInterval(tickClock, 1000); tickClock();
// live mode: reload the page when the tool has rewritten it
if(DATA.refresh_seconds>0) setTimeout(()=>location.reload(), DATA.refresh_seconds*1000);
window.addEventListener('resize', draw);
select(state.sym); selectTf(state.tf);
})();
"""


def _trend_cell(s: dict | None) -> str:
    if s is None:
        return '<td class="c muted">&ndash;</td>'
    warn = ' <span title="structure broken">&#9888;</span>' if s["broken"] else ""
    return (f'<td class="c"><span class="trend {s["trend"]}" title="{esc(s["labels"])}">'
            f'{TREND_ICON[s["trend"]]} {TREND_WORD[s["trend"]]}</span>{warn}</td>')


def _change(snap: dict) -> str:
    ch = snap["day_change"]
    if ch is None:
        return ""
    cls = "up" if ch > 0 else "down" if ch < 0 else "muted"
    pct = ch / snap["prev_day_close"] * 100 if snap["prev_day_close"] else 0
    return f'<span class="num {cls}">{ch:+.{snap["decimals"]}f} ({pct:+.2f}%)</span>'


def _p(snap: dict, v) -> str:
    return "&ndash;" if v is None else f'{v:.{snap["decimals"]}f}'


def _ticker(snaps: list[dict]) -> str:
    out = ""
    for s in snaps:
        cls = VERDICT_CLASS[s["verdict"]]
        out += (f'<button class="tick" data-sym="{esc(s["symbol"])}"><span class="sym">{esc(s["symbol"])}</span>'
                f'<span class="px num">{_p(s, s["shown_price"])}</span>{_change(s)}'
                f'<span class="si {cls}" title="{esc(s["verdict"])}">{STATE_ICON[cls]}</span>'
                f'<span class="sw">{esc(s["verdict"].lower())}</span></button>')
    return out


def _watch_panel(snaps: list[dict], source: str) -> str:
    rows = ""
    for s in snaps:
        live = "live bid" if s["live"] else "last H1 close"
        rows += (f'<tr class="watch-row" data-sym="{esc(s["symbol"])}" style="cursor:pointer">'
                 f'<td><b>{esc(s["symbol"])}</b><div class="detail">{live}</div></td>'
                 f'<td class="r num">{_p(s, s["shown_price"])}<div>{_change(s)}</div></td></tr>')
    return (f'<div class="panel"><h3>01 / Market watch <span class="hint">{esc(source)}</span></h3>'
            f'<table>{rows}</table><div class="detail" style="margin-top:6px">Change = against '
            'the last closed D1 candle.</div></div>')


def _mtf_panel(snaps: list[dict]) -> str:
    rows = "".join(
        f'<tr><td><b>{esc(s["symbol"])}</b></td>'
        + "".join(_trend_cell(s["structure"].get(tf)) for tf in ("D1", "H4", "H1")) + "</tr>"
        for s in snaps)
    return ('<div class="panel"><h3>02 / Multi-timeframe direction</h3><table class="mtf">'
            '<tr><th>Symbol</th><th class="c">D1</th><th class="c">H4</th><th class="c">H1</th></tr>'
            f'{rows}</table><div class="detail" style="margin-top:6px">Last two confirmed swings: '
            'HH+HL = up, LH+LL = down, else range. &#9888; = structure broken. D1/H4 give the '
            'direction, H1 confirms.</div></div>')


def _session_panel(sess: dict) -> str:
    cls = "on" if sess["market"] == "open" else "off"
    best = ' style="color:var(--up)"' if sess["overlap"] else ""
    return ('<div class="panel"><h3>03 / Session</h3>'
            f'<div class="pill {cls}"><span class="dot"></span>{esc(sess["text"])}</div>'
            f'<div class="detail" style="margin:8px 0 6px"{best}>{esc(sess["next"])} '
            f'&middot; best hours {esc(sess["window"])}</div>'
            '<table><tr><td>Next H1 close in</td><td class="r num" id="h1close">--:--</td></tr>'
            '<tr><td>UTC</td><td class="r num" id="utc2">--:--:--</td></tr>'
            '<tr><td>Your time</td><td class="r num" id="local">--:--</td></tr></table>'
            '<div class="detail" style="margin-top:6px">Analysis updates when an H1 candle '
            'closes. Never on a forming candle.</div></div>')


def _manual_panel(cfg: dict) -> str:
    items = "".join(f'<li><span class="si you">?</span><span>{esc(t)}</span><span class="sw">your call</span></li>'
                    for t in manual_checks(cfg))
    return ('<div class="panel"><h3>Your part of the checklist</h3>'
            f'<ul class="steps">{items}</ul><div class="detail" style="margin-top:6px">'
            'The tool cannot see your account or your head. Every box must be yes.</div></div>')


def _engine_panel(s: dict) -> str:
    cls = VERDICT_CLASS[s["verdict"]]
    # The direction is shown only for a setup that passed: a refused buy is not a buy.
    direction = (f' &middot; {esc(s["direction"].upper())}'
                 if s["direction"] and s["verdict"] == "SETUP CANDIDATE" else "")
    steps = "".join(
        f'<li><span class="si {c["state"]}">{STATE_ICON[c["state"]]}</span>'
        f'<span>{esc(c["step"])}<div class="detail">{esc(c["detail"])}</div></span>'
        f'<span class="sw">{STATE_WORD[c["state"]]}</span></li>' for c in s["checklist"])
    plan = ""
    if s["plan"]:
        p = s["plan"]
        plan = (f'<div class="levels-grid num"><div><span class="lab">Entry</span>{_p(s, p["entry"])}</div>'
                f'<div><span class="lab">Stop</span>{_p(s, p["sl"])}</div>'
                f'<div><span class="lab">Target</span>{_p(s, p["tp"])}</div>'
                f'<div><span class="lab">R:R</span>1:{p["rr"]:.1f}</div></div>')
    reasons = "".join(f"<li>{esc(r)}</li>" for r in s["reasons"])
    return (f'<div class="panel"><h3>Rulebook engine <span class="hint">as of H1 close '
            f'{s["as_of"]:%a %H:%M} UTC</span></h3>'
            f'<div class="verdict"><span class="vbadge {cls}">{STATE_ICON[cls]} {esc(s["verdict"])}{direction}</span></div>'
            f'<ul class="steps">{steps}</ul>{plan}<ul class="reasons">{reasons}</ul>'
            '<div class="detail" style="margin-top:6px">Rule checks, not a forecast. No '
            'percentages: the backtest found no proven edge.</div></div>')


def _bar(value: float | None, scale: float = 0.5) -> str:
    """A small bar centred on 0R: right/green = made money in the test, left/red = lost."""
    if value is None:
        return '<span class="bar0"></span>'
    w = min(abs(value) / scale, 1) * 50
    if value >= 0:
        style = f"left:50%;width:{w:.0f}%;background:var(--up)"
    else:
        style = f"right:50%;width:{w:.0f}%;background:var(--down)"
    return f'<span class="bar0"><span style="{style}"></span></span>'


def _deals_panel(s: dict, cfg: dict) -> str:
    lot = cfg["risk"]["fixed_lot"]
    body = ""
    for row in s["deals"]:
        d, rec = row["deal"], row["record"]
        if d is None:
            body += f'<div class="deal none"><b>{esc(row["strategy"])}</b> &middot; no deal</div>'
            continue
        risk = "unknown" if d["risk"] is None else f'${d["risk"]:.2f}'
        weak = ('<span class="tag" style="color:var(--warn);border-color:var(--warn)">weak record</span>'
                if rec["weak"] else '<span class="tag val">positive record</span>')
        new = (f'newest part {rec["new_trades"]} trades {rec["new_avg_r"]:+.2f}R'
               if rec["new_trades"] else "too few past trades")
        body += (f'<div class="deal" data-entry="{d["entry"]}" data-sl="{d["sl"]}" data-tp="{d["tp"]}" '
                 f'data-label="{esc(row["strategy"])}"><div class="t">{esc(d["order"])} '
                 f'{"about" if d["kind"] == "market" else "at"} <span class="num">{_p(s, d["entry"])}</span> '
                 f'<span class="muted">&middot; {esc(row["strategy"])}</span></div>'
                 f'<div class="num detail">SL {_p(s, d["sl"])} &middot; TP {_p(s, d["tp"])} &middot; '
                 f'R:R 1:{d["rr"]:.1f} &middot; risk at {lot} lot {risk}</div>'
                 f'<div class="detail">{esc(row["why"])}{" [" + esc(d["pattern"]) + "]" if d["pattern"] else ""}</div>'
                 f'<div class="rec">{_bar(rec["new_avg_r"])}<span>{esc(new)}</span>{weak}</div></div>')
    return (f'<div class="panel"><h3>Strategy deals <span class="hint">hover = draw on chart</span></h3>{body}'
            '<div class="detail">Mechanical signals, not predictions. Demo only, one trade at a '
            'time, always with the stop.</div></div>')


def _record_panel(s: dict) -> str:
    rows = ""
    for row in s["deals"]:
        rec = row["record"]
        if rec is None:
            rows += (f'<tr><td><b>{esc(row["strategy"])}</b></td>'
                     '<td colspan="3" class="muted">record shown when the strategy has a deal</td></tr>')
            continue
        avg = "&ndash;" if rec["avg_r"] is None else f'{rec["avg_r"]:+.2f}R'
        new = "&ndash;" if rec["new_avg_r"] is None else f'{rec["new_avg_r"]:+.2f}R'
        rows += (f'<tr><td><b>{esc(row["strategy"])}</b></td><td class="r num">{rec["trades"]}</td>'
                 f'<td class="r num">{avg}</td><td class="r num">{new} ({rec["new_trades"]})</td>'
                 f'<td>{_bar(rec["new_avg_r"])}</td></tr>')
    return rows


def _backtest_panel(s: dict, records: dict | None) -> str:
    rows = ""
    if records:
        for name in STRATEGIES:
            rec = records.get(name)
            if rec is None or not rec["trades"]:
                rows += f'<tr><td><b>{esc(name)}</b></td><td colspan="4" class="muted">no trades</td></tr>'
                continue
            new = "&ndash;" if not rec["new_trades"] else f'{rec["new_avg_r"]:+.2f}R ({rec["new_trades"]})'
            rows += (f'<tr><td><b>{esc(name)}</b></td><td class="r num">{rec["trades"]}</td>'
                     f'<td class="r num">{rec["win_rate"] * 100:.0f}%</td>'
                     f'<td class="r num">{rec["avg_r"]:+.2f}R</td><td class="r num">{new}</td>'
                     f'<td>{_bar(rec["new_avg_r"] if rec["new_trades"] else None)}</td></tr>')
    else:
        rows = _record_panel(s)
    return ('<div class="panel"><h3>Backtest record <span class="hint">with spread, 1R = amount risked</span></h3>'
            '<table><tr><th>Strategy</th><th class="r">Trades</th><th class="r">Win</th>'
            '<th class="r">Avg</th><th class="r">Newest part (unseen)</th><th></th></tr>'
            f'{rows}</table><div class="detail" style="margin-top:6px">Only the newest part '
            'says anything about the future, and it is one small sample. Bar: green = made '
            'money in the test, red = lost.</div></div>')


def _location_panel(s: dict, cfg: dict) -> str:
    pct = s["range_pct"]
    half = cfg["strategy"]["middle_of_range_pct"] / 2
    where = {"inside_zone": "inside a zone", "middle": "MIDDLE of the range: no trade",
             "at_support": "at support", "at_resistance": "at resistance",
             "between": "between levels", "no_range": "no measurable range"}[s["location"]]
    gauge = ""
    if pct is not None and s["support"] and s["resistance"]:
        mark = min(max(pct, 0), 100)
        gauge = (f'<div class="gauge"><div class="mid" style="left:{50 - half:.0f}%;width:{2 * half:.0f}%"></div>'
                 f'<div class="mark" style="left:calc({mark:.1f}% - 1px)"></div></div>'
                 f'<div class="gauge-labels num"><span>support {_p(s, s["support"][1])}</span>'
                 f'<span>{pct:.0f}%</span><span>resistance {_p(s, s["resistance"][0])}</span></div>')
    # level ladder: nearest zones above and below price
    each = cfg["report"]["levels_each_side"]
    above = sorted([z for z in s["zones"] if z["low"] > s["price"]], key=lambda z: z["low"])[:each]
    below = sorted([z for z in s["zones"] if z["high"] < s["price"]], key=lambda z: -z["high"])[:each]
    inside = [z for z in s["zones"] if z["inside"]]

    def row(z):
        tag = "val" if z["status"] == "validated" else ""
        dist = "" if z["dist_atr"] is None else f'{z["dist_atr"]:.1f}'
        return (f'<tr class="{"in" if z["inside"] else ""}"><td class="num">{_p(s, z["low"])} &ndash; {_p(s, z["high"])}</td>'
                f'<td>{esc(z["role"])}</td><td><span class="tag {tag}">{esc(z["status"])} {z["touches"]}x</span></td>'
                f'<td>{esc(z["tf"])}<div class="detail">{esc(z["note"])}</div></td>'
                f'<td class="r num">{dist}</td></tr>')
    ladder = "".join(row(z) for z in reversed(above)) + "".join(row(z) for z in inside)
    ladder += f'<tr class="price"><td class="num">&#9654; {_p(s, s["price"])}</td><td colspan="4">last H1 close</td></tr>'
    ladder += "".join(row(z) for z in below)
    return (f'<div class="panel"><h3>Where is price? <span class="hint">{esc(where)}</span></h3>{gauge}'
            '<table class="ladder" style="margin-top:10px"><tr><th>Zone</th><th>Role</th><th>Status</th>'
            f'<th>Seen on</th><th class="r">ATR away</th></tr>{ladder}</table>'
            f'<div class="detail" style="margin-top:6px">Striped band = middle '
            f'{cfg["strategy"]["middle_of_range_pct"]}% of the range (no trade). Zones from D1 '
            'and H4 wick tips; 2+ touches = validated.</div></div>')


def _feed_panel(s: dict) -> str:
    items = "".join(
        f'<li class="{e["tone"]}"><span class="when">{e["time"]:%a %d %b %H:%M}</span>'
        f'<span class="k">{esc(e["kind"])}</span><div>{esc(e["text"])}</div></li>'
        for e in s["events"]) or '<li class="neutral">Nothing notable on the recent H1 candles.</li>'
    return ('<div class="panel"><h3>Event feed <span class="hint">H1 closes, point in time</span></h3>'
            f'<ul class="feed">{items}</ul></div>')


def _news_panel(s: dict) -> str:
    if s["news"] is None:
        return ""
    items = "".join(f"<li>{esc(n)}</li>" for n in s["news"]) or \
        "<li class='muted'>No high-impact event in the next 7 days for this symbol.</li>"
    return (f'<div class="panel"><h3>High-impact news <span class="hint">next 7 days</span></h3>'
            f'<ul class="reasons news">{items}</ul></div>')


def _chart_panel(cfg: dict) -> str:
    tfs = "".join(f'<button data-tf="{tf}">{tf}</button>' for tf in cfg["desk"]["chart_timeframes"])
    return ('<div class="panel"><h3>Chart <span class="hint">candles + D1/H4 zones + plan</span></h3>'
            f'<div class="chart-head"><span class="seg">{tfs}</span>'
            '<span class="detail">Hover a candle for OHLC; hover a deal to draw its SL/TP.</span></div>'
            '<div class="chart" id="chart"><div class="tip" id="tip"></div></div>'
            '<div class="legend"><span><i style="background:var(--zone);opacity:.5"></i>validated zone</span>'
            '<span><i style="border:1px dashed var(--zone)"></i>zone to watch</span>'
            '<span><i style="background:var(--up)"></i>up candle / TP</span>'
            '<span><i style="background:var(--down)"></i>down candle / SL</span>'
            '<span><i style="background:var(--warn)"></i>entry</span>'
            '<span>&#9650;&#9660; confirmed swings</span></div></div>')


def _json_default(o):
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    raise TypeError(type(o))


def build_page(cfg: dict, source: str, snaps: list[dict], generated: datetime,
               source_note: str | None, refresh_seconds: int,
               records: dict[str, dict] | None = None) -> str:
    """The whole page. `records` = {symbol: {strategy: backtest summary}} (optional)."""
    sess = session_status(cfg, generated)
    records = records or {}
    center = right = ""
    for s in snaps:
        sym = esc(s["symbol"])
        center += (f'<div class="sym-view" data-sym="{sym}">{_location_panel(s, cfg)}'
                   f'{_backtest_panel(s, records.get(s["symbol"]))}</div>')
        right += (f'<div class="sym-view" data-sym="{sym}">{_engine_panel(s)}{_deals_panel(s, cfg)}'
                  f'{_feed_panel(s)}{_news_panel(s)}</div>')
    data = {"symbols": [{k: s[k] for k in ("symbol", "decimals", "shown_price", "live", "plan",
                                           "zones", "charts", "swings")} for s in snaps],
            "default_tf": cfg["desk"]["default_timeframe"], "generated": generated,
            "refresh_seconds": refresh_seconds}
    payload = json.dumps(data, default=_json_default).replace("</", "<\\/")
    feed = "MT5 terminal (your broker's prices)" if source == "mt5" else f"public feed ({source})"
    live = (f'<span class="pill on"><span class="dot"></span>Live &middot; every {refresh_seconds} s</span>'
            if refresh_seconds else '<span class="pill"><span class="dot"></span>Snapshot</span>')
    note = f' &middot; {esc(source_note)}' if source_note else ""
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Market Analyst desk</title><style>{CSS}</style></head>
<body><div class="app">
<header class="bar">
  <div class="brand">MARKET ANALYST<small>PERSONAL ANALYSIS DESK</small></div>
  {live}
  <span class="pill ro">Read-only &middot; never places orders</span>
  <span class="pill">Data: {esc(feed)}</span>
  <span class="spacer"></span>
  <div class="clock">updated {generated:%H:%M:%S} UTC (<span id="age">0 s</span> ago)<br>
    <b class="num" id="utc">--:--:--</b> UTC</div>
</header>
<nav class="ticker">{_ticker(snaps)}</nav>
<div class="honest"><b>No proven edge.</b> The backtest found no strategy that makes money on
unseen history (RESEARCH.md). Everything here is a rule check to support YOUR decision, not a
prediction. Demo account, {cfg["risk"]["fixed_lot"]} lot, always with the stop loss.{note}</div>
<main>
  <div class="col left">{_watch_panel(snaps, source)}{_mtf_panel(snaps)}
  {_session_panel(sess)}{_manual_panel(cfg)}</div>
  <div class="col center">{_chart_panel(cfg)}{center}</div>
  <div class="col right">{right}</div>
</main>
<footer><span>Built on your computer by market-analyst. Nothing is sent anywhere; no AI service is used.</span>
<span>Run <b>market-analyst desk --watch</b> to keep it live.</span></footer>
</div>
<script type="application/json" id="desk-data">{payload}</script>
<script>{JS}</script>
</body></html>"""


# =============================================================================
# 6. Helpers used by the `desk` command
# =============================================================================

def backtest_records(h: History, cfg: dict) -> dict[str, dict]:
    """Backtest summary per strategy on one symbol: all history, and the
    newest part that the research never used to choose settings."""
    from . import backtest, research
    cut = research.split_time(h)
    out = {}
    for name in STRATEGIES:
        spec = cfg.get("filters", {}).get(name) or {}
        trades = research.run_all({h.symbol: h}, name, cfg, spec)
        _, unseen = research.split(trades, {h.symbol: cut})
        a, b = backtest.summarize(trades), backtest.summarize(unseen)
        out[name] = {"trades": a["trades"], "win_rate": a["win_rate"], "avg_r": a["expectancy_r"],
                     "new_trades": b["trades"], "new_avg_r": b["expectancy_r"]}
    return out


def read_quote(mt5, symbol: str) -> dict | None:
    """Current bid / ask from the MT5 terminal (read-only, like the rest).
    Shown on the page only: the analysis never uses a price that is not a
    closed candle."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None or not tick.bid:
        return None
    return {"bid": float(tick.bid), "ask": float(tick.ask),
            "spread": float(tick.ask) - float(tick.bid)}


def with_live(snap: dict, live: dict | None) -> dict:
    """The same snapshot with a new live quote (between two H1 closes only
    the price moves; the analysis stays the same)."""
    out = dict(snap)
    out["live"] = live
    out["shown_price"] = live["bid"] if live else snap["price"]
    prev = snap["prev_day_close"]
    out["day_change"] = None if prev is None else out["shown_price"] - prev
    return out
