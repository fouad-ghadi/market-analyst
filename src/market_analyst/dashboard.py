"""A web page that shows everything at once: verdict, deals, charts, levels,
news and the backtest record, for every symbol.

It is one self-contained HTML file (charts are embedded as images), built
by this tool on your computer. Nothing is sent anywhere and no AI service is
used. Open it in any browser.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path

import pandas as pd

from . import backtest, report, research
from .analysis import Analysis
from .deals import STRATEGY_TEXT, Deal
from .levels import clear_reactions
from .strategies import STRATEGIES
from .structure import classify
from .swings import find_swings

# Status colours never carry meaning alone: each badge also has an icon and a word.
VERDICT_STYLE = {
    "NO TRADE": ("critical", "&#10005;", "No trade"),
    "WAIT": ("warning", "&#9208;", "Wait"),
    "SETUP CANDIDATE": ("good", "&#10003;", "Setup candidate"),
}
TREND_TEXT = {"up": "Uptrend", "down": "Downtrend", "range": "Range", "unknown": "Not enough swings"}

CSS = """
.viz-root {
  color-scheme: light;
  --surface-0: #f4f3f0; --surface-1: #fcfcfb; --border: #e4e3df;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #6f6e69;
  --good: #0ca30c; --warning: #fab219; --critical: #d03b3b; --accent: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  .viz-root {
    color-scheme: dark;
    --surface-0: #121211; --surface-1: #1a1a19; --border: #33332f;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #9a998f;
    --accent: #3987e5;
  }
}
* { box-sizing: border-box; }
body { margin: 0; }
.viz-root { background: var(--surface-0); color: var(--text-primary);
  font: 15px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  min-height: 100vh; padding: 24px; }
.wrap { max-width: 1240px; margin: 0 auto; }
header.top { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 24px;
  margin-bottom: 8px; }
h1 { font-size: 22px; margin: 0; }
h2 { font-size: 19px; margin: 0; }
h3 { font-size: 13px; margin: 0 0 10px; text-transform: uppercase; letter-spacing: .06em;
  color: var(--text-secondary); font-weight: 600; }
.muted { color: var(--text-muted); }
.small { font-size: 13px; }
nav a { color: var(--accent); margin-right: 16px; text-decoration: none; font-weight: 600; }
.notice { border: 1px solid var(--border); border-left: 4px solid var(--warning);
  background: var(--surface-1); border-radius: 8px; padding: 10px 14px; margin: 14px 0 22px;
  color: var(--text-secondary); font-size: 14px; }
section.symbol { margin: 26px 0 40px; }
.symbol-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 18px;
  margin-bottom: 14px; }
.price { font-size: 26px; font-weight: 700; font-variant-numeric: tabular-nums; }
.grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
  padding: 16px 18px; }
.card.wide { grid-column: 1 / -1; }
.card.span2 { grid-column: span 2; }
@media (max-width: 760px) { .card.span2 { grid-column: auto; } }
td.nowrap { white-space: nowrap; }
.badge { display: inline-flex; align-items: center; gap: 8px; font-weight: 700; font-size: 18px; }
.badge .icon { display: inline-grid; place-items: center; width: 28px; height: 28px;
  border-radius: 50%; color: #fff; font-size: 15px; }
.badge.good .icon { background: var(--good); }
.badge.warning .icon { background: var(--warning); color: #0b0b0b; }
.badge.critical .icon { background: var(--critical); }
ul.reasons { margin: 10px 0 0; padding-left: 18px; color: var(--text-secondary); }
ul.reasons li { margin: 4px 0; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 14px; }
th { text-align: left; color: var(--text-muted); font-weight: 600; font-size: 12px;
  text-transform: uppercase; letter-spacing: .04em; padding: 6px 8px;
  border-bottom: 1px solid var(--border); }
td { padding: 7px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }
td.num, th.num { text-align: right; }
tr.here td { font-weight: 700; }
.deal { border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; margin-bottom: 10px; }
.deal-title { font-weight: 700; font-size: 16px; }
.deal-levels { display: grid; grid-template-columns: repeat(4, auto); gap: 4px 18px;
  margin: 8px 0; font-variant-numeric: tabular-nums; justify-content: start; }
.deal-levels span.label { color: var(--text-muted); font-size: 12px; display: block; }
.tag { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; font-weight: 600;
  color: var(--text-secondary); }
.tag .dot { width: 9px; height: 9px; border-radius: 50%; }
.tag.weak .dot { background: var(--warning); }
.tag.ok .dot { background: var(--good); }
.none { color: var(--text-muted); padding: 2px 0; }
img.chart { width: 100%; height: auto; border-radius: 6px; border: 1px solid var(--border);
  background: #fcfcfb; }
details summary { cursor: pointer; color: var(--accent); font-weight: 600; margin-top: 8px; }
.pos { color: var(--text-primary); }
footer { color: var(--text-muted); font-size: 13px; margin-top: 30px; }
"""


def esc(value) -> str:
    return html.escape(str(value))


def image_tag(path: Path, alt: str) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img class="chart" alt="{esc(alt)}" src="data:image/png;base64,{data}">'


def verdict_card(a: Analysis, cfg: dict, f) -> str:
    style, icon, label = VERDICT_STYLE[a.verdict]
    direction = f" &middot; {a.direction.upper()}" if a.direction else ""
    reasons = "".join(f"<li>{esc(r)}</li>" for r in a.reasons)
    plan = ""
    if a.plan is not None:
        p = a.plan
        plan = (f'<div class="deal-levels" style="margin-top:12px">'
                f'<div><span class="label">Entry</span>{f(p.entry)}</div>'
                f'<div><span class="label">Stop loss</span>{f(p.sl)}</div>'
                f'<div><span class="label">Take profit</span>{f(p.tp)}</div>'
                f'<div><span class="label">R:R</span>1:{p.rr:.1f}</div></div>'
                '<div class="small muted">A plan to check against your checklist, '
                'not a prediction.</div>')
    return (f'<div class="card"><h3>Rulebook verdict</h3>'
            f'<div class="badge {style}"><span class="icon" aria-hidden="true">{icon}</span>'
            f'{label}{direction}</div>'
            f'<ul class="reasons">{reasons}</ul>{plan}</div>')


def structure_card(a: Analysis) -> str:
    rows = ""
    for tf, role in (("D1", "direction"), ("H4", "direction"), ("H1", "confirmation")):
        s = a.structures.get(tf)
        if s is None:
            continue
        note = f'<div class="small muted">{esc(s.note)}</div>' if s.note else ""
        rows += (f"<tr><td><b>{tf}</b> <span class='muted small'>{role}</span></td>"
                 f"<td>{TREND_TEXT[s.trend]}</td><td>{s.highs_label} + {s.lows_label}{note}</td></tr>")
    return (f'<div class="card"><h3>Market structure</h3><table>'
            f"<tr><th>Timeframe</th><th>Trend</th><th>Swings</th></tr>{rows}</table></div>")


def levels_card(a: Analysis, cfg: dict, f) -> str:
    shown = report.select_zones(a.zones, a.price, cfg["report"]["levels_each_side"])
    rows, price_done = "", False
    for z in shown:
        if not price_done and z.high < a.price:
            rows += (f'<tr class="here"><td colspan="5">&#9654; price {f(a.price)}</td></tr>')
            price_done = True
        inside = z.low <= a.price <= z.high
        tfs = "+".join([z.timeframe] + z.confirmed_by)
        note = ("role reversal" if z.role_reversal else
                "broken, not retested" if z.untested_flip else "")
        rows += (f'<tr class="{"here" if inside else ""}"><td class="nowrap">{f(z.low)} &ndash; {f(z.high)}'
                 f'{" &#9664; price" if inside else ""}</td>'
                 f"<td>{z.role}</td><td>{z.status}</td><td class='num'>{z.touches}</td>"
                 f"<td>{tfs} <span class='muted small'>{note}</span></td></tr>")
    if not price_done and not any(z.low <= a.price <= z.high for z in shown):
        rows += f'<tr class="here"><td colspan="5">&#9654; price {f(a.price)}</td></tr>'
    return (f'<div class="card span2"><h3>Key levels (D1 and H4 zones nearest to price)</h3><table>'
            "<tr><th>Zone</th><th>Role</th><th>Status</th><th class='num'>Touches</th>"
            f"<th>Seen on</th></tr>{rows}</table></div>")


def deals_card(deals: dict[str, Deal | None], cfg: dict, f) -> str:
    lot = cfg["risk"]["fixed_lot"]
    body = ""
    for name in STRATEGIES:
        d = deals.get(name)
        if d is None:
            body += f'<div class="none"><b>{name}</b> &middot; no deal</div>'
            continue
        tag = ('<span class="tag weak"><span class="dot"></span>Weak record</span>'
               if d.record.weak else
               '<span class="tag ok"><span class="dot"></span>Positive record</span>')
        where = "about" if d.kind == "market" else "at"
        pending = ("" if d.kind == "market" else
                   '<div class="small muted">Pending order: valid for the next H1 candle only.</div>')
        body += (f'<div class="deal"><div class="deal-title">{d.order_text} {where} {f(d.entry)}'
                 f' <span class="muted small">&middot; {name}</span></div>'
                 f'<div class="deal-levels">'
                 f'<div><span class="label">Stop loss</span>{f(d.sl)}</div>'
                 f'<div><span class="label">Take profit</span>{f(d.tp)}</div>'
                 f'<div><span class="label">R:R</span>1:{d.rr:.1f}</div>'
                 f'<div><span class="label">Risk at {lot} lot</span>{d.risk_text}</div>'
                 f'</div><div class="small">{esc(STRATEGY_TEXT[name])}'
                 f'{" [" + esc(d.pattern) + "]" if d.pattern else ""}</div>'
                 f'<div class="small" style="margin-top:4px">{tag} '
                 f'<span class="muted">{esc(d.record.text)}</span></div>{pending}</div>')
    return (f'<div class="card"><h3>Deals right now</h3>{body}'
            '<div class="small muted" style="margin-top:8px">Mechanical signals, not predictions. '
            'Demo only, one trade at a time, always with the stop loss.</div></div>')


def news_card(lines: list[str] | None) -> str:
    if lines is None:
        return ""
    items = "".join(f"<li>{esc(line)}</li>" for line in lines) or "<li>None scheduled (the calendar only covers the current week)</li>"
    return (f'<div class="card"><h3>High-impact news, next 7 days</h3>'
            f'<ul class="reasons" style="margin-top:0">{items}</ul></div>')


def backtest_card(h: backtest.History, cfg: dict) -> str:
    cut = research.split_time(h)
    rows = ""
    for name in STRATEGIES:
        spec = cfg.get("filters", {}).get(name) or {}
        trades = research.run_all({h.symbol: h}, name, cfg, spec)
        _, unseen = research.split(trades, {h.symbol: cut})
        a, b = backtest.summarize(trades), backtest.summarize(unseen)
        if not a["trades"]:
            rows += f"<tr><td>{name}</td><td colspan='5' class='muted'>no trades</td></tr>"
            continue
        newest = f"{b['expectancy_r']:+.2f}R ({b['trades']})" if b["trades"] else "&ndash;"
        rows += (f"<tr><td><b>{name}</b></td><td class='num'>{a['trades']}</td>"
                 f"<td class='num'>{a['win_rate'] * 100:.0f}%</td>"
                 f"<td class='num'>{a['expectancy_r']:+.2f}R</td>"
                 f"<td class='num'>{a['total_r']:+.1f}R</td><td class='num'>{newest}</td></tr>")
    return (f'<div class="card wide"><h3>Backtest record on {esc(h.symbol)} '
            f'({h.time[h.first_bar]:%Y-%m-%d} to {h.time[-1]:%Y-%m-%d}, spread {h.spread:g})</h3>'
            "<table><tr><th>Strategy</th><th class='num'>Trades</th><th class='num'>Win rate</th>"
            "<th class='num'>Avg per trade</th><th class='num'>Total</th>"
            "<th class='num'>Newest part (unseen)</th></tr>"
            f"{rows}</table><div class='small muted' style='margin-top:8px'>1R = the amount "
            "risked. Above 0 made money in the test. Only the newest part says anything about "
            "the future, and even it is one small sample.</div></div>")


def chart_card(df: pd.DataFrame, zones, cfg: dict, symbol: str, tf: str, source: str,
               folder: Path) -> str:
    """Candles of `tf` with the analysis zones (D1 + H4) drawn on top, so the
    chart, the levels table and the verdict always show the same levels."""
    swings = find_swings(df, cfg["swings"]["fractal_n"])
    trend = classify(df, cfg, tf, swings).summary
    path = folder / f"{symbol}_{tf}_dashboard.png"
    report.draw_chart(df, zones, clear_reactions(df, swings, cfg), cfg, symbol, tf, source,
                      trend, path)
    return (f'<div class="card wide"><h3>{tf} candles with the D1/H4 zones</h3>'
            f'{image_tag(path, f"{symbol} {tf} chart")}</div>')


def build(cfg: dict, source: str, data: list[dict], generated: pd.Timestamp,
          source_note: str | None) -> str:
    """`data` = one dict per symbol: analysis, deals, history, frames, news."""
    folder = Path(cfg["_out_dir"])
    nav = "".join(f'<a href="#{esc(d["symbol"])}">{esc(d["symbol"])}</a>' for d in data)
    sections = ""
    for d in data:
        sym, a = d["symbol"], d["analysis"]
        f = lambda v, s=sym: report.fmt_price(v, cfg, s)  # noqa: E731
        charts = "".join(chart_card(d["frames"][tf], a.zones, cfg, sym, tf, source, folder)
                         for tf in ("H4", "H1") if tf in d["frames"])
        record = backtest_card(d["history"], cfg) if d["history"] is not None else ""
        sections += (
            f'<section class="symbol" id="{esc(sym)}"><div class="symbol-head">'
            f'<h2>{esc(sym)}</h2><span class="price">{f(a.price)}</span>'
            f'<span class="muted small">as of the H1 candle that closed '
            f'{a.as_of + pd.Timedelta(hours=1):%a %d %b %H:%M} UTC</span></div>'
            f'<div class="grid">'
            f'{verdict_card(a, cfg, f)}{deals_card(d["deals"], cfg, f)}'
            f'{structure_card(a)}{levels_card(a, cfg, f)}{news_card(d["news"])}'
            f'{charts}{record}</div></section>')
    note = f'<div class="notice">{esc(source_note)}</div>' if source_note else ""
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Market Analyst dashboard</title><style>{CSS}</style></head>
<body><div class="viz-root"><div class="wrap">
<header class="top"><h1>Market Analyst</h1>
<span class="muted small">generated {generated:%Y-%m-%d %H:%M} UTC &middot; data: {esc(source)}</span>
<nav>{nav}</nav></header>
<div class="notice">Decision support only. This page never places orders, and nothing on it is a
prediction: the backtest has not shown a proven edge (see RESEARCH.md). Demo account, 0.01 lot,
always with the stop loss. Re-run <b>market-analyst dashboard</b> after each H1 close to refresh.</div>
{note}{sections}
<footer>Built locally by market-analyst. Charts: blue = candle closed up, orange = closed down,
green band = validated zone, light band = zone to watch.</footer>
</div></div></body></html>"""
