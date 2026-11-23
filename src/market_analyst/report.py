"""Turn an analysis into something you can read: text for the terminal, and a
chart image with the zones drawn so you can compare with your own lines.

Chart colours (checked for colour-blind safety):
    blue candle = closed up, orange candle = closed down, aqua band = zone.
Zones are also labelled with text, so colour is never the only clue.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .analysis import Analysis
from .levels import Zone

UP, DOWN, ZONE = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK_SOFT, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

LOCATION_TEXT = {
    "inside_zone": "INSIDE a validated zone",
    "at_support": "AT validated support",
    "at_resistance": "AT validated resistance",
    "middle": "in the MIDDLE of the range",
    "between": "between levels (not at a level)",
    "no_range": "outside a measurable range",
}


def fmt_price(value: float, cfg: dict, symbol: str) -> str:
    return f"{value:.{cfg['symbols'][symbol].get('price_decimals', 5)}f}"


def zone_line(z: Zone, cfg: dict, symbol: str, price: float | None = None) -> str:
    """One text line describing a zone."""
    f = lambda v: fmt_price(v, cfg, symbol)  # noqa: E731
    tfs = "+".join([z.timeframe] + z.confirmed_by)
    parts = [
        f"{f(z.low)} - {f(z.high)}",
        f"{z.role:10s}",
        f"{z.status:9s}",
        f"{z.touches} touch{'es' if z.touches != 1 else ''} "
        f"({z.touch_highs} high, {z.touch_lows} low)",
        f"{tfs:5s}",
        f"score {z.score:.1f}",
        f"last touch {z.last_touch:%Y-%m-%d}",
    ]
    if z.role_reversal:
        parts.append("role reversal")
    elif z.untested_flip:
        was = "resistance" if z.role == "support" else "support"
        parts.append(f"broken {was}, not retested yet")
    if price is not None and z.low <= price <= z.high:
        parts.append("<-- price is here")
    return "  ".join(parts)


def select_zones(zones: list[Zone], price: float, each_side: int) -> list[Zone]:
    """The zones nearest to price: `each_side` above, `each_side` below, plus any
    zone price is inside. Returned from the highest price to the lowest."""
    above = sorted((z for z in zones if z.low > price), key=lambda z: z.low)[:each_side]
    below = sorted((z for z in zones if z.high < price), key=lambda z: -z.high)[:each_side]
    inside = [z for z in zones if z.low <= price <= z.high]
    return sorted(above + inside + below, key=lambda z: -z.center)


def format_levels(zones: list[Zone], price: float, cfg: dict, symbol: str) -> str:
    shown = select_zones(zones, price, cfg["report"]["levels_each_side"])
    lines, price_printed = [], False
    for z in shown:
        if not price_printed and z.high < price:
            lines.append(f"  ---- price {fmt_price(price, cfg, symbol)} ----")
            price_printed = True
        lines.append("  " + zone_line(z, cfg, symbol, price))
    if not price_printed and not any(z.low <= price <= z.high for z in shown):
        lines.append(f"  ---- price {fmt_price(price, cfg, symbol)} ----")
    return "\n".join(lines) if lines else "  (no zones found)"


def rule_reminders(cfg: dict) -> list[str]:
    """Rulebook limits that depend on YOUR account and your day (open trades,
    losses today). The tool is read-only and does not look at your trades yet,
    so it cannot check them: it prints them next to every plan instead. The
    trade journal milestone will check them against your MT5 history."""
    r, t = cfg["risk"], cfg["time"]["best_session_utc"]
    expiry = cfg["orders"]["pending_expiration"]
    return [
        f"At most {r['max_open_trades']} open trade(s) at a time.",
        (f"Stop for the day after {r['max_losses_per_day']} losses or "
         f"-{r['max_daily_loss_pct']:g}% on the day."),
        f"Pending orders: set the expiration to \"{expiry}\" in MT5.",
        f"Best session: {t['start']} to {t['end']} UTC (London-New York overlap).",
    ]


def format_analysis(a: Analysis, cfg: dict, source: str, news_lines: list[str] | None = None
                    ) -> str:
    f = lambda v: fmt_price(v, cfg, a.symbol)  # noqa: E731
    out = [
        f"=== {a.symbol} analysis  |  data: {source}  |  last closed "
        f"{cfg['strategy']['confirmation_timeframe']} candle: {a.as_of:%Y-%m-%d %H:%M} UTC",
        f"Price: {f(a.price)}",
        "",
        "STRUCTURE (D1/H4 = direction, H1 = confirmation)",
    ]
    for tf in ("D1", "H4", "H1"):
        if tf in a.structures:
            out.append(f"  {tf}: {a.structures[tf].summary}")

    out += ["", "KEY LEVELS (nearest to price, from highest to lowest)",
            format_levels(a.zones, a.price, cfg, a.symbol), "", "WHERE IS PRICE"]
    where = LOCATION_TEXT[a.location]
    if a.range_position_pct is not None and a.location in ("middle", "between"):
        where += f": {a.range_position_pct:.0f}% of the way from support to resistance"
    out.append(f"  {where}")
    if a.support:
        out.append(f"  nearest validated support   : {f(a.support.low)} - {f(a.support.high)}")
    if a.resistance:
        out.append(f"  nearest validated resistance: {f(a.resistance.low)} - {f(a.resistance.high)}")

    out += ["", "CANDLES AT LEVELS (last "
            f"{cfg['patterns']['lookback_bars']} closed candles)"]
    if a.signals:
        for s in a.signals:
            z = s["zone"]
            out.append(f"  {s['time']:%m-%d %H:%M}  {s['label']} ({s['direction']}) at "
                       f"{z.timeframe} zone {f(z.low)} - {f(z.high)}")
    else:
        out.append("  none")

    if news_lines is not None:
        out += ["", f"HIGH-IMPACT NEWS (next {cfg['news']['warn_hours_ahead']} h)"]
        out += [f"  {line}" for line in news_lines] or ["  none scheduled"]

    out += ["", f"VERDICT: {a.verdict}" + (f" ({a.direction.upper()})" if a.direction else "")]
    out += [f"  - {r}" for r in a.reasons]
    if a.plan is not None:
        p = a.plan
        risk_cfg = cfg["risk"]
        lot = (f"{risk_cfg['fixed_lot']} lot (learning phase: fixed size)"
               if risk_cfg["learning_phase"] else
               f"size so that the stop costs {risk_cfg['risk_pct_per_trade']}% of the balance")
        tp_note = (f"fixed at {cfg['stops']['fixed_rr']:g} x the risk"
                   if cfg["stops"].get("tp_mode") == "fixed_rr" else "before the next level")
        out += ["", f"PLAN TO CHECK ({p.direction.upper()}, market entry at the candle close)",
                f"  entry about {f(p.entry)}   (includes the spread)",
                f"  stop loss   {f(p.sl)}   (beyond the level, risk {f(p.risk)})",
                f"  take profit {f(p.tp)}   ({tp_note}, reward {f(p.reward)})",
                f"  risk:reward 1:{p.rr:.1f}",
                f"  size        {lot}",
                "  This is a plan to check, not a prediction. `market-analyst backtest` shows "
                "how such setups did in the past. Finish your pre-trade checklist "
                "(own analysis? calm? one trade at a time?) before doing anything."]
        out += ["", "YOUR RULES THE TOOL CANNOT SEE (it does not read your account)"]
        out += [f"  - {line}" for line in rule_reminders(cfg)]
    return "\n".join(out)


def draw_chart(df: pd.DataFrame, zones: list[Zone], swings: pd.DataFrame, cfg: dict,
               symbol: str, timeframe: str, source: str, trend_text: str,
               path: Path) -> Path:
    """Save a candlestick chart with zones and swing points to `path` (PNG)."""
    import matplotlib
    matplotlib.use("Agg")  # draw to a file, no window needed
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n_bars = min(cfg["report"]["chart_bars"], len(df))
    first = len(df) - n_bars
    view = df.iloc[first:]
    x = range(n_bars)                     # candle number, so weekends leave no hole
    price = float(df["close"].iloc[-1])

    fig, ax = plt.subplots(figsize=(14, 7.5), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # --- zones first, so candles are drawn on top ---
    lo, hi = float(view["low"].min()), float(view["high"].max())
    pad = (hi - lo) * 0.06
    visible = [z for z in zones if z.high >= lo - pad and z.low <= hi + pad]
    # Labels: strongest zones first; a label that would sit on top of an
    # already placed one is skipped (its band is still drawn).
    min_gap = (hi - lo + 2 * pad) * 0.026
    label_at: list[float] = []
    for z in sorted(visible, key=lambda z: (z.status != "validated", -z.score)):
        validated = z.status == "validated"
        ax.add_patch(Rectangle(
            (-1, z.low), n_bars + 1, z.high - z.low, facecolor=ZONE,
            alpha=0.30 if validated else 0.10, edgecolor="none", zorder=1))
        if validated:
            ax.hlines([z.low, z.high], -1, n_bars, colors=ZONE, linewidth=0.8, zorder=1)
        tfs = "+".join([z.timeframe] + z.confirmed_by)
        label = (f"{fmt_price(z.center, cfg, symbol)}  {z.role}  {z.touches}x  {tfs}"
                 if validated else f"{fmt_price(z.center, cfg, symbol)}  watch (1 touch)")
        if all(abs(z.center - y) >= min_gap for y in label_at):
            label_at.append(z.center)
            ax.text(n_bars + 1.5, z.center, label, va="center", ha="left", fontsize=8.5,
                    color=INK if validated else INK_SOFT, zorder=5)

    # --- candles: thin wick + body ---
    for i, (_, c) in zip(x, view.iterrows()):
        color = UP if c["close"] >= c["open"] else DOWN
        ax.vlines(i, c["low"], c["high"], colors=color, linewidth=1, zorder=3)
        body_low, body_high = sorted((c["open"], c["close"]))
        ax.add_patch(Rectangle((i - 0.32, body_low), 0.64, max(body_high - body_low, 1e-9),
                               facecolor=color, edgecolor=color, linewidth=0.6, zorder=4))

    # --- confirmed swing points (the touches zones are built from) ---
    shown = swings[swings["pos"] >= first]
    offset = (hi - lo) * 0.012
    highs, lows = shown[shown["kind"] == "high"], shown[shown["kind"] == "low"]
    ax.scatter(highs["pos"] - first, highs["price"] + offset, marker="v", s=34,
               color=INK_SOFT, zorder=6)
    ax.scatter(lows["pos"] - first, lows["price"] - offset, marker="^", s=34,
               color=INK_SOFT, zorder=6, facecolors="none")

    # --- last close ---
    ax.axhline(price, color=INK, linewidth=0.8, linestyle=(0, (4, 3)), zorder=2)
    ax.text(0.2, price, f"last close {fmt_price(price, cfg, symbol)}", va="bottom",
            ha="left", fontsize=8.5, color=INK, zorder=7,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5, "alpha": 0.85})

    # --- axes: dates on x, price on the left, quiet grid ---
    step = max(n_bars // 9, 1)
    ticks = list(range(0, n_bars, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([view.index[i].strftime("%d %b\n%H:%M") for i in ticks],
                       fontsize=8.5, color=INK_SOFT)
    ax.set_xlim(-1, n_bars + 1)
    ax.set_ylim(lo - pad, hi + pad)
    ax.tick_params(axis="y", labelsize=8.5, labelcolor=INK_SOFT, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    feed = "your MT5 terminal" if source == "mt5" else f"public feed ({source}), not broker prices"
    fig.suptitle(f"{symbol} {timeframe}: support / resistance zones", x=0.06, ha="left",
                 fontsize=14, color=INK, fontweight="bold")
    ax.set_title(
        f"{trend_text}   |   last closed candle {df.index[-1]:%Y-%m-%d %H:%M} UTC   |   "
        f"data: {feed}", loc="left", fontsize=9.5, color=INK_SOFT, pad=10)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=UP, label="candle closed up"),
        Patch(facecolor=DOWN, label="candle closed down"),
        Patch(facecolor=ZONE, alpha=0.30, label="validated zone (2+ touches)"),
        Patch(facecolor=ZONE, alpha=0.10, label="zone to watch (1 touch)"),
        Line2D([], [], marker="v", linestyle="none", color=INK_SOFT,
               label="touch: swing high with clear reaction"),
        Line2D([], [], marker="^", linestyle="none", color=INK_SOFT,
               markerfacecolor="none", label="touch: swing low with clear reaction"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.09), fontsize=8.5,
              frameon=False, ncols=3, labelcolor=INK_SOFT)

    fig.subplots_adjust(left=0.07, right=0.82, top=0.88, bottom=0.17)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path
