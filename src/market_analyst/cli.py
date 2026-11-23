"""Command-line interface.

Run `market-analyst --help` (or `python -m market_analyst --help`).

Data commands:
    fetch       download candles for every symbol x timeframe and cache them
    status      (MT5 only) show which account the terminal is on
    specs       (MT5 only) contract specs: digits, pip value, spread, lot limits
    check-time  (MT5 only) compare the configured server-time offset with the server

Analysis commands (they read the cache, so run `fetch` first):
    levels      list support / resistance zones with touch counts
    chart       save a chart image with the zones drawn
    analyze     structure + levels + location + candles -> NO TRADE / WAIT / SETUP CANDIDATE
    news        high-impact economic events this week
    deals       the orders each strategy wants right now (entry / SL / TP + past record)
    dashboard   all of the above on one web page, opened in your browser
    review      what happened to the logged deals, taken vs skipped
    journal     your real closed trades checked against your rulebook, with stats
    import-csv  load candles exported from MT5 (exact XM prices, no account access)
    backtest    replay history: how did each strategy do, with spread?
    research    search for better settings in-sample, judge them out-of-sample

Each command is a small function that receives the parsed arguments.
`main()` reads the command line and calls the right function.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import __version__, backtest, journal, news, report, research
from . import trade_journal as tj
from . import deals as deals_mod
from .analysis import analyze
from .config import ConfigError, load_config, output_dir
from .data import (
    MT5Connection,
    MT5Error,
    account_summary,
    account_warnings,
    detect_server_offset,
    expected_server_offset_hours,
    get_symbol_spec,
    load_cache,
    merge_into_cache,
    resolve_source,
    summarize_frame,
    update_cache,
    utc_now,
)
from .levels import build_zones, clear_reactions
from .strategies import STRATEGIES
from .structure import classify
from .swings import find_swings
from .yahoo import FeedError, fetch_yahoo

SOURCE_NOTES = {
    "yahoo": "NOTE: data comes from a free public feed, not your broker. GOLD uses COMEX "
             "futures (GC=F), a few dollars away from XM spot GOLD. Levels are for study; "
             "on your PC, MT5 data is the reference.",
}


def _symbols(cfg: dict, wanted: list[str] | None, include_test_markets: bool = False
             ) -> list[str]:
    """Symbols given on the command line, or the symbols in config.yaml.
    Markets marked `trade: false` are only included when asked for."""
    if not wanted:
        return [s for s, spec in cfg["symbols"].items()
                if include_test_markets or spec.get("trade", True)]
    unknown = [s for s in wanted if s not in cfg["symbols"]]
    if unknown:
        raise ConfigError(f"Not in config.yaml: {unknown}. Add them under 'symbols:' first.")
    return wanted


def _print_account(mt5, cfg: dict) -> None:
    acc = account_summary(mt5)
    print(f"Account : {acc['login']} on {acc['server']} ({acc['company']})")
    print(f"Type    : {acc['type']}, {acc['margin_mode']} mode, leverage 1:{acc['leverage']}")
    print(f"Balance : {acc['balance']:.2f} {acc['currency']}  (equity {acc['equity']:.2f})")
    for w in account_warnings(acc, cfg):
        print(f"WARNING : {w}")


# --- commands ----------------------------------------------------------------

def cmd_status(args, cfg, mt5_module=None) -> int:
    _need_mt5(cfg, "status")
    with MT5Connection(cfg, mt5_module) as mt5:
        info = mt5.terminal_info()
        if info is not None:
            print(f"Terminal: {info.name} (connected to broker: {bool(info.connected)})")
        _print_account(mt5, cfg)
    return 0


def cmd_specs(args, cfg, mt5_module=None) -> int:
    _need_mt5(cfg, "specs")
    with MT5Connection(cfg, mt5_module) as mt5:
        for sym in _symbols(cfg, args.symbols):
            spec = get_symbol_spec(mt5, sym, cfg["symbols"][sym]["pip_size"])
            print(f"\n=== {spec.name}: {spec.description}")
            print(f"  digits / point        : {spec.digits} / {spec.point}")
            print(f"  pip size (config)     : {spec.pip_size}")
            print(f"  tick size / tick value: {spec.tick_size} / {spec.tick_value} "
                  f"(per 1 lot, account currency)")
            print(f"  pip value per lot     : {spec.pip_value_per_lot:.4f}")
            print(f"  pip value at 0.01 lot : {spec.pip_value_per_lot * 0.01:.4f}")
            print(f"  contract size         : {spec.contract_size}")
            print(f"  volume min/step/max   : {spec.volume_min} / {spec.volume_step} / "
                  f"{spec.volume_max}")
            print(f"  current spread        : {spec.spread_points} points "
                  f"= {spec.spread_pips:.1f} pips")
            print(f"  broker min stop dist. : {spec.stops_level_points} points")
    return 0


def _need_mt5(cfg: dict, command: str) -> None:
    if cfg["_source"] != "mt5":
        raise ConfigError(
            f"'{command}' needs your MT5 terminal, and this machine uses data source "
            f"'{cfg['_source']}'. Run it on your Windows PC with MT5 open.")


def cmd_fetch(args, cfg, mt5_module=None) -> int:
    source = cfg["_source"]
    tfs = args.timeframes or list(cfg["data"]["bars"])
    for tf in tfs:
        if tf not in cfg["data"]["bars"]:
            raise ConfigError(f"Timeframe {tf} has no bar count under data.bars in config.yaml")
    symbols = _symbols(cfg, args.symbols, include_test_markets=cfg["_source"] != "mt5")

    def show(sym, tf, df):
        s = summarize_frame(df)
        print(f"{sym:7s} {tf:4s} {s['bars']:6d} bars  {s['first_utc']} -> {s['last_utc']} UTC  "
              f"last close {report.fmt_price(s['last_close'], cfg, sym)}")

    if source == "mt5":
        with MT5Connection(cfg, mt5_module) as mt5:
            for sym in symbols:
                for tf in tfs:
                    show(sym, tf, update_cache(mt5, cfg, sym, tf))
    else:
        for sym in symbols:
            frames = fetch_yahoo(sym, cfg)
            for tf in tfs:
                if tf not in frames:
                    raise ConfigError(f"The {source} feed only provides {sorted(frames)}")
                show(sym, tf, merge_into_cache(cfg, sym, tf, frames[tf], source))
        print("\n" + SOURCE_NOTES[source])
    print(f"\nSaved to the cache folder (data/), source: {source}. "
          "Only CLOSED candles are stored.")
    return 0


def cmd_check_time(args, cfg, mt5_module=None) -> int:
    _need_mt5(cfg, "check-time")
    sym = _symbols(cfg, args.symbols)[0]
    now = utc_now()
    expected = expected_server_offset_hours(cfg, now)
    print(f"Config mode '{cfg['time']['server_time_mode']}' predicts server = UTC{expected:+g} now.")
    with MT5Connection(cfg, mt5_module) as mt5:
        m = detect_server_offset(mt5, sym, now)
    if not m["fresh"]:
        print(f"No recent {sym} tick (market closed?). "
              "Run this again while the market is open to verify.")
        return 0
    print(f"Live {sym} tick says server = UTC{m['measured_offset_hours']:+d}.")
    if m["measured_offset_hours"] == expected:
        print("OK: config matches the server.")
        return 0
    print("MISMATCH: fix 'time' in config.yaml (try server_time_mode: fixed).")
    return 1


def _load(cfg: dict, symbol: str, tf: str):
    """Candles from the cache, or a clear message telling you to fetch first."""
    df = load_cache(cfg, symbol, tf, cfg["_source"])
    if df is None or df.empty:
        raise ConfigError(
            f"No cached {symbol} {tf} candles for source '{cfg['_source']}'. "
            "Run `market-analyst fetch` first.")
    return df


def _news_lines(cfg: dict, symbol: str, hours: float | None = None) -> list[str] | None:
    """Upcoming high-impact events for the symbol, or None if news is off/unreachable."""
    if not cfg.get("news", {}).get("enabled", False):
        return None
    try:
        events = news.parse_events(news.load_calendar(cfg))
    except news.NewsError as exc:
        return [f"(calendar unavailable: {exc})"]
    now = utc_now()
    mine = news.events_for_symbol(events, symbol, cfg)
    if hours is not None:
        mine = news.upcoming(mine, now, hours)
    return [news.describe(e, now) for e in mine]


def cmd_levels(args, cfg, mt5_module=None) -> int:
    for sym in _symbols(cfg, args.symbols):
        for tf in args.timeframes or cfg["strategy"]["level_timeframes"]:
            df = _load(cfg, sym, tf)
            zones = build_zones(df, cfg, tf)
            price = float(df["close"].iloc[-1])
            validated = sum(z.status == "validated" for z in zones)
            print(f"\n=== {sym} {tf}  last close {report.fmt_price(price, cfg, sym)}  "
                  f"({df.index[-1]:%Y-%m-%d %H:%M} UTC)  "
                  f"{len(zones)} zones, {validated} validated")
            shown = zones if args.all else None
            if shown is None:
                print(report.format_levels(zones, price, cfg, sym))
            else:
                for z in sorted(zones, key=lambda z: -z.center):
                    print("  " + report.zone_line(z, cfg, sym, price))
    if cfg["_source"] in SOURCE_NOTES:
        print("\n" + SOURCE_NOTES[cfg["_source"]])
    return 0


def cmd_chart(args, cfg, mt5_module=None) -> int:
    for sym in _symbols(cfg, args.symbols):
        for tf in args.timeframes or ["H4"]:
            df = _load(cfg, sym, tf)
            swings = find_swings(df, cfg["swings"]["fractal_n"])
            zones = build_zones(df, cfg, tf, swings)
            trend = classify(df, cfg, tf, swings).summary
            path = output_dir(cfg) / f"{sym}_{tf}_levels.png"
            touches = clear_reactions(df, swings, cfg)
            report.draw_chart(df, zones, touches, cfg, sym, tf, cfg["_source"], trend, path)
            print(f"Chart saved: {path}")
    return 0


def cmd_analyze(args, cfg, mt5_module=None) -> int:
    needed = list(dict.fromkeys(
        cfg["strategy"]["level_timeframes"] + [cfg["strategy"]["confirmation_timeframe"]]))
    for sym in _symbols(cfg, args.symbols):
        frames = {tf: _load(cfg, sym, tf) for tf in needed}
        result = analyze(sym, cfg, frames)
        lines = _news_lines(cfg, sym, cfg["news"]["warn_hours_ahead"]) if not args.no_news else None
        print("\n" + report.format_analysis(result, cfg, cfg["_source"], lines))
    if cfg["_source"] in SOURCE_NOTES:
        print("\n" + SOURCE_NOTES[cfg["_source"]])
    return 0


def cmd_news(args, cfg, mt5_module=None) -> int:
    for sym in _symbols(cfg, args.symbols):
        lines = _news_lines(cfg, sym) or []
        print(f"\n=== {sym}: {'/'.join(cfg['news']['impact'])}-impact events this week "
              f"({', '.join(cfg['symbols'][sym]['news_currencies'])})")
        print("\n".join(f"  {line}" for line in lines) or "  none")
    return 0


def _histories(cfg: dict, symbols: list[str]) -> dict:
    """Load cached candles and prepare them for replay."""
    tfs = list(dict.fromkeys(cfg["strategy"]["level_timeframes"]
                             + [cfg["backtest"]["entry_timeframe"]]))
    out = {}
    for sym in symbols:
        h = backtest.History(sym, cfg, {tf: _load(cfg, sym, tf) for tf in tfs})
        if h.n - h.first_bar < 200:
            raise ConfigError(
                f"Not enough {sym} history to backtest (need backtest.min_history_bars of "
                "H4/D1 candles plus at least 200 H1 candles after that). Fetch more data.")
        out[sym] = h
    return out


BACKTEST_NOTES = (
    "How to read this: 'avg' is the average result per trade in R (1R = the amount risked).\n"
    "Above 0 made money in the test, below 0 lost money. The '90% range' shows how much\n"
    "that average could move by luck alone: if it includes 0, the result is not\n"
    "distinguishable from no edge. A backtest describes the past, not the future."
)


def cmd_backtest(args, cfg, mt5_module=None) -> int:
    symbols = _symbols(cfg, args.symbols)
    names = args.strategies or list(STRATEGIES)
    histories = _histories(cfg, symbols)
    cuts = {sym: research.split_time(h) for sym, h in histories.items()}
    for sym, h in histories.items():
        print(f"{sym}: {h.time[h.first_bar]:%Y-%m-%d} -> {h.time[-1]:%Y-%m-%d}, "
              f"{h.n - h.first_bar} H1 candles, spread {h.spread:g}, "
              f"out-of-sample from {cuts[sym]:%Y-%m-%d}")
    all_trades = []
    for name in names:
        spec = cfg.get("filters", {}).get(name) or {}
        trades = research.run_all(histories, name, cfg, spec)
        all_trades += trades
        print(f"\n=== {name}" + (f"  (filters: {spec})" if spec else ""))
        for sym in symbols:
            mine = [t for t in trades if t.symbol == sym]
            print("  " + backtest.format_summary(f"{sym} all", backtest.summarize(mine)))
        if len(symbols) > 1:
            print("  " + backtest.format_summary("both all", backtest.summarize(trades)))
        inside, outside = research.split(trades, cuts)
        print("  " + backtest.format_summary("  older part (in-sample)",
                                             backtest.summarize(inside)))
        print("  " + backtest.format_summary("  newest part (out-of-sample)",
                                             backtest.summarize(outside)))
    if args.save and all_trades:
        path = output_dir(cfg) / "backtest_trades.csv"
        backtest.trades_frame(all_trades).to_csv(path, index=False)
        print(f"\nEvery trade saved to {path}")
    print("\n" + BACKTEST_NOTES)
    if cfg["_source"] in SOURCE_NOTES:
        print(SOURCE_NOTES[cfg["_source"]])
    return 0


def cmd_research(args, cfg, mt5_module=None) -> int:
    symbols = _symbols(cfg, args.symbols, include_test_markets=True)
    print(f"Markets: {', '.join(symbols)}")
    histories = _histories(cfg, symbols)
    fmt = backtest.format_summary
    for name in args.strategies or list(STRATEGIES):
        r = research.study(histories, name, cfg)
        print(f"\n=== {name}: {r['combinations_tried']} combinations tried on the older part")
        print("  chosen settings  :", r["overrides"] or "none (configured values were best)")
        print("  chosen conditions:", "; ".join(r["labels"]) or "none helped enough")
        print("  " + fmt("as configured, older part", r["baseline_in"]))
        print("  " + fmt("as configured, newest part", r["baseline_out"]))
        print("  " + fmt("tuned, older part (searched)", r["tuned_in"]))
        print("  " + fmt("tuned, newest part (UNSEEN)", r["tuned_out"]))
        for sym, summary in r["tuned_out_by_symbol"].items():
            print("  " + fmt(f"    {sym} newest part", summary))
        label, why = research.verdict(r)
        print(f"  -> {label}: {why}")
        if r["filters"]:
            print(f"  filter spec for config.yaml: {r['filters']}")
    print("\n" + BACKTEST_NOTES)
    print("The 'tuned, older part' line is always flattering: those settings were picked "
          "because they scored well there. Only the UNSEEN line counts.")
    if cfg["_source"] in SOURCE_NOTES:
        print(SOURCE_NOTES[cfg["_source"]])
    return 0


def cmd_deals(args, cfg, mt5_module=None) -> int:
    symbols = _symbols(cfg, args.symbols)
    histories = _histories(cfg, symbols)
    lot = cfg["risk"]["fixed_lot"]
    shown: list = []
    for sym, h in histories.items():
        f = lambda v: report.fmt_price(v, cfg, sym)  # noqa: E731
        print(f"\n=== {sym}  price {f(h.close[-1])}  "
              f"(H1 candle closed {deals_mod.candle_close(h):%Y-%m-%d %H:%M} UTC)")
        for name, d in deals_mod.find_deals(h, cfg).items():
            if d is None:
                print(f"  {name:7s} no deal")
                continue
            shown.append(d)
            print(f"  {name:7s} {d.order_text}"
                  + (f" at {f(d.entry)}" if d.kind != "market" else f" about {f(d.entry)}"))
            print(f"          SL {f(d.sl)}   TP {f(d.tp)}   R:R 1:{d.rr:.1f}"
                  f"   risk at {lot} lot about ${d.risk_dollars:.2f}")
            print(f"          why: {deals_mod.STRATEGY_TEXT[name]}"
                  + (f" [{d.pattern}]" if d.pattern else ""))
            print(f"          record on {sym}: {d.record.text}"
                  + ("  <- weak record" if d.record.weak else ""))
            if d.kind != "market":
                print("          pending order: valid for the next H1 candle only "
                      "(set expiry, re-run `deals` after each candle)")
    _news_lines(cfg, symbols[0])       # also saves this week's calendar to the archive
    if shown:
        path = output_dir(cfg) / cfg["deals"]["log_file"]
        deals_mod.log_deals(shown, path, lambda v, s: report.fmt_price(v, cfg, s), utc_now())
        print(f"\n{len(shown)} deal(s). Logged to {path} "
              "(fill in 'taken' and 'result' yourself).")
    else:
        print("\nNo deal right now. Most hours have none: re-run after the next H1 close.")
    print("These are the strategies' mechanical signals, not predictions. The backtest found "
          "no proven edge (RESEARCH.md): take them on DEMO only, at "
          f"{lot} lot, one trade at a time, always with the SL.")
    if cfg["_source"] in SOURCE_NOTES:
        print(SOURCE_NOTES[cfg["_source"]])
    return 0


def cmd_import_csv(args, cfg, mt5_module=None) -> int:
    if args.symbol not in cfg["symbols"]:
        raise ConfigError(f"{args.symbol} is not in config.yaml")
    try:
        df = journal.import_mt5_csv(cfg, args.symbol, args.timeframe, args.file)
    except journal.ImportError_ as exc:
        raise ConfigError(str(exc)) from exc
    s = summarize_frame(df)
    print(f"{args.symbol} {args.timeframe}: {s['bars']} candles in the MT5 cache, "
          f"{s['first_utc']} -> {s['last_utc']} UTC. Use `--source mt5` to analyse them.")
    return 0


def cmd_review(args, cfg, mt5_module=None) -> int:
    path = output_dir(cfg) / cfg["deals"]["log_file"]
    if not path.exists():
        raise ConfigError("No deals logged yet. Run `market-analyst deals` first.")
    tf = cfg["backtest"]["entry_timeframe"]
    candles = {sym: df for sym in cfg["symbols"]
               if (df := load_cache(cfg, sym, tf, cfg["_source"])) is not None}
    table = journal.review_deals(path, candles, cfg)
    if table.empty:
        print("No deals to review.")
        return 0
    for row in table.to_dict("records"):
        result = f"{row['r']:+.2f}R" if row["r"] is not None and row["r"] == row["r"] else ""
        print(f"  {row['candle_close_utc']}  {row['symbol']:7s} {row['strategy']:7s} "
              f"{row['direction']:4s} {row['order']:6s} taken: {row['taken']:3s}  "
              f"{row['status']:10s} {result}")
    done = table[table["r"].notna()]
    print()
    for label, part in (("deals you TOOK", done[done["taken"] == "yes"]),
                        ("deals you SKIPPED", done[done["taken"] == "no"])):
        if len(part):
            print(f"  {label}: {len(part)} finished, average {part['r'].mean():+.2f}R, "
                  f"total {part['r'].sum():+.1f}R")
        else:
            print(f"  {label}: none finished yet")
    print("Mark a deal as taken by writing yes in the 'taken' column of "
          f"{path}. With fewer than about 30 finished deals, the averages mean little.")
    return 0


def _fmt_stats(label: str, s: dict) -> str:
    if not s["trades"]:
        return f"  {label}: no trades"
    avg = f"{s['avg_r']:+.2f}R" if s["avg_r"] is not None else "n/a"
    pf = f"{s['profit_factor']:.2f}" if s["profit_factor"] is not None else "n/a"
    return (f"  {label}: {s['trades']} trades, {s['wins']} won / {s['losses']} lost "
            f"({s['win_rate']:.0f}%), net {s['net']:+.2f}, average {avg}, "
            f"profit factor {pf}, max drawdown {s['max_drawdown']:.2f}")


def cmd_journal(args, cfg, mt5_module=None) -> int:
    path = tj.trades_path(cfg)
    if args.do_import:
        _need_mt5(cfg, "journal --import")
        with MT5Connection(cfg, mt5_module) as mt5:
            new = tj.import_from_mt5(mt5, cfg)
        old = tj.load_trades(path) if path.exists() else None
        merged = tj.merge_trades(old, new)
        tj.save_trades(merged, path)
        print(f"Imported {len(new)} closed trade(s) from MT5 into {path}.")
    source = args.file or path
    try:
        trades = tj.load_trades(source)
    except tj.JournalError as exc:
        hint = "" if args.file else " Run `market-analyst journal --import` on your PC first."
        raise ConfigError(f"{exc}.{hint}") from exc
    if trades.empty:
        print("No closed trades in the journal yet.")
        return 0
    histories = tj.load_histories(cfg, trades["symbol"].unique(),
                                  lambda s, tf: load_cache(cfg, s, tf, cfg["_source"]))
    reviews = tj.review_all(trades, histories, cfg)
    for rv in reviews:
        t = rv.trade
        f = lambda v: report.fmt_price(v, cfg, t["symbol"]) if t["symbol"] in cfg["symbols"] \
            else f"{v:g}"  # noqa: E731
        r = f"{rv.r:+.2f}R" if rv.r is not None else "  ?R"
        net = t["profit"] + t["commission"] + t["swap"]
        print(f"\n{t['open_time_utc']:%Y-%m-%d %H:%M}  {t['symbol']:7s} {t['direction']:4s} "
              f"{t['volume']:g} lot  entry {f(t['open_price'])}  SL {f(t['sl'])}  "
              f"TP {f(t['tp'])}  -> {net:+.2f} ({r})"
              + (f"  [{t['exit_reason']}]" if t["exit_reason"] else ""))
        if rv.followed_rules:
            print("  every checkable rule followed")
        for c in rv.checks:
            if c.status in (tj.BROKEN, tj.NOTE) or args.verbose:
                print(f"  {c.status:6s} {c.rule}: {c.detail}")
    summary = tj.summarize(reviews)
    print("\n=== Summary")
    print(_fmt_stats("all trades          ", summary["all"]))
    print(_fmt_stats("no rule broken      ", summary["followed"]))
    print(_fmt_stats("broke at least one  ", summary["broke"]))
    print("\n=== Rules (broken / checked)")
    for rule, n in summary["rules"].items():
        checked = n[tj.OK] + n[tj.BROKEN] + n[tj.NOTE]
        extra = f", {n[tj.NOTE]} note(s)" if n[tj.NOTE] else ""
        extra += f", {n[tj.NA]} not checkable" if n[tj.NA] else ""
        print(f"  {rule:32s} {n[tj.BROKEN]} / {checked}{extra}")
    out = output_dir(cfg) / cfg["journal"]["report_file"]
    tj.reviews_frame(reviews).to_csv(out, index=False)
    print(f"\nOne row per trade, one column per rule: {out}")
    print("With fewer than about 30 trades these numbers mean little. "
          "'n/a' = the data cannot tell (e.g. fill in the 'source' column).")
    if cfg["_source"] in SOURCE_NOTES and histories:
        print(SOURCE_NOTES[cfg["_source"]])
    return 0


def cmd_dashboard(args, cfg, mt5_module=None) -> int:
    import argparse
    import webbrowser
    from . import dashboard
    from .analysis import analyze
    if args.fetch:
        cmd_fetch(argparse.Namespace(symbols=args.symbols, timeframes=None), cfg, mt5_module)
    symbols = _symbols(cfg, args.symbols)
    histories = _histories(cfg, symbols) if not args.no_backtest else {}
    needed = list(dict.fromkeys(
        cfg["strategy"]["level_timeframes"] + [cfg["strategy"]["confirmation_timeframe"]]))
    out = output_dir(cfg)
    cfg["_out_dir"] = str(out)
    data, shown = [], []
    for sym in symbols:
        frames = {tf: _load(cfg, sym, tf) for tf in needed}
        h = histories.get(sym) or _histories(cfg, [sym])[sym]
        found = deals_mod.find_deals(h, cfg)
        shown += [d for d in found.values() if d is not None]
        data.append({"symbol": sym, "analysis": analyze(sym, cfg, frames), "deals": found,
                     "history": histories.get(sym), "frames": frames,
                     "news": _news_lines(cfg, sym, 7 * 24) if not args.no_news else None})
        print(f"{sym}: done")
    deals_mod.log_deals(shown, out / cfg["deals"]["log_file"],
                        lambda v, s: report.fmt_price(v, cfg, s), utc_now())
    page = dashboard.build(cfg, cfg["_source"], data, pd.Timestamp(utc_now()),
                           SOURCE_NOTES.get(cfg["_source"]))
    path = out / "dashboard.html"
    path.write_text(page, encoding="utf-8")
    print(f"Dashboard saved: {path}")
    if not args.no_open:
        webbrowser.open(path.resolve().as_uri())
    return 0


# --- argument parsing ------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="market-analyst",
        description="Personal market analysis assistant (read-only, never places orders).",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--config", help="path to config.yaml (default: the project's one)")
    p.add_argument("--source", choices=["auto", "mt5", "yahoo"],
                   help="data source (default: data.source in config.yaml)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="connect to MT5 and show the account")

    sp = sub.add_parser("specs", help="print symbol contract specs")
    sp.add_argument("symbols", nargs="*", help="e.g. GOLD EURUSD (default: all in config)")

    fp = sub.add_parser("fetch", help="download candles and update the cache")
    fp.add_argument("symbols", nargs="*", help="default: all symbols in config")
    fp.add_argument("-t", "--timeframes", nargs="+", help="e.g. -t H4 H1 (default: all)")

    tp = sub.add_parser("check-time", help="verify the server-time offset")
    tp.add_argument("symbols", nargs="*", help="symbol used for the live tick (default: first)")

    lp = sub.add_parser("levels", help="list support / resistance zones")
    lp.add_argument("symbols", nargs="*", help="default: all symbols in config")
    lp.add_argument("-t", "--timeframes", nargs="+", help="default: D1 H4")
    lp.add_argument("--all", action="store_true", help="show every zone, not only the nearest")

    cp = sub.add_parser("chart", help="save a chart image with zones drawn")
    cp.add_argument("symbols", nargs="*", help="default: all symbols in config")
    cp.add_argument("-t", "--timeframes", nargs="+", help="default: H4")

    ap = sub.add_parser("analyze", help="full analysis and verdict")
    ap.add_argument("symbols", nargs="*", help="default: all symbols in config")
    ap.add_argument("--no-news", action="store_true", help="skip the economic calendar")

    np_ = sub.add_parser("news", help="high-impact economic events this week")
    np_.add_argument("symbols", nargs="*", help="default: all symbols in config")

    bp = sub.add_parser("backtest", help="replay history for each strategy")
    bp.add_argument("symbols", nargs="*", help="default: all symbols in config")
    bp.add_argument("-s", "--strategies", nargs="+", choices=list(STRATEGIES),
                    help="default: all (level retest h2l2)")
    bp.add_argument("--save", action="store_true", help="save every trade to reports/")

    dp = sub.add_parser("deals", help="orders the strategies want right now, with their record")
    dp.add_argument("symbols", nargs="*", help="default: all symbols in config")

    wp = sub.add_parser("dashboard", help="build a web page with everything and open it")
    wp.add_argument("symbols", nargs="*", help="default: EURUSD and GOLD")
    wp.add_argument("--fetch", action="store_true", help="download fresh candles first")
    wp.add_argument("--no-open", action="store_true", help="do not open the browser")
    wp.add_argument("--no-backtest", action="store_true", help="skip the backtest record (faster)")
    wp.add_argument("--no-news", action="store_true", help="skip the economic calendar")

    ip = sub.add_parser("import-csv", help="load candles exported from MT5 (exact XM prices)")
    ip.add_argument("symbol")
    ip.add_argument("timeframe", choices=["H1", "H4", "D1"])
    ip.add_argument("file", help="the file saved by MT5's Export Bars")

    sub.add_parser("review", help="what happened to the logged deals: taken vs skipped")

    jp = sub.add_parser("journal", help="your closed trades checked against your rulebook")
    jp.add_argument("--import", dest="do_import", action="store_true",
                    help="first read your closed trades from MT5 (read-only)")
    jp.add_argument("--file", help="read trades from this CSV instead of the journal file")
    jp.add_argument("-v", "--verbose", action="store_true",
                    help="show every check, not only the broken ones")

    rp = sub.add_parser("research", help="search settings in-sample, judge out-of-sample")
    rp.add_argument("symbols", nargs="*", help="default: all symbols in config")
    rp.add_argument("-s", "--strategies", nargs="+", choices=list(STRATEGIES),
                    help="default: all")
    return p


COMMANDS = {
    "status": cmd_status,
    "specs": cmd_specs,
    "fetch": cmd_fetch,
    "check-time": cmd_check_time,
    "levels": cmd_levels,
    "chart": cmd_chart,
    "analyze": cmd_analyze,
    "news": cmd_news,
    "backtest": cmd_backtest,
    "research": cmd_research,
    "deals": cmd_deals,
    "import-csv": cmd_import_csv,
    "review": cmd_review,
    "journal": cmd_journal,
    "dashboard": cmd_dashboard,
}


def main(argv: list[str] | None = None, mt5_module=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        if args.source:
            cfg["data"]["source"] = args.source
        # A test (or caller) that hands us an MT5 module wants MT5.
        cfg["_source"] = "mt5" if mt5_module is not None else resolve_source(cfg)
        return COMMANDS[args.command](args, cfg, mt5_module)
    except (ConfigError, MT5Error, FeedError, tj.JournalError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
