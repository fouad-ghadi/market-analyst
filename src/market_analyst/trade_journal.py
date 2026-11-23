"""Trade journal: your REAL closed trades, checked against your rulebook.

Three steps:

1. Get the trades.
   * `import_from_mt5` reads your account history from the MT5 terminal
     (read-only: `history_deals_get` and `history_orders_get`, nothing else)
     and turns the deals into one row per closed position.
   * or `load_trades` reads a CSV file with one row per trade (the file the
     import writes, or one you typed yourself for trades from before).

2. Check each trade against the rules (`check_trade`). Every check gives
   ok / BROKEN / n/a (cannot be checked from the data) / note (allowed, but
   worth knowing, e.g. a counter-trend trade). All thresholds come from
   config.yaml (stops, risk, strategy, patterns, time).

3. Summarize (`summarize`): win rate, average R, expectancy, profit factor,
   max drawdown, how often each rule was broken, and, most useful, how the
   trades that followed every rule did compared with those that broke one.

R = result divided by the money that was at risk between entry and stop.
+2R = won twice the risk. -1R = full stop loss.

This module never sends, changes or closes an order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import locate, trend_alignment
from .backtest import TF_HOURS, History
from .config import cache_dir
from .data import MT5Error, server_to_utc, utc_to_server

# Columns of the trades file. The last two are yours to fill in (in Excel or a
# text editor): `source` = own / signal, `note` = anything. A new import keeps them.
COLUMNS = ["position", "symbol", "direction", "order_type", "volume",
           "open_time_utc", "open_price", "sl", "tp", "close_time_utc", "close_price",
           "exit_reason", "profit", "commission", "swap", "balance_before", "source", "note"]
USER_COLUMNS = ["source", "note"]

# MetaTrader 5 API codes (fixed by MetaQuotes, not trading rules).
DEAL_BUY, DEAL_SELL, DEAL_BALANCE = 0, 1, 2
DEAL_ENTRY_IN = 0
DEAL_REASONS = {4: "sl", 5: "tp", 6: "stop out"}           # anything else = closed by you
ORDER_TYPES = {0: "market", 1: "market", 2: "limit", 3: "limit", 4: "stop", 5: "stop",
               6: "stop limit", 7: "stop limit"}

OK, BROKEN, NA, NOTE = "ok", "BROKEN", "n/a", "note"


class JournalError(ValueError):
    """Raised when a trades file cannot be understood."""


# =============================================================================
# 1. Getting the trades
# =============================================================================

def trades_path(cfg: dict) -> Path:
    return cache_dir(cfg) / cfg["journal"]["trades_file"]


def load_trades(path: str | Path) -> pd.DataFrame:
    """Read a trades CSV. Required columns: symbol, direction, volume,
    open_time_utc, open_price, sl, tp, close_time_utc, close_price, profit.
    The other columns are optional."""
    path = Path(path)
    if not path.exists():
        raise JournalError(f"File not found: {path}")
    df = pd.read_csv(path, dtype={"source": str, "note": str, "position": str})
    df.columns = [c.strip().lower() for c in df.columns]
    needed = {"symbol", "direction", "volume", "open_time_utc", "open_price", "sl", "tp",
              "close_time_utc", "close_price", "profit"}
    missing = needed - set(df.columns)
    if missing:
        raise JournalError(f"{path.name} is missing the columns {sorted(missing)}")
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = np.nan if col in ("balance_before", "commission", "swap") else ""
    df["position"] = df["position"].fillna("").astype(str)
    df[["commission", "swap"]] = df[["commission", "swap"]].fillna(0.0)
    for col in ("order_type", "exit_reason", "source", "note"):
        df[col] = df[col].fillna("").astype(str).str.strip()
    df["direction"] = df["direction"].str.strip().str.lower()
    bad = set(df["direction"]) - {"buy", "sell"}
    if bad:
        raise JournalError(f"direction must be buy or sell, found {sorted(bad)}")
    for col in ("open_time_utc", "close_time_utc"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df[COLUMNS].sort_values("open_time_utc").reset_index(drop=True)


def save_trades(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    for col in ("open_time_utc", "close_time_utc"):
        out[col] = out[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    out[COLUMNS].to_csv(path, index=False)


def merge_trades(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Add newly imported trades to the file. Your own `source` and `note`
    entries on trades already in the file are kept."""
    if old is None or old.empty:
        return new
    keep = old.set_index("position")[USER_COLUMNS]
    new = new.copy()
    for col in USER_COLUMNS:
        mine = new["position"].map(keep[col]).fillna("")
        new[col] = mine.where(mine != "", new[col])
    hand_typed = old[old["position"].eq("") | ~old["position"].isin(new["position"])]
    out = pd.concat([hand_typed, new], ignore_index=True)
    return out.sort_values("open_time_utc").reset_index(drop=True)


def _server_seconds_to_utc(seconds, cfg: dict) -> pd.Series:
    """MT5 gives deal times as server-time seconds (same convention as candles)."""
    return server_to_utc(pd.Series(pd.to_datetime(list(seconds), unit="s")), cfg)


def positions_from_deals(deals: list, orders: list, cfg: dict) -> pd.DataFrame:
    """Turn MT5 deals (one per fill) into one row per CLOSED position.

    A position has entry deals (IN) and exit deals (OUT). Prices are averaged by
    volume. SL / TP come from the order that opened the position: that is the
    stop you planned, even if you moved it later. If you added the stop only
    after entering, the opening order has none, and we take the first order of
    that position that had one.
    """
    if not deals:
        return pd.DataFrame(columns=COLUMNS)
    d = pd.DataFrame([x._asdict() if hasattr(x, "_asdict") else dict(vars(x)) for x in deals])
    for col in ("commission", "swap", "fee"):
        if col not in d.columns:
            d[col] = 0.0
    d["time_utc"] = _server_seconds_to_utc(d["time"], cfg).to_numpy()
    d = d.sort_values(["time_utc", "ticket"]).reset_index(drop=True)
    d["cash"] = d["profit"] + d["commission"] + d["swap"] + d["fee"]
    d["balance_after"] = d["cash"].cumsum()           # deposits are balance deals too

    o = pd.DataFrame([x._asdict() if hasattr(x, "_asdict") else dict(vars(x)) for x in orders]) \
        if orders else pd.DataFrame(columns=["ticket", "type", "sl", "tp", "position_id"])

    rows = []
    trades = d[d["type"].isin([DEAL_BUY, DEAL_SELL])]
    for pid, g in trades.groupby("position_id", sort=False):
        ins, outs = g[g["entry"] == DEAL_ENTRY_IN], g[g["entry"] != DEAL_ENTRY_IN]
        if ins.empty or outs.empty or outs["volume"].sum() + 1e-9 < ins["volume"].sum():
            continue                                    # still open (or partly open)
        first = ins.iloc[0]
        opening = o[o["ticket"] == first["order"]]
        own = o[o["position_id"] == pid] if "position_id" in o.columns else o.iloc[0:0]
        sl = _first_nonzero(opening, "sl") or _first_nonzero(own, "sl")
        tp = _first_nonzero(opening, "tp") or _first_nonzero(own, "tp")
        order_type = ORDER_TYPES.get(int(opening["type"].iloc[0]), "") if len(opening) else ""
        # Balance just before this trade, if the history includes your deposit.
        earlier = d.loc[: first.name - 1] if first.name > 0 else d.iloc[0:0]
        deposited = (earlier["type"] == DEAL_BALANCE).any()
        before = earlier["cash"].sum() if deposited else np.nan
        rows.append({
            "position": str(pid), "symbol": first["symbol"],
            "direction": "buy" if first["type"] == DEAL_BUY else "sell",
            "order_type": order_type, "volume": ins["volume"].sum(),
            "open_time_utc": first["time_utc"],
            "open_price": float(np.average(ins["price"], weights=ins["volume"])),
            "sl": sl, "tp": tp, "close_time_utc": outs["time_utc"].iloc[-1],
            "close_price": float(np.average(outs["price"], weights=outs["volume"])),
            "exit_reason": DEAL_REASONS.get(int(outs["reason"].iloc[-1]), "manual")
            if "reason" in outs.columns else "",
            "profit": g["profit"].sum(), "commission": g["commission"].sum() + g["fee"].sum(),
            "swap": g["swap"].sum(),
            "balance_before": before,
            "source": "", "note": "",
        })
    out = pd.DataFrame(rows, columns=COLUMNS)
    return out.sort_values("open_time_utc").reset_index(drop=True)


def _first_nonzero(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame.columns:
        return 0.0
    vals = frame[col][frame[col] > 0]
    return float(vals.iloc[0]) if len(vals) else 0.0


def import_from_mt5(mt5, cfg: dict, now: datetime | None = None) -> pd.DataFrame:
    """Read the account history from the terminal (read-only)."""
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(days=cfg["journal"]["history_days"])
    end = now + timedelta(days=1)            # server time runs ahead of UTC
    deals = mt5.history_deals_get(start, end)
    if deals is None:
        raise MT5Error(f"history_deals_get failed: {mt5.last_error()}")
    orders = mt5.history_orders_get(start, end) or []
    return positions_from_deals(list(deals), list(orders), cfg)


# =============================================================================
# 2. Checking one trade against the rules
# =============================================================================

@dataclass
class Check:
    rule: str
    status: str          # ok / BROKEN / n/a / note
    detail: str


@dataclass
class Review:
    trade: dict
    risk_money: float | None         # money lost if the stop had been hit
    r: float | None                  # result in R
    checks: list[Check] = field(default_factory=list)

    @property
    def broken(self) -> list[Check]:
        return [c for c in self.checks if c.status == BROKEN]

    @property
    def followed_rules(self) -> bool:
        return not self.broken


def money_per_price_unit(t: dict, cfg: dict) -> float | None:
    """How much the account gains or loses when price moves by 1.0 on this trade.

    Best source: the trade itself (profit / price move). That is exact, in your
    account currency, whatever the pair (works for USD/JPY too). Only when the
    trade closed at its entry price do we fall back to contract size x volume,
    which is only right when the quote currency is the account currency."""
    move = t["close_price"] - t["open_price"]
    if t["direction"] == "sell":
        move = -move
    if abs(move) > 1e-12 and t["profit"] != 0:
        return abs(t["profit"] / move)
    spec = cfg["symbols"].get(t["symbol"], {})
    if spec.get("contract_size"):
        return spec["contract_size"] * t["volume"]
    return None


def check_trade(t: dict, earlier: list[dict], balance: float, history: History | None,
                cfg: dict) -> Review:
    """Check one closed trade. `earlier` = the trades opened before it (for
    the one-trade-at-a-time, daily-stop and revenge rules). `history` = the
    candles of its symbol, or None if they are not available."""
    s, risk_cfg = cfg["stops"], cfg["risk"]
    buy = t["direction"] == "buy"
    entry, sl, tp = t["open_price"], t["sl"], t["tp"]
    has_sl = bool(sl and sl > 0) and ((sl < entry) if buy else (sl > entry))
    has_tp = bool(tp and tp > 0) and ((tp > entry) if buy else (tp < entry))
    per_unit = money_per_price_unit(t, cfg)
    risk_money = abs(entry - sl) * per_unit if has_sl and per_unit else None
    net = t["profit"] + t["commission"] + t["swap"]
    r = net / risk_money if risk_money else None
    checks: list[Check] = []

    # --- Never trade without a stop loss ---
    checks.append(Check("stop loss set", OK if has_sl else BROKEN,
                        "" if has_sl else "no stop loss on the order (or on the wrong side)"))

    # --- Minimum risk:reward ---
    if has_sl and has_tp:
        rr = abs(tp - entry) / abs(entry - sl)
        ok = rr + 1e-9 >= s["min_risk_reward"]
        checks.append(Check("risk:reward", OK if ok else BROKEN,
                            f"planned 1:{rr:.2f} (minimum 1:{s['min_risk_reward']:g})"))
    else:
        checks.append(Check("risk:reward", BROKEN if has_sl else NA,
                            "no take profit, so the reward was never planned" if has_sl
                            else "no stop: risk:reward cannot be measured"))

    # --- Risk per trade ---
    if risk_money is not None and balance > 0:
        pct = risk_money / balance * 100
        ok = pct <= risk_cfg["risk_pct_per_trade"] + 1e-9
        checks.append(Check("risk per trade", OK if ok else BROKEN,
                            f"{risk_money:.2f} at risk = {pct:.2f}% of {balance:.0f} "
                            f"(max {risk_cfg['risk_pct_per_trade']:g}%)"))
    else:
        checks.append(Check("risk per trade", NA, "no stop, so the risk is unknown"
                            if not has_sl else "balance unknown"))
    if risk_cfg["learning_phase"]:
        ok = t["volume"] <= risk_cfg["fixed_lot"] + 1e-9
        checks.append(Check("learning lot size", OK if ok else BROKEN,
                            f"{t['volume']:g} lot (learning phase: {risk_cfg['fixed_lot']:g})"))

    # --- One trade at a time ---
    open_then = [e for e in earlier if e["close_time_utc"] > t["open_time_utc"]]
    ok = len(open_then) < risk_cfg["max_open_trades"]
    checks.append(Check("max open trades", OK if ok else BROKEN,
                        f"{len(open_then) + 1} trade(s) open at once "
                        f"(max {risk_cfg['max_open_trades']})"))

    # --- Stop for the day after N losses or -X% ---
    day = _server_day(t["open_time_utc"], cfg)
    closed_today = [e for e in earlier if e["close_time_utc"] <= t["open_time_utc"]
                    and _server_day(e["close_time_utc"], cfg) == day]
    nets = [e["profit"] + e["commission"] + e["swap"] for e in closed_today]
    losses = sum(1 for x in nets if x < 0)
    day_pct = sum(nets) / balance * 100 if balance > 0 else 0.0
    stop_day = (losses >= risk_cfg["max_losses_per_day"]
                or day_pct <= -risk_cfg["max_daily_loss_pct"])
    checks.append(Check("daily stop", BROKEN if stop_day else OK,
                        f"{losses} loss(es) and {day_pct:+.2f}% earlier that day "
                        f"(stop at {risk_cfg['max_losses_per_day']} losses or "
                        f"-{risk_cfg['max_daily_loss_pct']:g}%)"))

    # --- No revenge trading: never a bigger size right after a loss ---
    prev = [e for e in earlier if e["close_time_utc"] <= t["open_time_utc"]]
    if prev and not risk_cfg["allow_size_increase_after_loss"]:
        last = max(prev, key=lambda e: e["close_time_utc"])
        lost = last["profit"] + last["commission"] + last["swap"] < 0
        bigger = t["volume"] > last["volume"] + 1e-9
        checks.append(Check("no size increase after a loss", BROKEN if lost and bigger else OK,
                            f"{t['volume']:g} lot after a {'loss' if lost else 'win'} "
                            f"at {last['volume']:g} lot"))

    # --- Own analysis, not a signal ---
    src = str(t.get("source") or "").strip().lower()
    if not src:
        checks.append(Check("own analysis", NA, "fill in the 'source' column (own / signal)"))
    else:
        own = src in ("own", "me", "mine", "my analysis", "analysis")
        checks.append(Check("own analysis", OK if own else BROKEN, f"source: {src}"))

    checks += chart_checks(t, history, has_sl, cfg)
    return Review(t, risk_money, r, checks)


def _server_day(ts: pd.Timestamp, cfg: dict):
    """The broker's trading day (server midnight = 17:00 New York)."""
    return utc_to_server(pd.DatetimeIndex([ts]), cfg)[0].date()


def chart_checks(t: dict, h: History | None, has_sl: bool, cfg: dict) -> list[Check]:
    """Rules that need the chart: at a level? middle of the range? stop beyond
    the level? with the H4 trend? Only candles that had CLOSED before the entry
    are used, so this sees what you could see when you clicked."""
    names = ("entry at a validated level", "not in the middle of the range",
             "stop beyond the level")
    trend_tf = cfg["journal"]["trend_timeframe"]
    if h is None:
        return [Check(n, NA, "no candles for this symbol: run `fetch` first")
                for n in names] + [Check(f"{trend_tf} trend", NA, "no candles")]
    closes_at = h.time + pd.Timedelta(hours=TF_HOURS[h.entry_tf])
    i = int(closes_at.searchsorted(t["open_time_utc"], side="right")) - 1
    if i < h.first_bar:
        return [Check(n, NA, "the cached candles do not go back to this trade")
                for n in names] + [Check(f"{trend_tf} trend", NA, "not enough history")]

    buy = t["direction"] == "buy"
    entry = t["open_price"]
    zones = h.zones(i)
    margin = cfg["patterns"]["near_level_atr_mult"] * h.atr[i]
    loc = locate(entry, entry, entry, zones, margin, cfg)
    level = loc.level
    right_side = (loc.where == "inside_zone" or (buy and loc.where == "at_support")
                  or (not buy and loc.where == "at_resistance"))
    f = cfg["symbols"].get(t["symbol"], {}).get("price_decimals", 5)
    if level is not None and right_side:
        at = Check(names[0], OK, f"at {loc.where.replace('_', ' ')} "
                   f"{level.low:.{f}f}-{level.high:.{f}f} ({level.timeframe}, "
                   f"{level.touches} touches)")
    elif level is not None:
        at = Check(names[0], BROKEN, f"{t['direction']} at {loc.where.replace('_', ' ')}: "
                   "wrong side of the level")
    else:
        at = Check(names[0], BROKEN, f"no validated level within {margin:.{f}f} of the entry")

    if loc.where == "middle":
        mid = Check(names[1], BROKEN, f"entry at {loc.pct:.0f}% of the range "
                    f"(middle {cfg['strategy']['middle_of_range_pct']}% = NO TRADE)")
    elif loc.pct is None:
        mid = Check(names[1], OK, "no validated range around the entry")
    else:
        mid = Check(names[1], OK, f"entry at {loc.pct:.0f}% of the range")

    if not has_sl:
        stop = Check(names[2], NA, "no stop loss")
    elif level is None or not right_side:
        stop = Check(names[2], NA, "no level to put the stop behind")
    else:
        beyond = t["sl"] < level.low if buy else t["sl"] > level.high
        stop = Check(names[2], OK if beyond else BROKEN,
                     f"stop {t['sl']:.{f}f}, level edge "
                     f"{(level.low if buy else level.high):.{f}f}")

    align = trend_alignment(h.trend(trend_tf, i), t["direction"])
    trend = Check(f"{trend_tf} trend", OK if align == "with trend" else NOTE,
                  align + ("" if align == "with trend" else " (allowed, lower conviction)"))
    return [at, mid, stop, trend]


def review_all(trades: pd.DataFrame, histories: dict[str, History], cfg: dict
               ) -> list[Review]:
    """Check every trade, oldest first, keeping a running balance."""
    reviews: list[Review] = []
    records = trades.to_dict("records")
    balance = float(cfg["journal"]["starting_balance"])
    for k, t in enumerate(records):
        if t.get("balance_before") == t.get("balance_before") and t.get("balance_before"):
            balance_now = float(t["balance_before"])
        else:
            done = [e for e in records[:k] if e["close_time_utc"] <= t["open_time_utc"]]
            balance_now = balance + sum(e["profit"] + e["commission"] + e["swap"] for e in done)
        reviews.append(check_trade(t, records[:k], balance_now,
                                   histories.get(t["symbol"]), cfg))
    return reviews


# =============================================================================
# 3. Statistics
# =============================================================================

def stats(reviews: list[Review]) -> dict:
    """Plain numbers for a group of trades."""
    nets = [rv.trade["profit"] + rv.trade["commission"] + rv.trade["swap"] for rv in reviews]
    rs = [rv.r for rv in reviews if rv.r is not None]
    wins, losses = [x for x in nets if x > 0], [x for x in nets if x < 0]
    curve = np.cumsum([0.0] + nets)
    return {
        "trades": len(nets),
        "wins": len(wins), "losses": len(losses),
        "win_rate": len(wins) / len(nets) * 100 if nets else None,
        "net": float(sum(nets)),
        "avg_r": float(np.mean(rs)) if rs else None,
        "trades_with_r": len(rs),
        "profit_factor": sum(wins) / -sum(losses) if losses else None,
        "max_drawdown": float(np.max(np.maximum.accumulate(curve) - curve)),
    }


def summarize(reviews: list[Review]) -> dict:
    rules: dict[str, dict[str, int]] = {}
    for rv in reviews:
        for c in rv.checks:
            rules.setdefault(c.rule, {OK: 0, BROKEN: 0, NA: 0, NOTE: 0})[c.status] += 1
    return {
        "all": stats(reviews),
        "followed": stats([rv for rv in reviews if rv.followed_rules]),
        "broke": stats([rv for rv in reviews if not rv.followed_rules]),
        "rules": rules,
    }


def reviews_frame(reviews: list[Review]) -> pd.DataFrame:
    """One row per trade, one column per rule (for Excel)."""
    rows = []
    for rv in reviews:
        t = rv.trade
        row = {"open_time_utc": f"{t['open_time_utc']:%Y-%m-%d %H:%M}", "symbol": t["symbol"],
               "direction": t["direction"], "volume": t["volume"],
               "entry": t["open_price"], "sl": t["sl"], "tp": t["tp"],
               "net": round(t["profit"] + t["commission"] + t["swap"], 2),
               "r": None if rv.r is None else round(rv.r, 2),
               "rules_broken": len(rv.broken)}
        row.update({c.rule: c.status for c in rv.checks})
        rows.append(row)
    return pd.DataFrame(rows)


def load_histories(cfg: dict, symbols, load) -> dict[str, History]:
    """Candles for each traded symbol that is in config.yaml. `load(symbol, tf)`
    returns cached candles or None. Missing data = chart rules become n/a."""
    tfs = list(dict.fromkeys(cfg["strategy"]["level_timeframes"]
                             + [cfg["backtest"]["entry_timeframe"]]))
    out = {}
    for sym in symbols:
        if sym not in cfg["symbols"]:
            continue
        frames = {tf: load(sym, tf) for tf in tfs}
        if any(f is None or f.empty for f in frames.values()):
            continue
        out[sym] = History(sym, cfg, frames)
    return out


__all__ = ["JournalError", "load_trades", "save_trades", "merge_trades",
           "positions_from_deals", "import_from_mt5", "check_trade", "review_all",
           "summarize", "reviews_frame", "load_histories", "trades_path"]
