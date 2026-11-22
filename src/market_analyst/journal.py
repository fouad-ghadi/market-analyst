"""Two bridges to your real trading:

1. import_mt5_csv: load candles you exported from MT5 (exact XM prices) into
   the cache, without connecting anything to your account.
2. review_deals: go back over every deal the tool showed (reports/deals_log.csv),
   work out what happened to it afterwards, and compare the deals you took
   with the ones you skipped. This is how the tool learns from YOUR choices.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .backtest import TF_HOURS, Order, check_exit, try_fill
from .data import merge_into_cache, rates_to_frame


class ImportError_(ValueError):
    """Raised when an exported file cannot be understood."""


def import_mt5_csv(cfg: dict, symbol: str, tf: str, path: str | Path) -> pd.DataFrame:
    """Read an MT5 bar export and merge it into the MT5 cache.

    In MT5: View > Symbols > select the symbol > "Bars" tab > choose the
    timeframe and dates > Request > Export Bars. The file has columns
    <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>
    (D1 files have no <TIME>). Times are broker server time.
    """
    path = Path(path)
    if not path.exists():
        raise ImportError_(f"File not found: {path}")
    raw = pd.read_csv(path, sep=None, engine="python", encoding_errors="ignore")
    raw.columns = [str(c).strip("<> ﻿").lower() for c in raw.columns]
    needed = {"date", "open", "high", "low", "close"}
    if not needed <= set(raw.columns):
        raise ImportError_(f"{path.name} does not look like an MT5 bar export "
                           f"(columns found: {list(raw.columns)})")
    stamp = raw["date"].astype(str).str.replace(".", "-", regex=False)
    if "time" in raw.columns:
        stamp = stamp + " " + raw["time"].astype(str)
    server = pd.to_datetime(stamp)
    rates = pd.DataFrame({
        # rates_to_frame expects MT5's own convention: server time as epoch seconds
        "time": (server - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1),
        "open": raw["open"], "high": raw["high"], "low": raw["low"], "close": raw["close"],
        "tick_volume": raw.get("tickvol", 0), "spread": raw.get("spread", 0),
        "real_volume": raw.get("vol", 0),
    })
    df = rates_to_frame(rates, cfg)
    # The newest exported candle may still be forming: keep closed candles only.
    closes_at = df.index + pd.Timedelta(hours=TF_HOURS[tf])
    df = df[closes_at <= pd.Timestamp.now(tz="UTC")]
    if df.empty:
        raise ImportError_("No closed candles in the file")
    return merge_into_cache(cfg, symbol, tf, df, "mt5")


def resolve_deal(row: dict, h1: pd.DataFrame, spread: float, max_hold: int) -> dict:
    """What happened to one logged deal? Uses the same fill and exit rules as
    the backtest. `h1` are the H1 candles of the deal's symbol."""
    closed = pd.Timestamp(row["candle_close_utc"], tz="UTC")
    after = h1[h1.index >= closed]             # candles that opened after the signal
    entry, sl, tp = float(row["entry"]), float(row["sl"]), float(row["tp"])
    order = Order(row["order"], row["direction"], sl=sl, tp=tp,
                  price=None if row["order"] == "market" else entry)
    if after.empty:
        return {"status": "waiting", "r": None}
    first = after.iloc[0]
    fill = try_fill(order, first["open"], first["high"], first["low"], spread)
    if fill is None:
        return {"status": "not filled", "r": None}
    risk = abs(fill - sl)
    buy = order.direction == "buy"
    for k, (_, c) in enumerate(after.iterrows()):
        result = check_exit(order.direction, sl, tp, c["open"], c["high"], c["low"], spread,
                            k == 0, order.kind == "market")
        if result is None and k >= max_hold:
            result = ("timeout", c["close"] if buy else c["close"] + spread)
        if result is not None:
            outcome, price = result
            gain = price - fill if buy else fill - price
            return {"status": outcome, "r": gain / risk}
    return {"status": "open", "r": None}


def review_deals(log_path: Path, candles: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    """Resolve every logged deal. Returns the log with 'status' and 'r' columns.
    The same deal shown several times (re-runs within one candle) counts once."""
    log = pd.read_csv(log_path, dtype=str).fillna("")
    log = log.drop_duplicates(subset=["candle_close_utc", "symbol", "strategy"], keep="last")
    out = []
    for row in log.to_dict("records"):
        sym = row["symbol"]
        if sym not in candles:
            continue
        spread = float(cfg["symbols"][sym].get("typical_spread", 0.0))
        row.update(resolve_deal(row, candles[sym], spread, cfg["backtest"]["max_hold_bars"]))
        row["taken"] = "yes" if row["taken"].strip().lower() in ("yes", "y", "1", "x") else "no"
        out.append(row)
    return pd.DataFrame(out)
