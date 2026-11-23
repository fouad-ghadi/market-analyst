"""Find the deals the strategies want right now, with their past record.

Used by `market-analyst deals` (terminal) and `market-analyst dashboard`
(web page), so both always show exactly the same deals.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import backtest, planner, research
from .backtest import History
from .strategies import STRATEGIES

STRATEGY_TEXT = {
    "level": "reversal candle at a validated level (with the tuned conditions)",
    "retest": "second entry: return to a level after a confirmation candle",
    "h2l2": "Brooks second entry: second attempt of a pullback, with the H4 trend",
}
ORDER_TEXT = {
    ("market", "buy"): "BUY now (market)", ("market", "sell"): "SELL now (market)",
    ("limit", "buy"): "BUY LIMIT", ("limit", "sell"): "SELL LIMIT",
    ("stop", "buy"): "BUY STOP", ("stop", "sell"): "SELL STOP",
}


@dataclass
class Record:
    """How a strategy did in the backtest on one symbol."""
    trades: int
    avg_r: float | None
    new_trades: int
    new_avg_r: float | None

    @property
    def weak(self) -> bool:
        """Negative overall, negative on the newest part, or too few trades to judge."""
        return not (self.new_trades >= 20 and (self.new_avg_r or 0) > 0
                    and (self.avg_r or 0) > 0)

    @property
    def text(self) -> str:
        if not self.trades or not self.new_trades:
            return "too few past trades to judge"
        return (f"{self.trades} past trades avg {self.avg_r:+.2f}R; "
                f"newest part {self.new_trades} trades avg {self.new_avg_r:+.2f}R")


@dataclass
class Deal:
    symbol: str
    strategy: str
    kind: str              # market / limit / stop
    direction: str         # buy / sell
    entry: float
    sl: float
    tp: float
    risk_dollars: float | None   # None = cannot convert to the account currency
    pattern: str
    record: Record
    candle_close: pd.Timestamp

    @property
    def rr(self) -> float:
        risk = abs(self.entry - self.sl)
        return abs(self.tp - self.entry) / risk if risk else 0.0

    @property
    def risk_text(self) -> str:
        return "unknown" if self.risk_dollars is None else f"${self.risk_dollars:.2f}"

    @property
    def order_text(self) -> str:
        return ORDER_TEXT[(self.kind, self.direction)]


def risk_in_account_currency(cfg: dict, symbol: str, distance: float, lots: float,
                             price: float) -> float | None:
    """Money lost if price moves `distance` against `lots` lots, in the account
    currency (risk.account_currency, USD for you).

    A price move is counted in the QUOTE currency (the second one of the pair):
      EURUSD 1.1500 -> 1.1490 on 1 lot = 0.0010 x 100,000 = 100 USD
      USDJPY 155.00 -> 154.00 on 1 lot = 1.00 x 100,000 = 100,000 JPY (not USD!)
    So we convert:
      quote currency = account currency (EURUSD, GOLD) -> nothing to do
      base currency  = account currency (USDJPY)       -> divide by the price
      neither (e.g. EURGBP)                            -> unknown: returns None
    """
    spec = cfg["symbols"][symbol]
    account = cfg["risk"].get("account_currency", "USD")
    in_quote = distance * spec.get("contract_size", 0) * lots
    if spec.get("quote_currency", account) == account:
        return in_quote
    if spec.get("base_currency") == account and price > 0:
        return in_quote / price
    return None


def record_of(h: History, name: str, cfg: dict) -> Record:
    spec = cfg.get("filters", {}).get(name) or {}
    trades = research.run_all({h.symbol: h}, name, cfg, spec)
    _, unseen = research.split(trades, {h.symbol: research.split_time(h)})
    a, b = backtest.summarize(trades), backtest.summarize(unseen)
    return Record(a["trades"], a["expectancy_r"], b["trades"], b["expectancy_r"])


def candle_close(h: History) -> pd.Timestamp:
    return h.time[-1] + pd.Timedelta(hours=backtest.TF_HOURS[h.entry_tf])


def find_deals(h: History, cfg: dict) -> dict[str, Deal | None]:
    """{strategy name: Deal or None} for one symbol, as of its last closed candle."""
    out: dict[str, Deal | None] = {}
    lot = cfg["risk"]["fixed_lot"]
    for name, cls in STRATEGIES.items():
        spec = cfg.get("filters", {}).get(name) or {}
        order = backtest.latest_order(h, cls(cfg, accept=research.make_accept(spec)),
                                      cfg["deals"]["warmup_bars"])
        if order is None:
            out[name] = None
            continue
        entry = order.price if order.price is not None else planner.fill_price(
            order.direction, h.close[-1], h.spread)
        risk = abs(entry - order.sl)
        out[name] = Deal(
            symbol=h.symbol, strategy=name, kind=order.kind, direction=order.direction,
            entry=entry, sl=order.sl, tp=order.tp,
            risk_dollars=risk_in_account_currency(cfg, h.symbol, risk, lot, entry),
            pattern=order.info.get("pattern", ""), record=record_of(h, name, cfg),
            candle_close=candle_close(h))
    return out


def log_deals(deals: list[Deal], path: Path, fmt, shown_at: pd.Timestamp) -> None:
    """Append deals to the CSV log (for `market-analyst review`)."""
    if not deals:
        return
    rows = [{
        "shown_at_utc": f"{shown_at:%Y-%m-%d %H:%M}",
        "candle_close_utc": f"{d.candle_close:%Y-%m-%d %H:%M}", "symbol": d.symbol,
        "strategy": d.strategy, "order": d.kind, "direction": d.direction,
        "entry": fmt(d.entry, d.symbol), "sl": fmt(d.sl, d.symbol), "tp": fmt(d.tp, d.symbol),
        "rr": round(d.rr, 2), "taken": "", "result": "",
    } for d in deals]
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
