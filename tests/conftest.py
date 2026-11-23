"""Shared test helpers: a fake MT5 terminal and a test config.

The real MetaTrader5 package only works on Windows with the terminal open.
FakeMT5 imitates the few functions we use and returns believable data, so
the tests run anywhere, in a second, without touching your account.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from market_analyst.config import DEFAULT_CONFIG_PATH, load_config

TF_SECONDS = {"M5": 300, "M15": 900, "H1": 3600, "H4": 14400, "D1": 86400}

RATES_DTYPE = np.dtype([
    ("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"), ("close", "<f8"),
    ("tick_volume", "<u8"), ("spread", "<i4"), ("real_volume", "<u8"),
])

SYMBOLS = {
    "EURUSD": dict(name="EURUSD", description="Euro vs US Dollar", digits=5, point=0.00001,
                   trade_tick_size=0.00001, trade_tick_value=1.0, trade_contract_size=100000,
                   volume_min=0.01, volume_max=50.0, volume_step=0.01, spread=16,
                   trade_stops_level=0, currency_profit="USD", base=1.1500),
    "GOLD": dict(name="GOLD", description="Gold vs US Dollar", digits=2, point=0.01,
                 trade_tick_size=0.01, trade_tick_value=1.0, trade_contract_size=100,
                 volume_min=0.01, volume_max=50.0, volume_step=0.01, spread=35,
                 trade_stops_level=0, currency_profit="USD", base=4100.0),
}


class FakeMT5:
    """Minimal stand-in for the MetaTrader5 module (read-only functions only)."""

    TIMEFRAME_M5, TIMEFRAME_M15, TIMEFRAME_H1, TIMEFRAME_H4, TIMEFRAME_D1 = (
        5, 15, 16385, 16388, 16408)
    _TF_NAMES = {5: "M5", 15: "M15", 16385: "H1", 16388: "H4", 16408: "D1"}

    def __init__(self, n_bars: int = 300, trade_mode: int = 0, init_ok: bool = True,
                 tick_time: int | None = None, deals=(), orders=()):
        self.deals, self.orders = list(deals), list(orders)   # account history
        self.n_bars = n_bars              # total candles that "exist" on the server
        self.trade_mode = trade_mode      # 0 demo, 2 real
        self.init_ok = init_ok
        self.tick_time = tick_time        # server-time seconds of the last tick
        self.initialized = False
        self.shutdown_called = False
        self.start_server = int(datetime(2026, 1, 5, tzinfo=timezone.utc).timestamp())

    # --- connection ---
    def initialize(self, *args, **kwargs):
        self.initialized = self.init_ok
        return self.init_ok

    def shutdown(self):
        self.shutdown_called = True
        self.initialized = False

    def last_error(self):
        return (1, "Success") if self.init_ok else (-10005, "IPC timeout")

    # --- account / terminal ---
    def account_info(self):
        return SimpleNamespace(login=12345678, server="XMGlobal-MT5 10", company="XM Global",
                               trade_mode=self.trade_mode, margin_mode=2, currency="USD",
                               balance=10000.0, equity=10000.0, leverage=1000)

    def terminal_info(self):
        return SimpleNamespace(name="XM Global MT5", connected=True)

    # --- symbols ---
    def symbol_info(self, symbol):
        spec = SYMBOLS.get(symbol)
        return SimpleNamespace(**spec) if spec else None

    def symbol_select(self, symbol, enable=True):
        return symbol in SYMBOLS

    def symbol_info_tick(self, symbol):
        if symbol not in SYMBOLS:
            return None
        return SimpleNamespace(time=self.tick_time or 0, bid=1.0, ask=1.0)

    # --- account history (read-only) ---
    def history_deals_get(self, date_from, date_to, **kwargs):
        return tuple(self.deals)

    def history_orders_get(self, date_from, date_to, **kwargs):
        return tuple(self.orders)

    # --- candles ---
    def all_bars(self, symbol, timeframe):
        """Every candle on the 'server', oldest first. Deterministic random walk."""
        step = TF_SECONDS[self._TF_NAMES[timeframe]]
        base = SYMBOLS[symbol]["base"]
        rng = np.random.default_rng(42)
        closes = base * (1 + np.cumsum(rng.normal(0, 0.001, self.n_bars)))
        opens = np.r_[base, closes[:-1]]
        bars = np.zeros(self.n_bars, dtype=RATES_DTYPE)
        bars["time"] = self.start_server + step * np.arange(self.n_bars)
        bars["open"], bars["close"] = opens, closes
        bars["high"] = np.maximum(opens, closes) * 1.0005
        bars["low"] = np.minimum(opens, closes) * 0.9995
        bars["tick_volume"] = 100
        bars["spread"] = SYMBOLS[symbol]["spread"]
        return bars

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        """Like MT5: position 0 = newest (still forming) candle, counting backwards."""
        if symbol not in SYMBOLS:
            return None
        bars = self.all_bars(symbol, timeframe)
        end = len(bars) - start_pos
        return bars[max(0, end - count):end]


@pytest.fixture
def fake_mt5():
    return FakeMT5()


def as_rulebook(c: dict) -> dict:
    """Switch off everything that came from research, so tests check the plain
    rulebook logic and do not change every time the tuning changes."""
    c["stops"]["min_risk_atr_mult"] = 0.0
    c["stops"]["tp_mode"] = "next_level"
    c["filters"] = {"level": {}, "retest": {}, "h2l2": {}}
    return c


@pytest.fixture
def tuned_cfg(tmp_path):
    """config.yaml exactly as shipped (with the researched settings)."""
    c = load_config()
    c["data"]["cache_dir"] = str(tmp_path / "cache")
    c["data"]["bars"] = {"D1": 100, "H4": 100, "H1": 100}
    return c


@pytest.fixture
def cfg(tuned_cfg):
    """config.yaml in rulebook mode: temporary cache folder, small bar counts."""
    return as_rulebook(tuned_cfg)


@pytest.fixture
def cfg_file(tmp_path) -> Path:
    """A config.yaml written to a temp folder (for CLI tests that load from a path)."""
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["data"]["bars"] = {"D1": 100, "H4": 100, "H1": 100}
    raw["backtest"]["min_history_bars"] = {"H4": 20, "D1": 3}   # the fake history is short
    as_rulebook(raw)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path
