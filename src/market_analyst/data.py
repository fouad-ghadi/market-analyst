"""MT5 data access: connection, candles, symbol specs, time zones, cache.

READ-ONLY. This module only *reads* from your MT5 terminal. It never calls
order_send (or anything that could place, modify or close a trade).
A test (tests/test_safety.py) fails if such a call ever appears in the code.

The flow for one symbol and timeframe:

    MT5 terminal --copy_rates_from_pos--> numpy array (times in SERVER time)
        --> pandas DataFrame --> add a UTC time column --> merge with CSV cache

Why the MetaTrader5 module is passed in as a parameter (`mt5`) instead of
imported at the top: the official package only exists on Windows. Passing it
in lets the tests use a fake MT5 on any computer, and lets this file be
imported on a machine without MT5 installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import cache_dir

NEW_YORK = ZoneInfo("America/New_York")
NY_CLOSE_SHIFT_HOURS = 7  # "new_york_close" convention: server time = New York time + 7 h

# MT5 constant values (from the MetaTrader5 docs), used to print readable names.
ACCOUNT_TRADE_MODES = {0: "DEMO", 1: "CONTEST", 2: "REAL"}
ACCOUNT_MARGIN_MODES = {0: "netting", 1: "exchange", 2: "hedging"}

PRICE_COLUMNS = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]


class MT5Error(RuntimeError):
    """Raised when the MT5 terminal refuses a request or is not reachable."""


# =============================================================================
# 1. Connection
# =============================================================================

def import_mt5():
    """Import the official MetaTrader5 package, with a helpful message if it fails."""
    try:
        import MetaTrader5 as mt5  # noqa: N813  (official name of the package)
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise MT5Error(
            "The 'MetaTrader5' package is not installed in this Python environment.\n"
            "It only works on Windows. Activate the conda env and run:\n"
            "    pip install MetaTrader5"
        ) from exc
    return mt5


class MT5Connection:
    """Context manager: connects on enter, always disconnects on exit.

    Usage:
        with MT5Connection(cfg) as mt5:
            ...  # use mt5 here

    `mt5.initialize()` attaches to the terminal that is already running and
    logged in. No login or password is stored anywhere in this project.
    """

    def __init__(self, cfg: dict, mt5_module: Any = None):
        self.cfg = cfg
        self.mt5 = mt5_module

    def __enter__(self):
        if self.mt5 is None:
            self.mt5 = import_mt5()
        mcfg = self.cfg.get("mt5", {})
        kwargs = {"timeout": int(mcfg.get("timeout_ms", 60000))}
        path = mcfg.get("terminal_path")
        ok = self.mt5.initialize(path, **kwargs) if path else self.mt5.initialize(**kwargs)
        if not ok:
            err = self.mt5.last_error()
            self.mt5.shutdown()
            raise MT5Error(
                f"Could not connect to MT5 (error {err}).\n"
                "Check that the MT5 terminal is open and logged in to your account. "
                "If several MT5 terminals are installed, set mt5.terminal_path in "
                "config.yaml to the right terminal64.exe."
            )
        return self.mt5

    def __exit__(self, exc_type, exc, tb):
        self.mt5.shutdown()
        return False  # never hide an exception


def account_summary(mt5) -> dict:
    """Read-only facts about the logged-in account (no password, nothing secret)."""
    info = mt5.account_info()
    if info is None:
        raise MT5Error(f"account_info() failed: {mt5.last_error()}")
    return {
        "login": info.login,
        "server": info.server,
        "company": getattr(info, "company", ""),
        "type": ACCOUNT_TRADE_MODES.get(info.trade_mode, f"unknown({info.trade_mode})"),
        "margin_mode": ACCOUNT_MARGIN_MODES.get(
            getattr(info, "margin_mode", -1), "unknown"
        ),
        "currency": info.currency,
        "balance": info.balance,
        "equity": info.equity,
        "leverage": info.leverage,
    }


def account_warnings(summary: dict, cfg: dict) -> list[str]:
    """Return warnings if the terminal is not on the expected demo account."""
    warnings = []
    mcfg = cfg.get("mt5", {})
    if mcfg.get("warn_if_not_demo", True) and summary["type"] != "DEMO":
        warnings.append(
            f"!!! This is a {summary['type']} account, not a DEMO account. "
            "The tool is read-only, but be careful."
        )
    expected = mcfg.get("expected_server")
    if expected and summary["server"] != expected:
        warnings.append(
            f"Terminal is logged in to '{summary['server']}', config expects '{expected}'."
        )
    return warnings


# =============================================================================
# 2. Symbol specifications
# =============================================================================

@dataclass
class SymbolSpec:
    """Everything we need about a symbol to compute pips, spreads and lot sizes."""

    name: str
    description: str
    digits: int            # decimals in the quote (EURUSD 5, GOLD 2 on XM)
    point: float           # smallest price step shown (EURUSD 0.00001)
    pip_size: float        # from config.yaml (EURUSD 0.0001)
    tick_size: float       # smallest price move that changes P/L
    tick_value: float      # P/L of 1 lot when price moves 1 tick, in account currency
    contract_size: float   # units per 1 lot (EURUSD 100000, GOLD 100 oz)
    volume_min: float
    volume_max: float
    volume_step: float
    spread_points: int     # current spread, in points
    stops_level_points: int  # minimum SL/TP distance imposed by the broker, in points
    currency_profit: str

    @property
    def spread_price(self) -> float:
        """Current spread expressed as a price distance."""
        return self.spread_points * self.point

    @property
    def spread_pips(self) -> float:
        return self.spread_price / self.pip_size

    @property
    def pip_value_per_lot(self) -> float:
        """Money made or lost by 1 lot when price moves 1 pip (account currency)."""
        return self.pip_size / self.tick_size * self.tick_value

    def as_dict(self) -> dict:
        d = asdict(self)
        d["spread_pips"] = round(self.spread_pips, 2)
        d["pip_value_per_lot"] = round(self.pip_value_per_lot, 4)
        return d


def ensure_symbol(mt5, symbol: str) -> None:
    """Make sure the symbol is visible in Market Watch (needed to get its data)."""
    if mt5.symbol_info(symbol) is None:
        raise MT5Error(
            f"Symbol '{symbol}' not found on this server. "
            "Check the exact name in MT5's Market Watch (XM uses 'GOLD', not 'XAUUSD')."
        )
    if not mt5.symbol_select(symbol, True):
        raise MT5Error(f"Could not add '{symbol}' to Market Watch: {mt5.last_error()}")


def get_symbol_spec(mt5, symbol: str, pip_size: float) -> SymbolSpec:
    """Read the symbol's contract details from the terminal."""
    ensure_symbol(mt5, symbol)
    s = mt5.symbol_info(symbol)
    return SymbolSpec(
        name=s.name,
        description=getattr(s, "description", ""),
        digits=s.digits,
        point=s.point,
        pip_size=pip_size,
        tick_size=s.trade_tick_size,
        tick_value=s.trade_tick_value,
        contract_size=s.trade_contract_size,
        volume_min=s.volume_min,
        volume_max=s.volume_max,
        volume_step=s.volume_step,
        spread_points=s.spread,
        stops_level_points=getattr(s, "trade_stops_level", 0),
        currency_profit=getattr(s, "currency_profit", ""),
    )


# =============================================================================
# 3. Time zones (server time -> UTC)
# =============================================================================

def expected_server_offset_hours(cfg: dict, at_utc: datetime) -> float:
    """Server-minus-UTC offset (hours) the config predicts at a given UTC moment."""
    tcfg = cfg["time"]
    if tcfg["server_time_mode"] == "fixed":
        return float(tcfg["server_utc_offset_hours"])
    ny_offset = at_utc.astimezone(NEW_YORK).utcoffset()  # -5 h in winter, -4 h in summer
    return ny_offset.total_seconds() / 3600 + NY_CLOSE_SHIFT_HOURS


def server_to_utc(server_times: pd.Series, cfg: dict) -> pd.Series:
    """Convert naive server timestamps to timezone-aware UTC timestamps.

    fixed mode:          utc = server - offset
    new_york_close mode: server - 7 h = New York wall-clock time -> convert to UTC.
                         This follows US daylight saving automatically, so old
                         candles get the right offset too (+2 in winter, +3 in summer).
    """
    tcfg = cfg["time"]
    s = pd.to_datetime(server_times)
    if tcfg["server_time_mode"] == "fixed":
        utc = s - pd.Timedelta(hours=tcfg["server_utc_offset_hours"])
        return utc.dt.tz_localize("UTC")

    ny_wall = s - pd.Timedelta(hours=NY_CLOSE_SHIFT_HOURS)
    # The only ambiguous/non-existent New York hours (daylight saving switches)
    # fall on Sunday morning, when the market is closed. We pick a safe default.
    localized = ny_wall.dt.tz_localize(
        NEW_YORK,
        ambiguous=np.zeros(len(ny_wall), dtype=bool),
        nonexistent="shift_forward",
    )
    return localized.dt.tz_convert("UTC")


def utc_to_server(utc_times: pd.DatetimeIndex, cfg: dict) -> pd.DatetimeIndex:
    """The reverse of server_to_utc: UTC timestamps -> naive server timestamps.

    Used to build H4 and D1 candles that start at the same moments as your
    broker's candles (H4 at 00:00, 04:00, ... server time; D1 at server midnight).
    """
    tcfg = cfg["time"]
    if tcfg["server_time_mode"] == "fixed":
        shifted = utc_times + pd.Timedelta(hours=tcfg["server_utc_offset_hours"])
        return shifted.tz_localize(None)
    ny_wall = utc_times.tz_convert(NEW_YORK).tz_localize(None)
    return ny_wall + pd.Timedelta(hours=NY_CLOSE_SHIFT_HOURS)


def detect_server_offset(mt5, symbol: str, now_utc: datetime | None = None) -> dict:
    """Measure the real server offset using the latest tick.

    Only reliable while the market is open: on a weekend the last tick is
    from Friday, so the result is marked as not fresh.
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise MT5Error(f"symbol_info_tick({symbol}) failed: {mt5.last_error()}")
    diff_s = tick.time - now_utc.timestamp()  # tick.time is in server time
    offset_h = round(diff_s / 3600)
    lag_s = abs(diff_s - offset_h * 3600)     # how old the last tick is (if offset is right)
    # Rounding alone is not enough: a tick from Friday (~40 h old) would round to
    # a "-37 h offset" with a tiny lag. So the offset must also be a real time
    # zone, and the market must be open.
    fresh = lag_s < 600 and -12 <= offset_h <= 14 and not market_probably_closed(now_utc)
    return {
        "measured_offset_hours": offset_h,
        "tick_age_seconds": int(lag_s) if fresh else None,
        "fresh": fresh,
    }


def market_probably_closed(now_utc: datetime) -> bool:
    """Forex/gold weekend: roughly Friday 22:00 UTC -> Sunday 21:00 UTC."""
    wd, h = now_utc.weekday(), now_utc.hour   # Monday = 0 ... Sunday = 6
    return wd == 5 or (wd == 4 and h >= 22) or (wd == 6 and h < 21)


# =============================================================================
# 4. Candles
# =============================================================================

def timeframe_constant(mt5, tf: str) -> int:
    """'H4' -> mt5.TIMEFRAME_H4."""
    try:
        return getattr(mt5, f"TIMEFRAME_{tf}")
    except AttributeError as exc:
        raise MT5Error(f"Unknown timeframe '{tf}'") from exc


def rates_to_frame(rates, cfg: dict) -> pd.DataFrame:
    """Turn the numpy array returned by MT5 into a clean DataFrame.

    Index  : time_utc (timezone-aware, UTC), the candle's OPEN time
    Columns: time_server, open, high, low, close, tick_volume, spread, real_volume
    """
    df = pd.DataFrame(rates)
    if df.empty:
        raise MT5Error("MT5 returned no candles")
    df["time_server"] = pd.to_datetime(df["time"], unit="s")  # seconds -> datetime (server)
    df["time_utc"] = server_to_utc(df["time_server"], cfg)
    df = df.drop(columns=["time"]).set_index("time_utc").sort_index()
    return _normalize_times(df[["time_server"] + [c for c in PRICE_COLUMNS if c in df.columns]])


def _normalize_times(df: pd.DataFrame) -> pd.DataFrame:
    """Use one datetime precision everywhere (pandas versions differ: s, us, ns).
    Without this, data read back from the CSV would not compare equal to fresh data."""
    df.index = df.index.astype("datetime64[ns, UTC]")
    df.index.name = "time_utc"
    df["time_server"] = df["time_server"].astype("datetime64[ns]")
    return df


def fetch_rates(mt5, symbol: str, tf: str, count: int, cfg: dict) -> pd.DataFrame:
    """Download the last `count` candles of `symbol` on timeframe `tf`.

    Position 0 in MT5 is the candle that is still forming (its close changes
    every tick). By default we start at position 1 so that only CLOSED candles
    come back: analysing a candle that hasn't closed yet is look-ahead bias.
    """
    include_forming = cfg["data"].get("include_forming_bar", False)
    start_pos = 0 if include_forming else 1
    rates = mt5.copy_rates_from_pos(symbol, timeframe_constant(mt5, tf), start_pos, count)
    if rates is None or len(rates) == 0:
        raise MT5Error(f"copy_rates_from_pos({symbol}, {tf}) failed: {mt5.last_error()}")
    return rates_to_frame(rates, cfg)


# =============================================================================
# 5. Cache on disk (CSV, one file per symbol + timeframe)
# =============================================================================

def resolve_source(cfg: dict) -> str:
    """'mt5' or 'yahoo'. In auto mode: MT5 if its Python package is installed."""
    source = cfg["data"].get("source", "auto")
    if source != "auto":
        return source
    import importlib.util
    return "mt5" if importlib.util.find_spec("MetaTrader5") else "yahoo"


def cache_path(cfg: dict, symbol: str, tf: str, source: str = "mt5") -> Path:
    """MT5 data -> GOLD_H4.csv. Other sources get their own file (GOLD_H4.yahoo.csv)
    so prices from different sources are never mixed in one history."""
    suffix = "" if source == "mt5" else f".{source}"
    return cache_dir(cfg) / f"{symbol}_{tf}{suffix}.csv"


def load_cache(cfg: dict, symbol: str, tf: str, source: str = "mt5") -> pd.DataFrame | None:
    """Read cached candles, or None if nothing is cached yet."""
    path = cache_path(cfg, symbol, tf, source)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df["time_server"] = pd.to_datetime(df["time_server"])
    return _normalize_times(df.set_index("time_utc").sort_index())


def merge_into_cache(cfg: dict, symbol: str, tf: str, new: pd.DataFrame,
                     source: str = "mt5") -> pd.DataFrame:
    """Add new candles to the cache file. Old history is kept, so the cache
    grows over time even though each download only gives a limited number of bars.
    If a candle exists in both, the newer download wins."""
    old = load_cache(cfg, symbol, tf, source)
    merged = new if old is None else pd.concat([old, new])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.to_csv(cache_path(cfg, symbol, tf, source), index_label="time_utc")
    return merged


def update_cache(mt5, cfg: dict, symbol: str, tf: str) -> pd.DataFrame:
    """Fetch the configured number of bars and merge them into the cache."""
    count = cfg["data"]["bars"][tf]
    fresh = fetch_rates(mt5, symbol, tf, count, cfg)
    return merge_into_cache(cfg, symbol, tf, fresh)


def summarize_frame(df: pd.DataFrame) -> dict:
    """Short description of a candle table, for printing."""
    return {
        "bars": len(df),
        "first_utc": df.index[0].strftime("%Y-%m-%d %H:%M"),
        "last_utc": df.index[-1].strftime("%Y-%m-%d %H:%M"),
        "last_close": float(df["close"].iloc[-1]),
    }


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "MT5Connection", "MT5Error", "SymbolSpec", "account_summary", "account_warnings",
    "get_symbol_spec", "fetch_rates", "update_cache", "load_cache", "merge_into_cache",
    "server_to_utc", "utc_to_server", "expected_server_offset_hours",
    "detect_server_offset", "resolve_source", "cache_path", "summarize_frame", "utc_now",
]
