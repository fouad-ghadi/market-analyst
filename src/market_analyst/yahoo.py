"""Cloud data source: free hourly candles from Yahoo Finance.

Why this exists: the MetaTrader5 package only runs on Windows next to your
terminal. In the cloud we still need candles to analyse, so this module
downloads public hourly data and reshapes it to look exactly like MT5 data
(same columns, same candle start times).

Honest limits (also printed by the CLI):
  * It is NOT your broker's price feed. EURUSD is very close. GOLD uses COMEX
    gold futures (GC=F), which trade a few dollars away from XM's spot GOLD.
  * Yahoo's hourly history has some missing hours and goes back 730 days at most.
  * The endpoint is unofficial and may change.
On your PC, MT5 stays the reference (data.source: auto picks it automatically).

Steps:
  1. download_hourly  : fetch the JSON
  2. parse_chart      : JSON -> H1 table indexed by UTC time
  3. clean_hourly     : drop empty rows, the live "tick" row, the candle still forming
  4. resample         : build H4 and D1 from H1, aligned on broker server time
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime

import pandas as pd

from .data import _normalize_times, server_to_utc, utc_now, utc_to_server

URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1h&range={rng}"
TF_HOURS = {"H1": 1, "H4": 4, "D1": 24}


class FeedError(RuntimeError):
    """Raised when the public data feed cannot be read."""


def download_hourly(yahoo_symbol: str, rng: str, opener=None) -> dict:
    """Fetch the raw JSON. `opener` lets the tests replace the real download."""
    opener = opener or urllib.request.urlopen
    url = URL.format(symbol=urllib.request.quote(yahoo_symbol), rng=rng)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (market-analyst)"})
    try:
        with opener(req, timeout=45) as resp:
            return json.load(resp)
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise FeedError(f"Could not download {yahoo_symbol} from Yahoo: {exc}") from exc


def parse_chart(payload: dict) -> pd.DataFrame:
    """Yahoo chart JSON -> DataFrame(open, high, low, close, tick_volume), UTC index."""
    try:
        result = payload["chart"]["result"][0]
        quote = result["indicators"]["quote"][0]
        df = pd.DataFrame(
            {
                "open": quote["open"], "high": quote["high"],
                "low": quote["low"], "close": quote["close"],
                "tick_volume": quote.get("volume") or [0] * len(result["timestamp"]),
            },
            index=pd.to_datetime(result["timestamp"], unit="s", utc=True),
        )
    except (KeyError, TypeError, IndexError) as exc:
        raise FeedError(f"Unexpected answer from Yahoo: {str(payload)[:200]}") from exc
    df.index.name = "time_utc"
    return df


def clean_hourly(df: pd.DataFrame, cfg: dict, now_utc: datetime) -> pd.DataFrame:
    """Keep only complete, real hourly candles."""
    df = df.dropna(subset=["open", "high", "low", "close"])
    # Yahoo appends one row for "the price right now" with an odd timestamp
    # (e.g. 21:25:31). Real hourly candles start exactly on the hour.
    df = df[(df.index.minute == 0) & (df.index.second == 0)]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # The candle that started less than one hour ago is still forming: drop it
    # (analysing a candle before it closes is look-ahead bias).
    df = df[df.index + pd.Timedelta(hours=1) <= pd.Timestamp(now_utc)]
    # Your broker has no weekend candles (server time). Drop any stray ones.
    server = utc_to_server(df.index, cfg)
    return df[server.day_of_week < 5]


def resample(h1: pd.DataFrame, tf: str, cfg: dict, now_utc: datetime) -> pd.DataFrame:
    """Build H4 or D1 candles from H1 candles, on broker server-time boundaries.

    Each H1 candle is put in a bucket: the 4-hour block (or the day) of server
    time it belongs to. Inside a bucket: open = first open, high = highest
    high, low = lowest low, close = last close.
    """
    hours = TF_HOURS[tf]
    server = utc_to_server(h1.index, cfg)
    bucket = server.floor(f"{hours}h") if hours < 24 else server.normalize()
    grouped = h1.groupby(bucket)
    out = pd.DataFrame({
        "open": grouped["open"].first(),
        "high": grouped["high"].max(),
        "low": grouped["low"].min(),
        "close": grouped["close"].last(),
        "tick_volume": grouped["tick_volume"].sum(),
    })
    out["time_server"] = out.index
    out["time_utc"] = server_to_utc(pd.Series(out.index), cfg).array
    out["time_utc"] = pd.to_datetime(out["time_utc"], utc=True)
    out = out.set_index("time_utc").sort_index()
    # Drop the bucket that is not finished yet.
    out = out[out.index + pd.Timedelta(hours=hours) <= pd.Timestamp(now_utc)]
    return _finish(out)


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    """Same column layout as MT5 frames."""
    df = df.copy()
    df["spread"] = 0          # the public feed has no spread information
    df["real_volume"] = 0
    cols = ["time_server", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    return _normalize_times(df[cols])


def fetch_yahoo(symbol: str, cfg: dict, now_utc: datetime | None = None,
                opener=None) -> dict[str, pd.DataFrame]:
    """Download one symbol and return {'H1': df, 'H4': df, 'D1': df}."""
    ycfg = cfg["data"]["yahoo"]
    if symbol not in ycfg["symbols"]:
        raise FeedError(f"No Yahoo symbol configured for {symbol} (data.yahoo.symbols)")
    now_utc = now_utc or utc_now()
    payload = download_hourly(ycfg["symbols"][symbol], ycfg["hourly_range"], opener)
    h1 = clean_hourly(parse_chart(payload), cfg, now_utc)
    if h1.empty:
        raise FeedError(f"Yahoo returned no usable candles for {symbol}")
    h1_out = h1.copy()
    h1_out["time_server"] = utc_to_server(h1.index, cfg)
    return {
        "H1": _finish(h1_out),
        "H4": resample(h1, "H4", cfg, now_utc),
        "D1": resample(h1, "D1", cfg, now_utc),
    }
