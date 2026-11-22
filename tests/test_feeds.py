"""The cloud price feed (Yahoo) and the economic calendar. No real network is used:
each test hands the code a fake download."""

import io
import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from market_analyst import news
from market_analyst.data import utc_to_server
from market_analyst.yahoo import FeedError, clean_hourly, fetch_yahoo, parse_chart


def ts(text: str) -> int:
    return int(pd.Timestamp(text, tz="UTC").timestamp())


def fake_opener(payload):
    """Stands in for urllib.request.urlopen: returns the payload as a JSON 'response'."""
    calls = []

    def opener(req, timeout=0):
        calls.append(req.full_url)
        return io.BytesIO(json.dumps(payload).encode())
    opener.calls = calls
    return opener


# --- price feed -------------------------------------------------------------------

# November 2026 is winter time: server = UTC+2.
NOW = datetime(2026, 11, 9, 3, 30, tzinfo=timezone.utc)      # Monday 03:30 UTC
ROWS = [  # (time UTC, open, high, low, close)
    ("2026-11-06 19:00", None, None, None, None),            # empty row -> dropped
    ("2026-11-06 20:00", 10, 12, 9, 11),                     # Friday, server 22:00
    ("2026-11-06 21:00", 11, 13, 10, 12),                    # Friday, server 23:00
    ("2026-11-07 10:00", 50, 50, 50, 50),                    # Saturday -> dropped
    ("2026-11-08 22:00", 20, 21, 19, 20.5),                  # server Monday 00:00
    ("2026-11-08 23:00", 20.5, 25, 20, 24),
    ("2026-11-09 00:00", 24, 24.5, 18, 19),
    ("2026-11-09 01:00", 19, 22, 18.5, 21),
    ("2026-11-09 02:00", 21, 23, 20, 22),                    # next H4 block, not finished
    ("2026-11-09 03:00", 22, 22.5, 21, 22.2),                # still forming -> dropped
    ("2026-11-09 03:25:31", 22.2, 22.2, 22.2, 22.2),         # live tick row -> dropped
]
PAYLOAD = {"chart": {"result": [{
    "timestamp": [ts(r[0]) for r in ROWS],
    "indicators": {"quote": [{
        "open": [r[1] for r in ROWS], "high": [r[2] for r in ROWS],
        "low": [r[3] for r in ROWS], "close": [r[4] for r in ROWS],
        "volume": [0 if r[1] is not None else None for r in ROWS],
    }]},
}]}}


def test_clean_keeps_only_closed_weekday_hourly_candles(cfg):
    h1 = clean_hourly(parse_chart(PAYLOAD), cfg, NOW)
    assert [t.strftime("%d %H:%M") for t in h1.index] == [
        "06 20:00", "06 21:00", "08 22:00", "08 23:00", "09 00:00", "09 01:00", "09 02:00"]


def test_utc_to_server_is_the_reverse_of_server_to_utc(cfg):
    idx = pd.DatetimeIndex(["2026-11-08 22:00", "2026-07-20 13:00"], tz="UTC")
    server = utc_to_server(idx, cfg)
    assert server.tolist() == [pd.Timestamp("2026-11-09 00:00"),      # winter: +2
                               pd.Timestamp("2026-07-20 16:00")]      # summer: +3


def test_h4_candles_start_on_broker_server_time(cfg):
    frames = fetch_yahoo("GOLD", cfg, NOW, fake_opener(PAYLOAD))
    h4 = frames["H4"]
    # Friday's last block (2 hourly candles) and Monday's first full block.
    # The block that started at server 04:00 is not finished at 03:30 UTC: dropped.
    assert h4["time_server"].tolist() == [pd.Timestamp("2026-11-06 20:00"),
                                          pd.Timestamp("2026-11-09 00:00")]
    monday = h4.iloc[-1]
    assert (monday["open"], monday["high"], monday["low"], monday["close"]) == (20, 25, 18, 21)
    assert h4.index[-1] == pd.Timestamp("2026-11-08 22:00", tz="UTC")


def test_d1_drops_the_unfinished_day(cfg):
    d1 = fetch_yahoo("GOLD", cfg, NOW, fake_opener(PAYLOAD))["D1"]
    assert d1["time_server"].tolist() == [pd.Timestamp("2026-11-06")]
    assert (d1.iloc[0]["open"], d1.iloc[0]["close"]) == (10, 12)


def test_frames_have_the_same_columns_as_mt5_data(cfg):
    frames = fetch_yahoo("GOLD", cfg, NOW, fake_opener(PAYLOAD))
    for df in frames.values():
        assert list(df.columns) == ["time_server", "open", "high", "low", "close",
                                    "tick_volume", "spread", "real_volume"]
        assert str(df.index.tz) == "UTC"


def test_feed_uses_the_configured_symbol(cfg):
    opener = fake_opener(PAYLOAD)
    fetch_yahoo("GOLD", cfg, NOW, opener)
    assert "GC%3DF" in opener.calls[0]                       # GC=F, URL-encoded


def test_bad_answer_is_a_clear_error(cfg):
    bad = {"chart": {"result": None, "error": {"code": "Not Found"}}}
    with pytest.raises(FeedError, match="Unexpected answer"):
        fetch_yahoo("GOLD", cfg, NOW, fake_opener(bad))


# --- economic calendar ------------------------------------------------------------

CALENDAR = [
    {"title": "CPI m/m", "country": "USD", "date": "2026-11-10T08:30:00-05:00",
     "impact": "High", "forecast": "0.3%", "previous": "0.6%"},
    {"title": "German ZEW", "country": "EUR", "date": "2026-11-10T05:00:00-05:00",
     "impact": "Medium", "forecast": "", "previous": ""},
    {"title": "ECB President Speaks", "country": "EUR", "date": "2026-11-12T04:00:00-05:00",
     "impact": "High", "forecast": "", "previous": ""},
    {"title": "Cash Rate", "country": "AUD", "date": "2026-11-11T22:30:00-05:00",
     "impact": "High", "forecast": "", "previous": ""},
    {"title": "broken row", "country": "USD", "date": "not a date", "impact": "High"},
]
MONDAY = datetime(2026, 11, 9, 22, 0, tzinfo=timezone.utc)


def test_events_are_converted_to_utc_and_bad_rows_skipped():
    events = news.parse_events(CALENDAR)
    assert len(events) == 4
    cpi = next(e for e in events if e["title"] == "CPI m/m")
    assert cpi["time_utc"] == datetime(2026, 11, 10, 13, 30, tzinfo=timezone.utc)


def test_gold_only_shows_high_impact_usd_news(cfg):
    events = news.events_for_symbol(news.parse_events(CALENDAR), "GOLD", cfg)
    assert [e["title"] for e in events] == ["CPI m/m"]


def test_eurusd_shows_both_currencies(cfg):
    events = news.events_for_symbol(news.parse_events(CALENDAR), "EURUSD", cfg)
    assert [e["title"] for e in events] == ["CPI m/m", "ECB President Speaks"]


def test_upcoming_window():
    events = news.parse_events(CALENDAR)
    assert [e["title"] for e in news.upcoming(events, MONDAY, 24)] == ["German ZEW", "CPI m/m"]


def test_describe_shows_time_left():
    cpi = news.parse_events(CALENDAR)[1]
    text = news.describe(cpi, MONDAY)
    assert "Tue 10 Nov 13:30 UTC" in text and "USD" in text and "[in 15.5 h]" in text


def test_calendar_is_downloaded_once_then_read_from_cache(cfg):
    opener = fake_opener(CALENDAR)
    first = news.load_calendar(cfg, opener)
    second = news.load_calendar(cfg, opener)
    assert first == second == CALENDAR
    assert len(opener.calls) == 1
