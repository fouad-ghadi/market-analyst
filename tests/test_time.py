"""Server time -> UTC conversion. Getting this wrong would shift every session filter."""

from datetime import datetime, timezone

import pandas as pd

from market_analyst.data import detect_server_offset, expected_server_offset_hours, server_to_utc
from tests.conftest import FakeMT5


def _utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


def test_new_york_close_summer_is_utc_plus_3(cfg):
    s = pd.Series(pd.to_datetime(["2026-07-20 16:00"]))
    assert server_to_utc(s, cfg).iloc[0] == pd.Timestamp("2026-07-20 13:00", tz="UTC")


def test_new_york_close_winter_is_utc_plus_2(cfg):
    s = pd.Series(pd.to_datetime(["2026-12-15 16:00"]))
    assert server_to_utc(s, cfg).iloc[0] == pd.Timestamp("2026-12-15 14:00", tz="UTC")


def test_daily_candle_opens_at_new_york_17h(cfg):
    # A D1 candle stamped 00:00 server time opens at 17:00 New York = 21:00 UTC in summer.
    s = pd.Series(pd.to_datetime(["2026-07-21 00:00"]))
    assert server_to_utc(s, cfg).iloc[0] == pd.Timestamp("2026-07-20 21:00", tz="UTC")


def test_offset_follows_us_daylight_saving(cfg):
    # Europe switched on 25 Oct 2026, the US switches on 1 Nov 2026.
    assert expected_server_offset_hours(cfg, _utc(2026, 10, 30, 12)) == 3
    assert expected_server_offset_hours(cfg, _utc(2026, 11, 2, 12)) == 2


def test_fixed_mode(cfg):
    cfg["time"]["server_time_mode"] = "fixed"
    cfg["time"]["server_utc_offset_hours"] = 3
    s = pd.Series(pd.to_datetime(["2026-12-15 16:00"]))
    assert server_to_utc(s, cfg).iloc[0] == pd.Timestamp("2026-12-15 13:00", tz="UTC")
    assert expected_server_offset_hours(cfg, _utc(2026, 12, 15)) == 3


def test_detect_offset_with_fresh_tick():
    now = _utc(2026, 11, 3, 14, 0)
    fake = FakeMT5(tick_time=int(now.timestamp()) + 2 * 3600 - 5)  # server UTC+2, tick 5 s old
    m = detect_server_offset(fake, "GOLD", now)
    assert m == {"measured_offset_hours": 2, "tick_age_seconds": 5, "fresh": True}


def test_detect_offset_on_weekend_is_not_fresh():
    now = _utc(2026, 11, 1, 12, 0)  # Sunday
    friday_close = _utc(2026, 10, 30, 20, 58)
    fake = FakeMT5(tick_time=int(friday_close.timestamp()) + 3 * 3600)
    assert detect_server_offset(fake, "GOLD", now)["fresh"] is False
