"""Trend classification and candlestick patterns."""

import pandas as pd

from market_analyst import patterns
from market_analyst.levels import Zone
from market_analyst.structure import classify
from tests.helpers import candles, path


# --- structure ----------------------------------------------------------------------

def test_higher_highs_and_higher_lows_is_an_uptrend(cfg):
    s = classify(path([100, 110, 105, 115, 110, 120, 116]), cfg, "H4")
    assert (s.trend, s.highs_label, s.lows_label, s.broken) == ("up", "HH", "HL", False)


def test_lower_highs_and_lower_lows_is_a_downtrend(cfg):
    s = classify(path([120, 110, 115, 105, 110, 100, 104]), cfg, "H4")
    assert (s.trend, s.highs_label, s.lows_label) == ("down", "LH", "LL")


def test_mixed_swings_is_a_range(cfg):
    # Higher high (110.05 -> 110.15) but lower low (100.05 -> 99.95): no trend.
    s = classify(path([105, 110, 100.1, 110.1, 100, 105]), cfg, "H4")
    assert (s.trend, s.highs_label, s.lows_label) == ("range", "HH", "LL")


def test_not_enough_swings_is_unknown(cfg):
    assert classify(path([100, 110]), cfg, "H4").trend == "unknown"


def test_uptrend_is_in_question_after_a_lower_low(cfg):
    # Uptrend, then price falls below the last higher low (105 -> 103).
    s = classify(path([100, 110, 105, 115, 110, 120, 108.5]), cfg, "H4")
    assert s.trend == "up" and s.broken
    assert "lower low" in s.note


def test_downtrend_is_in_question_after_a_higher_high(cfg):
    s = classify(path([120, 110, 115, 105, 110, 100, 111.5]), cfg, "H4")
    assert s.trend == "down" and s.broken
    assert "higher high" in s.note


# --- pattern shapes -----------------------------------------------------------------

def shapes(rows, cfg) -> dict:
    """Patterns flagged on the LAST candle of `rows`."""
    flags = patterns.detect(candles(rows), cfg).iloc[-1]
    return {name for name, on in flags.items() if on}


def test_doji(cfg):
    assert "doji" in shapes([(100, 101, 99, 100.05)], cfg)           # tiny body
    assert "doji" not in shapes([(100, 101, 99, 100.8)], cfg)        # real body


def test_hammer(cfg):
    assert "hammer" in shapes([(100.4, 100.52, 100.0, 100.5)], cfg)  # long lower wick
    assert "hammer" not in shapes([(100.4, 101.0, 100.0, 100.5)], cfg)   # big upper wick too
    assert "hammer" not in shapes([(100.0, 100.52, 99.9, 100.5)], cfg)   # mostly body


def test_shooting_star(cfg):
    assert "shooting_star" in shapes([(100.1, 100.5, 99.98, 100.0)], cfg)
    assert "shooting_star" not in shapes([(100.4, 100.52, 100.0, 100.5)], cfg)   # that is a hammer


def test_bullish_engulfing(cfg):
    assert "bullish_engulfing" in shapes([(101, 101.1, 100.4, 100.5), (100.4, 101.6, 100.3, 101.5)], cfg)
    # second candle does not cover the first body
    assert "bullish_engulfing" not in shapes([(101, 101.1, 100.4, 100.5), (100.4, 100.9, 100.3, 100.8)], cfg)
    # first candle was not bearish
    assert "bullish_engulfing" not in shapes([(100.5, 101.1, 100.4, 101), (100.4, 101.6, 100.3, 101.5)], cfg)


def test_bearish_engulfing(cfg):
    assert "bearish_engulfing" in shapes([(100.5, 101.1, 100.4, 101), (101.1, 101.2, 100.2, 100.3)], cfg)


def test_patterns_use_only_past_candles(cfg):
    df = path([100, 110, 100, 110], bars_per_leg=10)
    full = patterns.detect(df, cfg)
    assert full.iloc[:15].equals(patterns.detect(df.iloc[:15], cfg))


# --- a pattern only matters at a level ----------------------------------------------

def zone(low, high, role="support") -> Zone:
    t = pd.Timestamp("2026-03-01", tz="UTC")
    return Zone(low=low, high=high, timeframe="H4", touches=2, touch_highs=0, touch_lows=2,
                first_touch=t, last_touch=t, bars_since_last_touch=5, score=5.0,
                status="validated", role=role)


HAMMER = (100.4, 100.52, 100.0, 100.5)
STAR = (100.1, 100.5, 99.98, 100.0)
FILLER = [(100.4, 100.6, 100.3, 100.5)] * 3


def test_hammer_at_support_is_a_bullish_signal(cfg):
    df = candles(FILLER + [HAMMER])
    found = patterns.signals_at_zones(df, [zone(99.8, 100.1)], atr_value=0.5, cfg=cfg)
    assert [(s["pattern"], s["direction"], s["bars_ago"]) for s in found] == [("hammer", "bullish", 0)]


def test_hammer_in_the_middle_of_nowhere_is_ignored(cfg):
    df = candles(FILLER + [HAMMER])
    assert patterns.signals_at_zones(df, [zone(95.0, 95.3)], atr_value=0.5, cfg=cfg) == []


def test_hammer_that_closes_below_the_zone_is_not_support(cfg):
    df = candles(FILLER + [HAMMER])
    assert patterns.signals_at_zones(df, [zone(101.0, 101.3)], atr_value=0.5, cfg=cfg) == []


def test_shooting_star_at_resistance_is_a_bearish_signal(cfg):
    df = candles(FILLER + [STAR])
    found = patterns.signals_at_zones(df, [zone(100.4, 100.7, "resistance")], 0.5, cfg)
    assert [(s["pattern"], s["direction"]) for s in found] == [("shooting_star", "bearish")]


def test_only_recent_candles_are_scanned(cfg):
    old_hammer = candles([HAMMER] + FILLER * 2)          # hammer is 6 candles back
    assert patterns.signals_at_zones(old_hammer, [zone(99.8, 100.1)], 0.5, cfg) == []
