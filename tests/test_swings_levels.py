"""ATR, swing points and S/R zones."""

import copy

import numpy as np
import pandas as pd
import pytest

from market_analyst.indicators import atr, last_atr, true_range
from market_analyst.levels import (
    build_zones,
    clear_reactions,
    combine_timeframes,
    nearest_zones,
    role_for,
)
from market_analyst.swings import find_swings
from tests.helpers import candles, from_highs_lows, path


# --- indicators -------------------------------------------------------------------

def test_true_range_includes_the_gap_from_previous_close():
    df = candles([(10, 11, 9, 10), (13, 14, 12.5, 13)])   # gap up: prev close 10, low 12.5
    assert true_range(df).tolist() == [2.0, 4.0]          # 14 - 10 = 4, not 14 - 12.5


def test_atr_is_the_average_candle_size_and_uses_no_future_data():
    df = path([100, 110, 100, 110], bars_per_leg=10)
    full = atr(df, 5)
    cut = atr(df.iloc[:20], 5)
    assert full.iloc[:20].equals(cut)                     # adding future candles changes nothing
    assert last_atr(df, 5) == pytest.approx(1.1)          # each candle spans 1.0 + two 0.05 wicks


# --- swings ----------------------------------------------------------------------

def test_swing_high_and_low_are_found_on_wick_tips():
    highs = [1, 2, 3, 9, 3, 2, 1, 2, 3, 2, 1]
    lows = [0, 1, 2, 3, 2, 1, -5, 1, 2, 1, 0]
    sw = find_swings(from_highs_lows(highs, lows), n=3)
    assert sw[["kind", "price", "pos", "confirmed_pos"]].values.tolist() == [
        ["high", 9.0, 3, 6], ["low", -5.0, 6, 9]]


def test_last_n_candles_are_never_swings():
    highs = [1, 2, 3, 4, 5, 6, 9]          # the highest candle is the last one
    sw = find_swings(from_highs_lows(highs, [h - 1 for h in highs]), n=3)
    assert sw.empty


def test_equal_highs_give_one_swing_not_two():
    highs = [1, 2, 3, 9, 9, 3, 2, 1, 0]
    sw = find_swings(from_highs_lows(highs, [h - 10 for h in highs]), n=3)
    assert sw[sw["kind"] == "high"]["pos"].tolist() == [3]


def test_no_look_ahead_swings_known_at_bar_t_never_change_later():
    """The key guarantee: cut the data at any candle t, and the swings found are
    exactly the swings of the full data that were confirmed by t."""
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 1, 400))
    df = candles([(c, c + abs(rng.normal(0, .6)), c - abs(rng.normal(0, .6)), c) for c in close])
    full = find_swings(df, 3)
    for t in (50, 123, 250, 399):
        seen_at_t = find_swings(df.iloc[:t + 1], 3)
        expected = full[full["confirmed_pos"] <= t]
        assert seen_at_t[["kind", "pos", "price"]].values.tolist() == \
            expected[["kind", "pos", "price"]].values.tolist()


# --- zones ------------------------------------------------------------------------

@pytest.fixture
def range_df():
    """Price bounces twice between about 100 and about 110, and ends at 105."""
    return path([105, 110, 100, 110.1, 100.1, 105])


def test_two_touches_make_a_validated_zone(range_df, cfg):
    zones = build_zones(range_df, cfg, "H4")
    validated = [z for z in zones if z.status == "validated"]
    assert len(validated) == 2
    support, resistance = validated
    assert (support.role, support.touches, support.touch_lows) == ("support", 2, 2)
    assert (resistance.role, resistance.touches, resistance.touch_highs) == ("resistance", 2, 2)
    # Built on wick tips: lows 99.95 and 100.05, highs 110.05 and 110.15.
    assert support.low <= 99.95 and support.high >= 100.05
    assert resistance.low <= 110.05 and resistance.high >= 110.15


def test_one_touch_is_only_a_level_to_watch(cfg):
    df = path([105, 110, 100, 104])           # one high, one low
    statuses = {z.status for z in build_zones(df, cfg, "H4")}
    assert statuses == {"watch"}


def test_zone_is_never_thinner_than_the_tolerance(cfg):
    df = path([105, 110, 100, 104])
    tolerance = cfg["levels"]["tolerance_atr_mult"] * last_atr(df, cfg["atr"]["period"])
    for z in build_zones(df, cfg, "H4"):
        assert z.high - z.low == pytest.approx(tolerance)


def test_small_wiggles_are_not_touches(cfg):
    """A swing with no clear reaction must not count as a touch."""
    # Up to 110, a small dip to 109.2, back to 110.1, then a real fall to 100.
    df = path([100, 110, 109.2, 110.1, 100])
    sw = find_swings(df, 3)
    assert sw["pos"].tolist() == [9, 19, 29]          # high, small low, high
    kept = clear_reactions(df, sw, cfg)
    # Only the last high was followed by a big move. The first high and the
    # small dip moved less than 1 point: no clear reaction.
    assert kept["pos"].tolist() == [29]


def test_reaction_filter_can_be_switched_off(range_df, cfg):
    c = copy.deepcopy(cfg)
    c["levels"]["min_reaction_atr_mult"] = 0
    sw = find_swings(range_df, 3)
    assert len(clear_reactions(range_df, sw, c)) == len(sw)


def test_old_swings_outside_lookback_are_ignored(range_df, cfg):
    c = copy.deepcopy(cfg)
    c["levels"]["lookback_bars"] = 12          # only the last leg is visible
    assert all(z.status == "watch" for z in build_zones(range_df, c, "H4"))


def test_recent_touches_score_higher_than_old_ones(cfg):
    old = build_zones(path([105, 110, 100, 110.1, 100.1, 105, 105, 105, 105]), cfg, "H4")
    new = build_zones(path([105, 110, 100, 110.1, 100.1, 105]), cfg, "H4")
    assert max(z.score for z in new) > max(z.score for z in old)


def test_higher_timeframe_zones_score_higher(range_df, cfg):
    h4 = build_zones(range_df, cfg, "H4")
    d1 = build_zones(range_df, cfg, "D1")
    assert d1[0].score - h4[0].score == pytest.approx(
        cfg["levels"]["timeframe_weight"]["D1"] - cfg["levels"]["timeframe_weight"]["H4"])


def test_role_reversal_follows_the_last_close():
    assert role_for(100, 101, last_close=103) == "support"       # closed above: support
    assert role_for(100, 101, last_close=98) == "resistance"     # closed below: resistance
    assert role_for(100, 101, last_close=100.5) == "inside"


def test_broken_resistance_becomes_support_and_is_flagged(cfg):
    # Two highs near 110, then price breaks out and closes at 115.
    df = path([105, 110, 104, 110.1, 104, 115])
    top = [z for z in build_zones(df, cfg, "H4") if z.touch_highs == 2][0]
    assert top.role == "support"
    assert top.untested_flip                   # not retested from above yet


def test_combining_timeframes_keeps_one_zone_and_notes_the_confirmation(range_df, cfg):
    h4, d1 = build_zones(range_df, cfg, "H4"), build_zones(range_df, cfg, "D1")
    combined = combine_timeframes({"H4": h4, "D1": d1}, cfg)
    assert len(combined) == len(d1)            # the same areas, not duplicated
    assert all(z.timeframe == "D1" and z.confirmed_by == ["H4"] for z in combined)
    assert combined[0].touches == 2            # touches are not double counted


def test_nearest_zones(range_df, cfg):
    zones = build_zones(range_df, cfg, "H4")
    support, inside, resistance = nearest_zones(zones, 105.0)
    assert support.role == "support" and resistance.role == "resistance" and inside is None
    assert nearest_zones(zones, 100.0)[1] is not None          # price inside the support zone
