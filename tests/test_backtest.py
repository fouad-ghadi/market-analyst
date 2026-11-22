"""Planner, filters, the replay engine and the strategies."""

import copy

import numpy as np
import pandas as pd
import pytest

from market_analyst import research
from market_analyst.analysis import analyze
from market_analyst.backtest import (
    History, Order, Strategy, check_exit, run, summarize, try_fill)
from market_analyst.filters import failed_conditions, make_accept
from market_analyst.levels import Zone
from market_analyst.planner import plan_at_level, plan_fixed_rr
from market_analyst.strategies import STRATEGIES
from market_analyst.structure import classify, trend_series
from market_analyst.swings import find_swings
from tests.helpers import candles, path


def zone(low, high):
    t = pd.Timestamp("2026-03-01", tz="UTC")
    return Zone(low, high, "H4", 2, 0, 2, t, t, 5, 5.0, "validated", "support")


# --- planner ------------------------------------------------------------------------

def test_buy_plan_puts_stop_beyond_level_and_target_before_next_level(cfg):
    plan, why = plan_at_level("buy", 100.5, zone(99.8, 100.1), zone(110, 110.3),
                              atr_value=1.0, spread=0.02, cfg=cfg)
    assert why == ""
    assert plan.entry == pytest.approx(100.52)            # a buy pays the spread
    assert plan.sl == pytest.approx(99.8 - 0.1)           # buffer = max(2 x 0.02, 0.1 x 1.0)
    assert plan.tp == pytest.approx(110 - 0.1)
    assert plan.rr == pytest.approx((109.9 - 100.52) / (100.52 - 99.7))


def test_sell_plan_is_the_mirror(cfg):
    plan, _ = plan_at_level("sell", 109.5, zone(110, 110.3), zone(99.8, 100.1), 1.0, 0.02, cfg)
    assert (plan.entry, plan.sl, plan.tp) == pytest.approx((109.5, 110.4, 100.2))


def test_plan_is_refused_below_minimum_risk_reward(cfg):
    plan, why = plan_at_level("buy", 100.5, zone(99.8, 100.1), zone(101, 101.3), 1.0, 0.02, cfg)
    assert plan is None and "below the 1:2 minimum" in why


def test_plan_is_refused_without_a_target(cfg):
    plan, why = plan_at_level("buy", 100.5, zone(99.8, 100.1), None, 1.0, 0.02, cfg)
    assert plan is None and "no validated opposite level" in why


def test_minimum_stop_distance_and_fixed_target(tuned_cfg):
    # Shipped settings: stop at least 1 x ATR away, target fixed at 3 x the risk.
    plan, _ = plan_at_level("buy", 100.5, zone(99.8, 100.1), None, atr_value=2.0,
                            spread=0.02, cfg=tuned_cfg)
    assert plan.sl == pytest.approx(100.52 - 2.0)         # wider than the zone-based stop
    assert plan.rr == pytest.approx(3.0)


def test_fixed_rr_plan():
    plan = plan_fixed_rr("sell", 100.0, 101.0, 2.0)
    assert (plan.tp, plan.rr) == (98.0, 2.0)


# --- filters ------------------------------------------------------------------------

def test_filter_conditions():
    spec = {"session": ["london"], "min_zone_touches": 3, "max_atr_ratio": 1.3, "zone_d1": True}
    good = {"session": "london", "zone_touches": 3, "atr_ratio": 1.0, "zone_d1": True}
    assert failed_conditions(spec, good) == []
    bad = {"session": "asia", "zone_touches": 2, "atr_ratio": 2.0, "zone_d1": False}
    assert len(failed_conditions(spec, bad)) == 4
    assert make_accept({}) is None and make_accept(spec)(good) and not make_accept(spec)(bad)
    assert failed_conditions({"session": ["london"]}, {}) != []      # missing fact fails


# --- order fills and exits ----------------------------------------------------------

def test_fills_include_the_spread():
    assert try_fill(Order("market", "buy", 0, 0), 100, 101, 99, 0.2) == 100.2
    assert try_fill(Order("market", "sell", 0, 0), 100, 101, 99, 0.2) == 100
    assert try_fill(Order("limit", "buy", 0, 0, price=99.5), 100, 101, 99, 0.2) == 99.5
    assert try_fill(Order("limit", "buy", 0, 0, price=99.1), 100, 101, 99, 0.2) is None
    assert try_fill(Order("stop", "buy", 0, 0, price=101.1), 100, 101, 99, 0.2) == 101.1
    assert try_fill(Order("stop", "buy", 0, 0, price=101.3), 100, 101, 99, 0.2) is None
    assert try_fill(Order("stop", "sell", 0, 0, price=99.5), 100, 101, 99, 0.2) == 99.5


def test_stop_is_assumed_first_when_both_are_inside_one_candle():
    assert check_exit("buy", 99, 101, 100, 102, 98, 0.0, False, True) == ("sl", 99)


def test_no_instant_win_on_the_fill_candle_of_a_pending_order():
    assert check_exit("buy", 95, 101, 100, 102, 99, 0.0, True, False) is None
    assert check_exit("buy", 95, 101, 100, 102, 99, 0.0, True, True) == ("tp", 101)


def test_sell_exits_at_the_ask():
    assert check_exit("sell", 101, 98, 100, 100.9, 99, 0.2, False, True) == ("sl", 101)
    assert check_exit("sell", 101, 98, 100, 100.5, 97.9, 0.2, False, True) is None   # ask 98.1
    assert check_exit("sell", 101, 98, 100, 100.5, 97.7, 0.2, False, True) == ("tp", 98)


def test_gap_only_after_the_market_was_closed():
    assert check_exit("buy", 99, 105, 97, 98, 96, 0.0, False, True, can_gap=True) == ("gap", 97)
    assert check_exit("buy", 99, 105, 97, 98, 96, 0.0, False, True, can_gap=False) == ("sl", 99)


# --- replay on a synthetic market ----------------------------------------------------

def make_frames(n=3000, seed=3):
    """Random-walk H1 candles plus H4 and D1 candles built from them."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    opens = np.r_[100, close[:-1]]
    rows = [(o, max(o, c) + abs(rng.normal(0, .15)), min(o, c) - abs(rng.normal(0, .15)), c)
            for o, c in zip(opens, close)]
    h1 = candles(rows, start="2025-01-06 00:00")
    def bigger(rule):
        g = h1.resample(rule)
        out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                            "low": g["low"].min(), "close": g["close"].last()}).dropna()
        out.insert(0, "time_server", out.index.tz_localize(None))
        out["tick_volume"], out["spread"], out["real_volume"] = 100, 0, 0
        return out
    return {"H1": h1, "H4": bigger("4h"), "D1": bigger("1D")}


@pytest.fixture(scope="module")
def frames():
    return make_frames()


@pytest.fixture
def bt_cfg(cfg):
    c = copy.deepcopy(cfg)
    c["backtest"]["min_history_bars"] = {"H4": 60, "D1": 10}
    return c


def test_history_sees_exactly_what_the_live_analysis_sees(frames, bt_cfg):
    """No look-ahead, checked end to end: at candle i the backtest must see the
    same zones, location and trend as the live analysis run on the candles that
    were closed at that moment."""
    h = History("GOLD", bt_cfg, frames)
    for i in (700, 1234, 2000, 2999):
        cut = {"H1": frames["H1"].iloc[:i + 1]}
        for tf in ("H4", "D1"):
            cut[tf] = frames[tf].iloc[:h.idx[tf][i] + 1]
        live = analyze("GOLD", bt_cfg, cut)
        snap = h.snapshot(i)
        assert [(round(z.low, 6), round(z.high, 6), z.touches, z.status) for z in snap.zones] \
            == [(round(z.low, 6), round(z.high, 6), z.touches, z.status) for z in live.zones]
        assert snap.loc.where == live.location
        for tf in ("H4", "D1"):
            assert h.trend(tf, i) == live.structures[tf].trend


def test_higher_timeframe_candles_are_used_only_once_closed(frames, bt_cfg):
    h = History("GOLD", bt_cfg, frames)
    i = 1001                                       # H1 candle 17:00-18:00
    close_time = h.time[i] + pd.Timedelta(hours=1)
    last_h4 = frames["H4"].index[h.idx["H4"][i]]
    assert last_h4 + pd.Timedelta(hours=4) <= close_time < last_h4 + pd.Timedelta(hours=8)


def test_trend_series_matches_classify_at_every_cut(bt_cfg):
    df = path([100, 110, 105, 115, 110, 104, 108, 99, 103])
    sw = find_swings(df, 3)
    series = trend_series(len(df), sw)
    for t in range(10, len(df)):
        assert series[t] == classify(df.iloc[:t + 1], bt_cfg, "H4").trend


class BuyOnce(Strategy):
    """Test strategy: one market buy decided at candle 5."""
    name = "buy_once"

    def on_close(self, h, i, flat):
        if i == 5 and flat:
            return Order("market", "buy", sl=h.close[i] - 1.0, tp=h.close[i] + 2.0)
        return None


def test_replay_enters_on_the_next_candle_and_pays_the_spread(bt_cfg):
    c = copy.deepcopy(bt_cfg)
    c["backtest"]["min_history_bars"] = {"H4": 0, "D1": 0}
    c["symbols"]["GOLD"]["typical_spread"] = 0.5
    rising = candles([(100 + k, 100.6 + k, 99.9 + k, 100.5 + k) for k in range(20)])
    h = History("GOLD", c, {"H1": rising, "H4": rising, "D1": rising})
    h.first_bar = 0
    (trade,) = run(h, BuyOnce(c))
    assert trade.entry_time == rising.index[6]             # decided at 5, filled at 6
    assert trade.entry == pytest.approx(106 + 0.5)         # open of candle 6 + spread
    assert trade.outcome == "tp"
    # target was 2.0 above the signal close, but the spread and the gap to the
    # next open ate part of it: less than the 2R that was planned
    assert 0 < trade.r < 2


def test_strategies_run_and_respect_their_rules(frames, bt_cfg):
    h = History("GOLD", bt_cfg, frames)
    for name, cls in STRATEGIES.items():
        trades = run(h, cls(bt_cfg))
        for a, b in zip(trades, trades[1:]):
            assert b.entry_time >= a.exit_time             # one trade at a time
        for t in trades:
            assert t.entry_time > t.signal_time            # never filled on the signal candle
            assert (t.sl < t.entry < t.tp) if t.direction == "buy" else (t.tp < t.entry < t.sl)
            if name == "h2l2":                             # only with the H4 trend
                assert t.info["trend_H4"] == "with trend"
                assert t.info["pattern"] == ("H2" if t.direction == "buy" else "L2")


def test_replay_gives_the_same_trades_whatever_comes_later(frames, bt_cfg):
    """Cutting the future off must not change the trades already finished."""
    full = run(History("GOLD", bt_cfg, frames), STRATEGIES["level"](bt_cfg))
    short_frames = {"H1": frames["H1"].iloc[:2000],
                    "H4": frames["H4"][frames["H4"].index < frames["H1"].index[2000]],
                    "D1": frames["D1"][frames["D1"].index < frames["H1"].index[2000]]}
    short = run(History("GOLD", bt_cfg, short_frames), STRATEGIES["level"](bt_cfg))
    assert len(short) > 0
    for a, b in zip(short, full):
        assert (a.entry_time, a.exit_time, round(a.r, 6)) == (b.entry_time, b.exit_time, round(b.r, 6))


def test_filter_reduces_trades(frames, bt_cfg):
    h = History("GOLD", bt_cfg, frames)
    everything = run(h, STRATEGIES["h2l2"](bt_cfg))
    london = run(h, STRATEGIES["h2l2"](bt_cfg, accept=make_accept({"session": ["london"]})))
    assert 0 < len(london) < len(everything)
    assert all(t.info["session"] == "london" for t in london)


# --- statistics and research ----------------------------------------------------------

def test_summary_numbers(frames, bt_cfg):
    trades = run(History("GOLD", bt_cfg, frames), STRATEGIES["h2l2"](bt_cfg))
    s = summarize(trades)
    r = [t.r for t in trades]
    assert s["trades"] == len(r) and s["expectancy_r"] == pytest.approx(np.mean(r))
    assert s["ci_low"] <= s["expectancy_r"] <= s["ci_high"]
    assert summarize([])["trades"] == 0


def test_overrides_do_not_touch_the_original(cfg):
    changed = research.with_overrides(cfg, {"stops.fixed_rr": 9})
    assert changed["stops"]["fixed_rr"] == 9 and cfg["stops"]["fixed_rr"] != 9


def test_study_never_scores_the_search_on_unseen_trades(frames, bt_cfg):
    hs = {"GOLD": History("GOLD", bt_cfg, frames)}
    r = research.study(hs, "h2l2", bt_cfg)
    cut = r["cuts"]["GOLD"]
    assert frames["H1"].index[0] < cut < frames["H1"].index[-1]
    assert r["combinations_tried"] > 0
    assert r["tuned_in"]["trades"] + r["tuned_out"]["trades"] > 0
    assert research.verdict(r)[0] in ("EDGE SHOWN", "BREAK-EVEN", "NO EDGE", "TOO FEW TRADES")


def test_live_analysis_applies_the_configured_filters(cfg):
    from tests.helpers import append
    df = append(path([105, 110, 100, 110.1, 100.1, 105, 100.4]), [(100.2, 100.32, 99.95, 100.3)])
    frames3 = {"D1": df, "H4": df, "H1": df}
    c = copy.deepcopy(cfg)
    c["backtest"]["min_history_bars"] = {"H4": 0, "D1": 0}
    assert analyze("GOLD", c, frames3).verdict == "SETUP CANDIDATE"
    c["filters"]["level"] = {"session": ["never"]}
    a = analyze("GOLD", c, frames3)
    assert a.verdict == "NO TRADE" and a.plan is None
    assert "extra conditions" in a.reasons[-1]


def test_latest_order_matches_what_the_backtest_traded(frames, bt_cfg):
    """`deals` must show the same signal the backtest acted on at that moment."""
    from market_analyst.backtest import latest_order
    full = History("GOLD", bt_cfg, frames)
    trade = run(full, STRATEGIES["h2l2"](bt_cfg))[-1]
    i = frames["H1"].index.get_loc(trade.signal_time)
    cut = {"H1": frames["H1"].iloc[:i + 1]}
    for tf in ("H4", "D1"):
        cut[tf] = frames[tf].iloc[:full.idx[tf][i] + 1]
    order = latest_order(History("GOLD", bt_cfg, cut), STRATEGIES["h2l2"](bt_cfg))
    assert order is not None
    assert (order.direction, round(order.sl, 6), round(order.tp, 6)) == \
        (trade.direction, round(trade.sl, 6), round(trade.tp, 6))
