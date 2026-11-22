"""The verdict logic: NO TRADE / WAIT / SETUP CANDIDATE, on charts where we know the answer.

The base chart bounces twice between support (about 100) and resistance (about 110).
Each test changes only how the chart ENDS.
"""

import copy

from market_analyst.analysis import analyze
from market_analyst.report import format_analysis
from tests.helpers import append, path

BASE = [105, 110, 100, 110.1, 100.1]
HAMMER = (100.2, 100.32, 99.95, 100.3)          # long lower wick, closes near its high
PLAIN = (100.4, 100.45, 100.25, 100.3)          # ordinary small down candle
UPTREND = [100, 110, 105, 115, 110, 120, 116]
DOWNTREND = [120, 110, 115, 105, 110, 100, 104]


def run(df, cfg, h4=None):
    """Use the same candles on every timeframe unless a separate H4 chart is given."""
    return analyze("GOLD", cfg, {"D1": df, "H4": h4 if h4 is not None else df, "H1": df})


def test_middle_of_the_range_is_no_trade(cfg):
    a = run(path(BASE + [105]), cfg)
    assert a.location == "middle" and a.verdict == "NO TRADE"
    assert 40 < a.range_position_pct < 60
    assert "middle of the range" in a.reasons[0]


def test_between_levels_is_wait(cfg):
    a = run(path(BASE + [108]), cfg)             # 80% of the way up, not yet at resistance
    assert a.location == "between" and a.verdict == "WAIT"
    assert "nearer to resistance" in a.reasons[0]


def test_at_support_without_confirmation_candle_is_wait(cfg):
    a = run(append(path(BASE + [105, 100.4]), [PLAIN]), cfg)
    assert a.location == "at_support" and a.verdict == "WAIT"
    assert "no confirmation candle" in a.reasons[0]
    assert a.direction is None


def test_hammer_at_validated_support_is_a_setup_candidate(cfg):
    a = run(append(path(BASE + [105, 100.4]), [HAMMER]), cfg)
    assert a.location == "at_support"
    assert (a.verdict, a.direction) == ("SETUP CANDIDATE", "buy")
    assert a.plan.rr > cfg["stops"]["min_risk_reward"]
    assert a.plan.sl < a.support.low < a.plan.entry < a.plan.tp < a.resistance.low
    assert any(s["pattern"] == "hammer" for s in a.signals)


def test_same_setup_is_refused_when_risk_reward_is_too_small(cfg):
    c = copy.deepcopy(cfg)
    c["stops"]["min_risk_reward"] = 50           # absurd minimum, to force the rule
    a = run(append(path(BASE + [105, 100.4]), [HAMMER]), c)
    assert a.verdict == "NO TRADE"
    assert "below the 1:50 minimum" in a.reasons[-1]


def test_no_validated_target_means_no_trade(cfg):
    # Support is validated (two lows near 100) but no resistance has two touches.
    df = append(path([105, 100, 108, 100.1, 104, 100.4]), [HAMMER])
    a = run(df, cfg)
    assert a.resistance is None
    assert a.verdict == "NO TRADE"
    assert "no validated opposite level" in a.reasons[-1]


def test_buy_with_the_h4_uptrend(cfg):
    a = run(append(path(BASE + [105, 100.4]), [HAMMER]), cfg, h4=path(UPTREND))
    assert (a.verdict, a.trend_alignment) == ("SETUP CANDIDATE", "with trend")


def test_buy_against_the_h4_downtrend_is_lower_conviction(cfg):
    a = run(append(path(BASE + [105, 100.4]), [HAMMER]), cfg, h4=path(DOWNTREND))
    assert (a.verdict, a.trend_alignment) == ("SETUP CANDIDATE", "counter-trend")
    assert "lower conviction" in a.reasons[-1]


def test_analysis_ignores_candles_after_the_cut(cfg):
    """No look-ahead: analysing the first 50 candles gives the same answer
    whether or not later candles exist in the file."""
    long_df = append(path(BASE + [105, 100.4]), [HAMMER])
    short = run(path(BASE + [105]), cfg)
    cut = run(long_df.iloc[:50], cfg)
    assert (cut.verdict, cut.location, cut.price) == (short.verdict, short.location, short.price)


def test_report_text_never_promises_anything(cfg):
    a = run(append(path(BASE + [105, 100.4]), [HAMMER]), cfg)
    text = format_analysis(a, cfg, "mt5", news_lines=[])
    assert "VERDICT: SETUP CANDIDATE (BUY)" in text
    assert "PLAN TO CHECK (BUY" in text and "stop loss" in text
    assert "not a prediction" in text
    assert "none scheduled" in text
    for word in ("guarantee", "will rise", "will fall", "sure"):
        assert word not in text.lower()


def test_middle_of_a_narrow_range_wins_over_being_near_a_level(cfg):
    """In a narrow range, 'near a level' and 'middle' can both be true. Rule 5
    (middle = no trade) must win."""
    c = copy.deepcopy(cfg)
    c["patterns"]["near_level_atr_mult"] = 20      # huge margin: everything is "near"
    a = run(path(BASE + [105]), c)
    assert a.location == "middle" and a.verdict == "NO TRADE"
