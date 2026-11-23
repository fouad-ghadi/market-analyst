"""Regression tests for the four audit findings (November 2026).

1. Deal review: a stop that price gapped through was counted as a win.
2. Deals: the dollar risk on USD/JPY was counted in yen.
3. `chart` drew different zones than the verdict used.
4. Rules in config.yaml that no code read.
"""

import copy
import re

import pytest
import yaml

from market_analyst import cli, deals, journal
from market_analyst.analysis import analyze
from market_analyst.config import (
    DEFAULT_CONFIG_PATH,
    FIXED_RULES,
    ConfigError,
    validate_config,
)
from market_analyst.report import format_analysis
from tests.helpers import append, candles, path
from tests.test_analysis import BASE, DOWNTREND, HAMMER
from tests.test_cli import _fetched

# --- 1. gapped-through stop ----------------------------------------------------------------


def _deal(order, direction, entry, sl, tp):
    return {"candle_close_utc": "2026-11-09 01:00", "order": order, "direction": direction,
            "entry": str(entry), "sl": str(sl), "tp": str(tp)}


def test_buy_limit_filled_by_a_gap_below_its_stop_is_a_loss():
    # Buy limit at 100, stop 99. The next candle OPENS at 97 (weekend gap):
    # the order fills at 97.2 (ask) and the stop is already behind price.
    # The old code booked the exit AT 99, above the fill: a +1.8R "win".
    h1 = candles([(100, 100.2, 99.8, 100), (97, 97.5, 96.8, 97.3)] + [(97.3, 97.5, 97, 97.2)] * 3,
                 start="2026-11-09 00:00")
    out = journal.resolve_deal(_deal("limit", "buy", 100, 99, 103), h1, spread=0.2, max_hold=50)
    assert out["status"] == "gap"
    assert out["r"] == pytest.approx(-0.2)          # closed at the open bid 97: loses the spread


def test_sell_limit_filled_by_a_gap_above_its_stop_is_a_loss():
    h1 = candles([(100, 100.2, 99.8, 100), (103, 103.5, 102.8, 103.2)] + [(103, 103.2, 103, 103)] * 3,
                 start="2026-11-09 00:00")
    out = journal.resolve_deal(_deal("limit", "sell", 100, 101, 97), h1, spread=0.2, max_hold=50)
    assert out["status"] == "gap" and out["r"] < 0


def test_normal_stop_out_is_still_minus_one_r():
    h1 = candles([(100, 100.2, 99.8, 100), (100, 100.1, 98.5, 98.8)], start="2026-11-09 00:00")
    out = journal.resolve_deal(_deal("market", "buy", 100, 99, 103), h1, spread=0.0, max_hold=50)
    assert (out["status"], out["r"]) == ("sl", pytest.approx(-1.0))


# --- 2. dollar risk on USD/JPY -----------------------------------------------------------------


def test_usdjpy_risk_is_converted_from_yen_to_dollars(cfg):
    # 1.00 yen against 0.01 lot = 1.00 x 100,000 x 0.01 = 1,000 yen = about 6.45 USD at 155
    usd = deals.risk_in_account_currency(cfg, "USDJPY", 1.00, 0.01, 155.0)
    assert usd == pytest.approx(1000 / 155)


def test_usd_quoted_symbols_are_unchanged(cfg):
    assert deals.risk_in_account_currency(cfg, "EURUSD", 0.0010, 0.01, 1.15) == pytest.approx(1.0)
    assert deals.risk_in_account_currency(cfg, "GOLD", 10.0, 0.01, 4100) == pytest.approx(10.0)


def test_unknown_conversion_says_unknown_instead_of_a_wrong_number(cfg):
    c = copy.deepcopy(cfg)
    c["symbols"]["EURGBP"] = {"pip_size": 0.0001, "contract_size": 100000,
                              "base_currency": "EUR", "quote_currency": "GBP"}
    assert deals.risk_in_account_currency(c, "EURGBP", 0.001, 0.01, 0.87) is None


# --- 3. chart and verdict use the same zones ----------------------------------------------------


def test_chart_draws_exactly_the_zones_the_verdict_uses(cfg_file, monkeypatch, capsys):
    fake = _fetched(cfg_file)
    drawn = {}
    monkeypatch.setattr(cli.report, "draw_chart",
                        lambda df, zones, *a, **k: drawn.setdefault(df.index[-1], zones))
    assert cli.main(["--config", str(cfg_file), "chart", "GOLD", "-t", "H4"],
                    mt5_module=fake) == 0
    (zones,) = drawn.values()
    cfg = cli.load_config(cfg_file)
    cfg["_source"] = "mt5"
    frames = {tf: cli.load_cache(cfg, "GOLD", tf, "mt5") for tf in ("D1", "H4", "H1")}
    verdict = analyze("GOLD", cfg, frames)
    key = lambda zs: [(z.low, z.high, z.timeframe, z.status, z.role) for z in zs]
    assert key(zones) == key(verdict.zones) and zones


# --- 4. config rules that nothing read -----------------------------------------------------------

# Keys whose NAME is data (symbol names, timeframes), not a setting the code looks up.
NAME_KEYS = ("symbols.", "data.yahoo.symbols.", "data.bars.", "levels.timeframe_weight.",
             "backtest.min_history_bars.", "filters.")


def _leaves(d, prefix=""):
    for k, v in d.items():
        dotted = prefix + str(k)
        if any(dotted.startswith(p) for p in NAME_KEYS):
            continue
        if isinstance(v, dict):
            yield from _leaves(v, dotted + ".")
        else:
            yield dotted


def test_every_setting_in_config_yaml_is_read_by_the_code():
    """A rule in config.yaml that no code reads is a rule you think is
    protecting you and is not. Every setting must be used somewhere."""
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    src = "".join(f.read_text(encoding="utf-8")
                  for f in (DEFAULT_CONFIG_PATH.parent / "src/market_analyst").glob("*.py"))
    unused = [d for d in _leaves(raw)
              if not re.search(r"[\"'.]" + re.escape(d.split(".")[-1]) + r"[\"']", src)]
    assert unused == []


@pytest.mark.parametrize("dotted", sorted(FIXED_RULES))
def test_fixed_rules_cannot_be_changed_silently(cfg, dotted):
    c = copy.deepcopy(cfg)
    section, key = dotted.split(".")
    value = c[section][key]
    c[section][key] = (not value) if isinstance(value, bool) else (
        value + 1 if isinstance(value, int) else "bodies")
    with pytest.raises(ConfigError, match=dotted):
        validate_config(c)


def test_breakeven_management_needs_breakeven_allowed(cfg):
    c = copy.deepcopy(cfg)
    c["stops"].update(allow_move_sl_to_breakeven=False, manage_at_r=1.0,
                      breakeven_after_manage=True)
    with pytest.raises(ConfigError, match="allow_move_sl_to_breakeven"):
        validate_config(c)


def _counter_trend_setup(cfg):
    df = append(path(BASE + [105, 100.4]), [HAMMER])
    return analyze("GOLD", cfg, {"D1": df, "H4": path(DOWNTREND), "H1": df})


def test_counter_trend_trades_are_refused_when_not_allowed(cfg):
    c = copy.deepcopy(cfg)
    c["strategy"]["allow_counter_trend"] = False
    a = _counter_trend_setup(c)
    assert (a.verdict, a.plan) == ("NO TRADE", None)
    assert "allow_counter_trend is false" in a.reasons[-1]


def test_plan_lists_the_account_rules_the_tool_cannot_check(cfg):
    a = _counter_trend_setup(cfg)
    assert a.verdict == "SETUP CANDIDATE"
    text = format_analysis(a, cfg, "mt5")
    assert "At most 1 open trade" in text and "after 2 losses or -3%" in text
    assert 'expiration to "today"' in text and "13:00 to 17:00 UTC" in text

