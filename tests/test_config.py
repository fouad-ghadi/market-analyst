"""config.yaml must load, contain the rulebook values, and reject dangerous edits."""

import copy

import pytest

from market_analyst.config import ConfigError, load_config, validate_config


@pytest.fixture
def raw():
    c = load_config()
    c.pop("_base_dir")
    return c


def test_real_config_contains_rulebook_values(raw):
    traded = {s for s, spec in raw["symbols"].items() if spec.get("trade", True)}
    assert traded == {"EURUSD", "GOLD"}                    # other markets are test-only
    assert raw["stops"]["min_risk_reward"] == 2.0          # minimum R:R 1:2
    assert raw["risk"]["risk_pct_per_trade"] == 1.0        # 1% max
    assert raw["risk"]["fixed_lot"] == 0.01                # learning phase
    assert raw["risk"]["max_open_trades"] == 1
    assert raw["risk"]["max_losses_per_day"] == 2
    assert raw["risk"]["max_daily_loss_pct"] == 3.0
    assert raw["levels"]["min_touches_validated"] == 2
    assert raw["levels"]["tolerance_atr_mult"] == 0.25
    assert raw["swings"]["fractal_n"] == 3
    assert raw["strategy"]["middle_of_range_pct"] == 40
    assert raw["stops"]["sl_buffer_spread_mult"] == 2.0
    assert raw["stops"]["sl_buffer_atr_mult"] == 0.1
    assert "M1" not in raw["timeframes"]["entry"]          # M1 is noise
    assert raw["levels"]["min_reaction_atr_mult"] == 3.0   # a touch needs a clear reaction
    assert raw["data"]["source"] == "auto"                 # MT5 on the PC, public feed in the cloud
    # researched settings shipped in the file (see RESEARCH.md); rulebook = 0.0 / next_level
    assert raw["stops"]["min_risk_atr_mult"] == 1.0
    assert (raw["stops"]["tp_mode"], raw["stops"]["fixed_rr"]) == ("fixed_rr", 3.0)
    assert raw["filters"]["level"] == {"trend_D1": ["with trend"], "ema50_slope": True}
    assert raw["filters"]["retest"] == {} and raw["filters"]["h2l2"] == {}


@pytest.mark.parametrize("dotted, bad_value", [
    ("risk.risk_pct_per_trade", 2.0),       # more than 1% per trade
    ("risk.risk_pct_per_trade", 0),
    ("stops.min_risk_reward", 0.5),
    ("stops.allow_widen_sl", True),          # never move SL further away
    ("risk.allow_size_increase_after_loss", True),  # no revenge trading
    ("time.server_time_mode", "local"),
    ("data.source", "telegram"),              # only auto / mt5 / yahoo
    ("stops.tp_mode", "moon"),
    ("stops.fixed_rr", 1.0),                  # a fixed target below the 1:2 minimum
    ("strategy.middle_of_range_pct", 120),
])
def test_dangerous_values_are_rejected(raw, dotted, bad_value):
    c = copy.deepcopy(raw)
    section, key = dotted.split(".")
    c[section][key] = bad_value
    with pytest.raises(ConfigError):
        validate_config(c)


def test_m1_timeframe_is_rejected(raw):
    c = copy.deepcopy(raw)
    c["data"]["bars"]["M1"] = 1000
    with pytest.raises(ConfigError, match="M1"):
        validate_config(c)


def test_missing_value_gives_clear_error(raw):
    c = copy.deepcopy(raw)
    del c["stops"]["min_risk_reward"]
    with pytest.raises(ConfigError, match="stops.min_risk_reward"):
        validate_config(c)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
