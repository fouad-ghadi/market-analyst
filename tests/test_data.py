"""Connection, symbol specs, candle download and cache."""

import pytest

from market_analyst.data import (
    MT5Connection,
    MT5Error,
    account_summary,
    account_warnings,
    fetch_rates,
    get_symbol_spec,
    load_cache,
    update_cache,
)
from tests.conftest import FakeMT5


# --- connection -----------------------------------------------------------------

def test_connection_opens_and_always_closes(cfg):
    fake = FakeMT5()
    with MT5Connection(cfg, fake) as mt5:
        assert mt5.initialized
    assert fake.shutdown_called


def test_connection_failure_gives_clear_error(cfg):
    fake = FakeMT5(init_ok=False)
    with pytest.raises(MT5Error, match="terminal is open"):
        with MT5Connection(cfg, fake):
            pass
    assert fake.shutdown_called


def test_demo_account_has_no_warning(cfg):
    acc = account_summary(FakeMT5(trade_mode=0))
    assert acc["type"] == "DEMO" and acc["margin_mode"] == "hedging"
    assert account_warnings(acc, cfg) == []


def test_real_account_is_flagged(cfg):
    acc = account_summary(FakeMT5(trade_mode=2))
    assert any("REAL" in w for w in account_warnings(acc, cfg))


# --- symbol specs ----------------------------------------------------------------

def test_eurusd_spec_and_pip_value(fake_mt5):
    spec = get_symbol_spec(fake_mt5, "EURUSD", 0.0001)
    assert spec.pip_value_per_lot == pytest.approx(10.0)    # $10 per pip per lot
    assert spec.spread_pips == pytest.approx(1.6)            # 16 points = 1.6 pips
    assert spec.volume_min == 0.01


def test_gold_spec_and_pip_value(fake_mt5):
    spec = get_symbol_spec(fake_mt5, "GOLD", 0.1)
    assert spec.pip_value_per_lot == pytest.approx(10.0)    # 100 oz x $0.10
    assert spec.spread_price == pytest.approx(0.35)


def test_unknown_symbol_is_explained(fake_mt5):
    with pytest.raises(MT5Error, match="XAUUSD"):
        get_symbol_spec(fake_mt5, "XAUUSD", 0.1)


# --- candles --------------------------------------------------------------------

def test_fetch_returns_only_closed_candles(fake_mt5, cfg):
    df = fetch_rates(fake_mt5, "GOLD", "H1", 50, cfg)
    all_bars = fake_mt5.all_bars("GOLD", fake_mt5.TIMEFRAME_H1)
    forming_close = all_bars["close"][-1]        # the newest candle, still open
    last_closed_close = all_bars["close"][-2]
    assert len(df) == 50
    assert df["close"].iloc[-1] == pytest.approx(last_closed_close)
    assert forming_close not in df["close"].values


def test_fetch_can_include_forming_candle_if_asked(fake_mt5, cfg):
    cfg["data"]["include_forming_bar"] = True
    df = fetch_rates(fake_mt5, "GOLD", "H1", 50, cfg)
    all_bars = fake_mt5.all_bars("GOLD", fake_mt5.TIMEFRAME_H1)
    assert df["close"].iloc[-1] == pytest.approx(all_bars["close"][-1])


def test_frame_shape(fake_mt5, cfg):
    df = fetch_rates(fake_mt5, "EURUSD", "H4", 20, cfg)
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert list(df.columns[:5]) == ["time_server", "open", "high", "low", "close"]
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()


def test_unknown_symbol_fetch_fails(fake_mt5, cfg):
    with pytest.raises(MT5Error):
        fetch_rates(fake_mt5, "NOPE", "H1", 10, cfg)


# --- cache ----------------------------------------------------------------------

def test_cache_roundtrip(fake_mt5, cfg):
    df = update_cache(fake_mt5, cfg, "GOLD", "H4")
    back = load_cache(cfg, "GOLD", "H4")
    assert back.index.equals(df.index)
    assert back["close"].tolist() == pytest.approx(df["close"].tolist())
    assert back["time_server"].equals(df["time_server"])


def test_cache_grows_without_duplicates(cfg):
    fake = FakeMT5(n_bars=300)
    first = update_cache(fake, cfg, "GOLD", "H1")      # 100 bars
    fake.n_bars = 310                                  # 10 new candles appear on the server
    second = update_cache(fake, cfg, "GOLD", "H1")
    assert len(first) == 100
    assert len(second) == 110                          # old 100 kept + 10 new
    assert not second.index.duplicated().any()


def test_load_cache_missing_returns_none(cfg):
    assert load_cache(cfg, "GOLD", "D1") is None
