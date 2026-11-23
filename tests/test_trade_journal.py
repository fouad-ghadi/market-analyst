"""Trade journal: import from MT5, rule checks, statistics, CLI."""

import copy
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from market_analyst import trade_journal as tj
from market_analyst.cli import main
from market_analyst.levels import Zone
from tests.conftest import FakeMT5

FIXTURE = Path(__file__).parent / "fixtures" / "journal_trades.csv"


def by_rule(review):
    return {c.rule: c for c in review.checks}


# --- the trades from CLAUDE.md, as a CSV ------------------------------------------------

def test_fixture_trades_break_the_rules_they_really_broke(cfg):
    reviews = tj.review_all(tj.load_trades(FIXTURE), {}, cfg)
    rules = {rv.trade["position"]: by_rule(rv) for rv in reviews}
    # 2026-07-20 EURUSD, own analysis, R:R 1:2.7: a "good loss"
    assert not reviews[0].broken and reviews[0].r == pytest.approx(-1.01, abs=0.01)
    # GOLD signals: not my own analysis
    assert rules["1002"]["own analysis"].status == tj.BROKEN
    # 2026-07-27: R:R 1:1.25 and two trades open at once
    assert rules["1004"]["risk:reward"].status == tj.BROKEN
    assert rules["1004"]["max open trades"].status == tj.BROKEN
    assert rules["1003"]["max open trades"].status == tj.OK      # the first one was alone
    # 2026-08-04: R:R 1:0.4, the biggest loss
    assert "1:0.41" in rules["1005"]["risk:reward"].detail
    assert reviews[4].r == pytest.approx(-1.0)
    # no source written -> cannot be checked, never guessed
    assert rules["1006"]["own analysis"].status == tj.NA
    # no candles -> the chart rules are n/a, not "ok"
    assert rules["1006"]["entry at a validated level"].status == tj.NA


def test_summary_splits_rule_followers_from_rule_breakers(cfg):
    s = tj.summarize(tj.review_all(tj.load_trades(FIXTURE), {}, cfg))
    assert s["all"]["trades"] == 6 and s["all"]["net"] == pytest.approx(5.36)
    assert s["followed"]["trades"] + s["broke"]["trades"] == 6
    assert s["rules"]["risk:reward"][tj.BROKEN] == 3
    assert s["all"]["max_drawdown"] == pytest.approx(25.13)     # -21.26 then -3.87


def trade(**kw):
    base = dict(position="", symbol="GOLD", direction="buy", order_type="market", volume=0.01,
                open_time_utc=pd.Timestamp("2026-11-10 13:00", tz="UTC"), open_price=100.0,
                sl=99.0, tp=103.0, close_time_utc=pd.Timestamp("2026-11-10 14:00", tz="UTC"),
                close_price=99.0, exit_reason="sl", profit=-1.0, commission=0.0, swap=0.0,
                balance_before=np.nan, source="own", note="")
    base.update(kw)
    return base


def frame(*trades):
    return pd.DataFrame(list(trades), columns=tj.COLUMNS)


def at(hour):
    return pd.Timestamp(f"2026-11-10 {hour:02d}:00", tz="UTC")


def test_third_trade_after_two_losses_breaks_the_daily_stop(cfg):
    t = frame(trade(open_time_utc=at(9), close_time_utc=at(10)),
              trade(open_time_utc=at(11), close_time_utc=at(12)),
              trade(open_time_utc=at(13), close_time_utc=at(14)))
    reviews = tj.review_all(t, {}, cfg)
    assert [by_rule(r)["daily stop"].status for r in reviews] == [tj.OK, tj.OK, tj.BROKEN]


def test_daily_loss_percent_also_stops_the_day(cfg):
    c = copy.deepcopy(cfg)
    c["journal"]["starting_balance"] = 100.0
    t = frame(trade(open_time_utc=at(9), close_time_utc=at(10), profit=-4.0),   # -4% day
              trade(open_time_utc=at(11), close_time_utc=at(12), profit=1.0))
    assert by_rule(tj.review_all(t, {}, c)[1])["daily stop"].status == tj.BROKEN


def test_bigger_size_after_a_loss_is_revenge_trading(cfg):
    t = frame(trade(open_time_utc=at(9), close_time_utc=at(10)),
              trade(open_time_utc=at(11), close_time_utc=at(12), volume=0.02,
                    profit=1.0, close_price=100.5))
    second = by_rule(tj.review_all(t, {}, cfg)[1])
    assert second["no size increase after a loss"].status == tj.BROKEN
    assert second["learning lot size"].status == tj.BROKEN


def test_risk_percent_uses_the_balance(cfg):
    c = copy.deepcopy(cfg)
    c["risk"]["learning_phase"] = False
    # 1 lot of gold, stop 10 dollars away = $1000 at risk = 10% of 10,000
    t = frame(trade(volume=1.0, sl=90.0, tp=130.0, close_price=90.0, profit=-1000.0))
    check = by_rule(tj.review_all(t, {}, c)[0])["risk per trade"]
    assert check.status == tj.BROKEN and "10.00%" in check.detail


def test_no_stop_loss_is_broken_and_r_unknown(cfg):
    rv = tj.review_all(frame(trade(sl=0.0)), {}, cfg)[0]
    assert by_rule(rv)["stop loss set"].status == tj.BROKEN and rv.r is None


def test_money_per_point_comes_from_the_trade_itself(cfg):
    # USD/JPY: 1 lot, price moved 0.50 yen, profit $333 (yen converted to dollars).
    # contract size x volume would say $50,000 per 1.0, which is in yen, not dollars.
    t = trade(symbol="USDJPY", volume=1.0, open_price=150.0, close_price=150.5,
              profit=333.0, sl=149.5, tp=151.5, direction="buy")
    assert tj.money_per_price_unit(t, cfg) == pytest.approx(666.0)


# --- chart rules (with hand-made zones) ------------------------------------------------------

def zone(low, high, touches=2):
    ts = pd.Timestamp("2026-11-01", tz="UTC")
    return Zone(low, high, "H4", touches, 1, 1, ts, ts, 5, 3.0,
                "validated" if touches >= 2 else "watch", "support")


def fake_history(zones, trend="up"):
    times = pd.date_range("2026-11-10 00:00", periods=24, freq="h", tz="UTC")
    return SimpleNamespace(time=times, entry_tf="H1", first_bar=0, atr=np.full(24, 2.0),
                           zones=lambda i: zones, trend=lambda tf, i: trend)


def test_buy_at_support_with_stop_below_it_passes(cfg):
    h = fake_history([zone(99.0, 100.0), zone(110.0, 111.0)])
    t = trade(open_price=100.2, sl=98.5, tp=109.0)
    checks = {c.rule: c for c in tj.chart_checks(t, h, True, cfg)}
    assert checks["entry at a validated level"].status == tj.OK
    assert checks["not in the middle of the range"].status == tj.OK
    assert checks["stop beyond the level"].status == tj.OK
    assert checks["H4 trend"].status == tj.OK


def test_stop_inside_the_zone_is_broken(cfg):
    h = fake_history([zone(99.0, 100.0), zone(110.0, 111.0)])
    t = trade(open_price=100.2, sl=99.5, tp=109.0)
    checks = {c.rule: c for c in tj.chart_checks(t, h, True, cfg)}
    assert checks["stop beyond the level"].status == tj.BROKEN


def test_buy_in_the_middle_of_the_range_is_broken(cfg):
    h = fake_history([zone(99.0, 100.0), zone(110.0, 111.0)], trend="down")
    t = trade(open_price=105.0, sl=98.0, tp=109.0)
    checks = {c.rule: c for c in tj.chart_checks(t, h, True, cfg)}
    assert checks["entry at a validated level"].status == tj.BROKEN
    assert checks["not in the middle of the range"].status == tj.BROKEN
    assert checks["H4 trend"].status == tj.NOTE          # counter-trend: allowed, flagged


def test_buying_at_resistance_is_the_wrong_side(cfg):
    h = fake_history([zone(90.0, 91.0), zone(100.0, 101.0)])
    t = trade(open_price=99.8, sl=97.0, tp=110.0)
    check = tj.chart_checks(t, h, True, cfg)[0]
    assert check.status == tj.BROKEN and "wrong side" in check.detail


def test_chart_rules_only_use_candles_closed_before_the_entry(cfg):
    seen = []
    h = fake_history([zone(99.0, 100.0)])
    h.zones = lambda i: seen.append(i) or [zone(99.0, 100.0)]
    # entry at 05:30: the 04:00 candle closed at 05:00 -> index 4, never 5
    tj.chart_checks(trade(open_time_utc=pd.Timestamp("2026-11-10 05:30", tz="UTC")),
                    h, True, cfg)
    assert seen == [4]


# --- import from MT5 -------------------------------------------------------------------------

Deal = namedtuple("Deal", "ticket order time type entry position_id volume price profit "
                          "commission swap fee symbol reason comment")
Ord = namedtuple("Ord", "ticket type sl tp position_id")
SERVER = pd.Timestamp("2026-11-10 12:00")          # winter: server = UTC+2


def sec(minutes):
    return int((SERVER + pd.Timedelta(minutes=minutes)).timestamp())


def history():
    deals = [
        Deal(1, 0, sec(-600), 2, 0, 0, 0, 0, 10000.0, 0, 0, 0, "", 0, "deposit"),
        # position 7: GOLD buy limit, half closed by hand, the rest stopped out
        Deal(2, 70, sec(0), 0, 0, 7, 0.02, 4100.0, 0, 0, 0, 0, "GOLD", 0, ""),
        Deal(3, 71, sec(30), 1, 1, 7, 0.01, 4110.0, 10.0, 0, 0, 0, "GOLD", 0, ""),
        Deal(4, 72, sec(60), 1, 1, 7, 0.01, 4095.0, -5.0, 0, 0, 0, "GOLD", 4, ""),
        # position 8: still open -> not in the journal
        Deal(5, 80, sec(90), 1, 0, 8, 0.01, 1.1500, 0, 0, 0, 0, "EURUSD", 0, ""),
    ]
    orders = [Ord(70, 2, 4095.0, 4120.0, 7), Ord(71, 1, 0, 0, 7), Ord(72, 1, 0, 0, 7),
              Ord(80, 1, 0, 0, 8)]
    return deals, orders


def test_deals_become_one_row_per_closed_position(cfg):
    deals, orders = history()
    t = tj.positions_from_deals(deals, orders, cfg)
    assert len(t) == 1
    row = t.iloc[0]
    assert (row["symbol"], row["direction"], row["order_type"]) == ("GOLD", "buy", "limit")
    assert row["volume"] == pytest.approx(0.02) and row["close_price"] == pytest.approx(4102.5)
    assert (row["sl"], row["tp"], row["exit_reason"]) == (4095.0, 4120.0, "sl")
    assert row["profit"] == pytest.approx(5.0) and row["balance_before"] == pytest.approx(10000)
    assert row["open_time_utc"] == pd.Timestamp("2026-11-10 10:00", tz="UTC")


def test_stop_added_after_entry_is_found_on_a_later_order(cfg):
    deals, orders = history()
    orders[0] = Ord(70, 2, 0, 0, 7)                  # opening order had no SL/TP
    orders[2] = Ord(72, 1, 4095.0, 4120.0, 7)
    row = tj.positions_from_deals(deals, orders, cfg).iloc[0]
    assert (row["sl"], row["tp"]) == (4095.0, 4120.0)


def test_reimport_keeps_what_you_wrote(cfg, tmp_path):
    deals, orders = history()
    new = tj.positions_from_deals(deals, orders, cfg)
    old = new.copy()
    old.loc[0, "source"], old.loc[0, "note"] = "own", "H4 support"
    hand = frame(trade(open_time_utc=at(1), close_time_utc=at(2)))
    merged = tj.merge_trades(pd.concat([hand, old], ignore_index=True), new)
    assert len(merged) == 2
    assert merged.iloc[1][["source", "note"]].tolist() == ["own", "H4 support"]
    path = tmp_path / "j.csv"
    tj.save_trades(merged, path)
    assert tj.load_trades(path)["note"].tolist() == ["", "H4 support"]


def test_bad_file_is_a_clear_error(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("a,b\n1,2\n")
    with pytest.raises(tj.JournalError, match="missing the columns"):
        tj.load_trades(f)


# --- CLI ------------------------------------------------------------------------------------

def test_cli_journal_on_the_fixture(cfg_file, capsys):
    assert main(["--config", str(cfg_file), "--source", "yahoo", "journal",
                 "--file", str(FIXTURE)]) == 0
    out = capsys.readouterr().out
    assert "BROKEN risk:reward: planned 1:0.41" in out
    assert "no rule broken" in out and "max open trades" in out
    report = pd.read_csv(cfg_file.parent / "reports" / "journal_review.csv")
    assert len(report) == 6 and report["rules_broken"].tolist()[3] == 3


def test_cli_journal_import_from_mt5_then_checks_against_candles(cfg_file, capsys):
    deals, orders = history()
    fake = FakeMT5(n_bars=300, deals=deals, orders=orders)
    assert main(["--config", str(cfg_file), "fetch"], mt5_module=fake) == 0
    assert main(["--config", str(cfg_file), "journal", "--import", "-v"], mt5_module=fake) == 0
    out = capsys.readouterr().out
    assert "Imported 1 closed trade(s)" in out and "4100.00" in out
    assert (cfg_file.parent / "data" / "journal_trades.csv").exists()


def test_cli_journal_without_file_says_how_to_import(cfg_file, capsys):
    assert main(["--config", str(cfg_file), "--source", "yahoo", "journal"]) == 2
    assert "journal --import" in capsys.readouterr().err
