"""Trade management, MT5 file import, deal review, news archive."""

import copy

import pandas as pd
import pytest

from market_analyst import journal, news
from market_analyst.backtest import History, Order, Strategy, run
from market_analyst.data import load_cache
from tests.helpers import candles


class BuyAt5(Strategy):
    name = "buy5"

    def on_close(self, h, i, flat):
        if i == 5 and flat:
            return Order("market", "buy", sl=h.close[i] - 1.0, tp=h.close[i] + 3.0)
        return None


def up_then_crash():
    """Rises about 1.7 above the entry, then falls far below the stop."""
    rows = [(100, 100.2, 99.8, 100)] * 6
    rows += [(100, 100.6, 99.9, 100.5), (100.5, 101.7, 100.4, 101.5), (101.5, 101.6, 97, 97.2)]
    rows += [(97.2, 97.4, 96.8, 97)] * 3
    return candles(rows)


def one_trade(cfg, **stops):
    c = copy.deepcopy(cfg)
    c["backtest"]["min_history_bars"] = {"H4": 0, "D1": 0}
    c["symbols"]["GOLD"]["typical_spread"] = 0.0
    c["stops"].update(stops)
    df = up_then_crash()
    h = History("GOLD", c, {"H1": df, "H4": df, "D1": df})
    h.first_bar = 0
    (trade,) = run(h, BuyAt5(c))
    return trade


def test_without_management_the_reversal_is_a_full_loss(cfg):
    t = one_trade(cfg)
    assert (t.outcome, t.r) == ("sl", pytest.approx(-1.0))


def test_breakeven_stop_turns_the_loss_into_zero(cfg):
    t = one_trade(cfg, manage_at_r=1.0)
    assert (t.outcome, t.r) == ("breakeven", pytest.approx(0.0))


def test_partial_exit_banks_half_at_one_r(cfg):
    t = one_trade(cfg, manage_at_r=1.0, partial_fraction=0.5)
    assert t.r == pytest.approx(0.5)        # half closed at +1R, the rest at breakeven


# --- MT5 export import ----------------------------------------------------------------

EXPORT = ("<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
          "2026.07.20\t16:00:00\t4030.10\t4035.00\t4028.00\t4031.65\t1500\t0\t35\n"
          "2026.07.20\t17:00:00\t4031.65\t4040.00\t4030.00\t4038.20\t1700\t0\t35\n")


def test_import_mt5_export_converts_server_time_to_utc(cfg, tmp_path):
    f = tmp_path / "GOLD_H1.csv"
    f.write_text(EXPORT)
    df = journal.import_mt5_csv(cfg, "GOLD", "H1", f)
    assert len(df) == 2
    assert df.index[0] == pd.Timestamp("2026-07-20 13:00", tz="UTC")   # summer: server = UTC+3
    assert df["close"].tolist() == [4031.65, 4038.20] and df["spread"].tolist() == [35, 35]
    assert load_cache(cfg, "GOLD", "H1", "mt5") is not None            # stored as MT5 data


def test_import_rejects_other_files(cfg, tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("a,b\n1,2\n")
    with pytest.raises(journal.ImportError_, match="does not look like"):
        journal.import_mt5_csv(cfg, "GOLD", "H1", f)


# --- deal review ------------------------------------------------------------------------

def test_review_resolves_deals_and_splits_taken_from_skipped(cfg, tmp_path):
    h1 = candles([(100, 100.5, 99.5, 100)] * 3 + [(100, 103.5, 99.8, 103)]
                 + [(103, 103.2, 102.8, 103)] * 2, start="2026-11-09 00:00")
    base = {"shown_at_utc": "x", "symbol": "GOLD", "order": "market", "rr": "3", "result": ""}
    rows = [
        # buy decided at the 03:00 close: next candle runs to 103.5 -> target hit
        dict(base, candle_close_utc="2026-11-09 03:00", strategy="level", direction="buy",
             entry="100.0", sl="99.0", tp="103.0", taken="yes"),
        # sell at the same moment: stopped out
        dict(base, candle_close_utc="2026-11-09 03:00", strategy="h2l2", direction="sell",
             entry="100.0", sl="101.0", tp="97.0", taken=""),
        # decided after the last candle we have: still waiting
        dict(base, candle_close_utc="2026-11-09 06:00", strategy="level", direction="buy",
             entry="103.0", sl="102.0", tp="106.0", taken=""),
    ]
    log = tmp_path / "deals_log.csv"
    pd.DataFrame(rows + rows[:1]).to_csv(log, index=False)          # one duplicate line
    c = copy.deepcopy(cfg)
    c["symbols"]["GOLD"]["typical_spread"] = 0.0
    table = journal.review_deals(log, {"GOLD": h1}, c)
    assert len(table) == 3                                          # duplicate counted once
    by = {(r["strategy"], r["candle_close_utc"]): r for r in table.to_dict("records")}
    won = by[("level", "2026-11-09 03:00")]
    assert (won["taken"], won["status"], won["r"]) == ("yes", "tp", pytest.approx(3.0))
    lost = by[("h2l2", "2026-11-09 03:00")]
    assert (lost["taken"], lost["status"], lost["r"]) == ("no", "sl", pytest.approx(-1.0))
    assert by[("level", "2026-11-09 06:00")]["status"] == "waiting"


# --- news archive -------------------------------------------------------------------------

def test_calendar_is_archived_once_per_week(cfg):
    from tests.test_feeds import CALENDAR, fake_opener
    monday = pd.Timestamp("2026-11-09 10:00", tz="UTC").timestamp()
    news.load_calendar(cfg, fake_opener(CALENDAR), now_ts=monday + 3 * 86400)   # a Thursday
    from market_analyst.config import cache_dir
    files = [p.name for p in (cache_dir(cfg) / "calendar_archive").iterdir()]
    assert files == ["week_2026-11-09.json"]
