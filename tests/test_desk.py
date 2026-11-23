"""The analysis desk (`market-analyst desk`): session clock, rulebook
checklist, point-in-time event feed, and the live page."""

import copy
import json
import re
from datetime import datetime, timezone

import pandas as pd

from market_analyst import cli, desk
from market_analyst.analysis import analyze
from market_analyst.backtest import History
from tests.conftest import FakeMT5
from tests.helpers import append, path
from tests.test_analysis import BASE, HAMMER
from tests.test_backtest import make_frames


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


# --- session --------------------------------------------------------------------------------

def test_session_overlap_open_and_weekend(cfg):
    assert desk.session_status(cfg, utc(2026, 11, 23, 14, 0))["overlap"] is True   # Monday 14:00
    before = desk.session_status(cfg, utc(2026, 11, 23, 9, 30))
    assert before["market"] == "open" and not before["overlap"]
    assert "3 h 30 min" in before["next"]
    friday_late = desk.session_status(cfg, utc(2026, 11, 27, 18, 0))
    assert "Monday" not in friday_late["next"] and "h" in friday_late["next"]
    assert desk.session_status(cfg, utc(2026, 11, 28, 12, 0))["market"] == "closed"  # Saturday


# --- checklist ------------------------------------------------------------------------------

def test_checklist_of_a_setup_candidate_passes_the_chart_steps(cfg):
    df = append(path(BASE + [105, 100.4]), [HAMMER])
    a = analyze("GOLD", cfg, {"D1": df, "H4": df, "H1": df})
    assert a.verdict == "SETUP CANDIDATE"
    states = {c["step"]: c["state"] for c in desk.checklist(a, cfg)}
    assert states["At a validated level"] == "pass"
    assert states["Confirmation candle at the level"] == "pass"
    assert states["Risk:reward at least 1:2"] == "pass"


def test_checklist_middle_of_range_fails_step_one(cfg):
    df = path(BASE + [105])
    a = analyze("GOLD", cfg, {"D1": df, "H4": df, "H1": df})
    assert a.verdict == "NO TRADE"
    first = desk.checklist(a, cfg)[0]
    assert (first["step"], first["state"]) == ("At a validated level", "fail")


def test_manual_checks_come_from_config(cfg):
    c = copy.deepcopy(cfg)
    c["risk"]["max_losses_per_day"] = 3
    assert any("after 3 losses" in line for line in desk.manual_checks(c))


# --- event feed: point in time ----------------------------------------------------------------

def test_event_feed_never_uses_later_candles(cfg):
    """The events of a candle must be the same whether or not later candles
    exist: if they changed, the feed would be showing hindsight."""
    frames = make_frames(n=1500)
    f = lambda v: f"{v:.2f}"  # noqa: E731
    full = History("GOLD", cfg, frames)
    cut_time = frames["H1"].index[-60]
    cut = {tf: df[df.index + pd.Timedelta(hours={"H1": 1, "H4": 4, "D1": 24}[tf])
                  <= cut_time + pd.Timedelta(hours=1)] for tf, df in frames.items()}
    part = History("GOLD", cfg, cut)
    ev_full = desk.recent_events(full, cfg, f, hours=200)
    ev_part = desk.recent_events(part, cfg, f, hours=140)
    assert ev_part, "the test needs some events"
    window = (min(e["time"] for e in ev_part), max(e["time"] for e in ev_part))
    in_window = [e for e in ev_full if window[0] <= e["time"] <= window[1]]
    key = lambda evs: sorted((e["time"], e["text"]) for e in evs)  # noqa: E731
    assert key(in_window) == key(ev_part)
    assert [e["time"] for e in ev_full] == sorted((e["time"] for e in ev_full), reverse=True)


# --- the page -----------------------------------------------------------------------------------

def _page_data(page: str) -> dict:
    raw = re.search(r'<script type="application/json" id="desk-data">(.*?)</script>', page, re.S)
    return json.loads(raw.group(1))


def test_desk_page_from_the_public_feed(cfg_file, capsys, monkeypatch):
    frames = make_frames()
    monkeypatch.setattr("market_analyst.cli.fetch_yahoo", lambda sym, cfg: frames)
    monkeypatch.setattr("market_analyst.news.load_calendar", lambda cfg, *a, **k: [])
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    base = ["--config", str(cfg_file), "--source", "yahoo"]
    assert cli.main(base + ["desk", "GOLD", "--fetch"]) == 0
    page = (cfg_file.parent / "reports" / "desk.html").read_text(encoding="utf-8")
    for text in ("Rulebook engine", "Multi-timeframe direction", "Where is price?",
                 "Event feed", "Strategy deals", "Backtest record", "No proven edge",
                 "never places orders", "Your part of the checklist", "Snapshot"):
        assert text in page, text
    assert "http://" not in page and "https://" not in page    # nothing loaded from the web
    assert "<form" not in page and "order_send" not in page    # nothing that could trade
    data = _page_data(page)
    gold = data["symbols"][0]
    assert gold["symbol"] == "GOLD" and set(gold["charts"]) == {"H1", "H4", "D1"}
    assert len(gold["charts"]["H4"]) == 160 and data["refresh_seconds"] == 0
    assert opened and opened[0].endswith("desk.html")


def test_desk_watch_refreshes_live_price_without_redoing_the_analysis(cfg_file, capsys,
                                                                      monkeypatch):
    fake = FakeMT5(n_bars=600)
    monkeypatch.setattr("market_analyst.news.load_calendar", lambda cfg, *a, **k: [])
    monkeypatch.setattr("webbrowser.open", lambda url: None)
    sleeps = []
    monkeypatch.setattr(cli.time, "sleep", lambda s: sleeps.append(s))
    args = cli.build_parser().parse_args(
        ["--config", str(cfg_file), "desk", "GOLD", "--watch", "--cycles", "3", "--no-backtest"])
    cfg = cli.load_config(cfg_file)
    cfg["_source"] = "mt5"
    cfg["data"]["bars"] = {"D1": 600, "H4": 600, "H1": 600}
    assert cli.cmd_desk(args, cfg, fake, sleep=lambda s: sleeps.append(s)) == 0
    out = capsys.readouterr().out
    assert out.count("GOLD: analysed") == 1          # same H1 candle: analysis done once
    assert sleeps == [30, 30]
    page = (cfg_file.parent / "reports" / "desk.html").read_text(encoding="utf-8")
    data = _page_data(page)
    assert data["refresh_seconds"] == 30 and data["symbols"][0]["live"]["bid"] == 1.0
    assert "Live &middot; every 30 s" in page and "live bid" in page


def test_desk_needs_its_config_section(cfg_file, capsys):
    cfg = cli.load_config(cfg_file)
    cfg["_source"] = "mt5"
    del cfg["desk"]
    args = cli.build_parser().parse_args(["desk"])
    try:
        cli.cmd_desk(args, cfg)
    except cli.ConfigError as exc:
        assert "desk:" in str(exc)
    else:
        raise AssertionError("expected a ConfigError")
