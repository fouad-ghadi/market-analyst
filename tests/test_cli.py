"""End-to-end: run the CLI commands against the fake terminal."""

from datetime import datetime, timezone

from market_analyst.cli import main
from tests.conftest import FakeMT5


def test_status(cfg_file, capsys):
    assert main(["--config", str(cfg_file), "status"], mt5_module=FakeMT5()) == 0
    out = capsys.readouterr().out
    assert "DEMO" in out and "XMGlobal-MT5 10" in out and "WARNING" not in out


def test_specs(cfg_file, capsys):
    assert main(["--config", str(cfg_file), "specs", "GOLD"], mt5_module=FakeMT5()) == 0
    out = capsys.readouterr().out
    assert "GOLD" in out and "pip value per lot" in out and "EURUSD" not in out


def test_fetch_writes_one_csv_per_symbol_and_timeframe(cfg_file, capsys):
    assert main(["--config", str(cfg_file), "fetch"], mt5_module=FakeMT5()) == 0
    files = sorted(p.name for p in (cfg_file.parent / "data").glob("*.csv"))
    assert files == [f"{s}_{tf}.csv" for s in ("EURUSD", "GOLD") for tf in ("D1", "H1", "H4")]
    assert "100 bars" in capsys.readouterr().out


def test_fetch_subset(cfg_file):
    assert main(["--config", str(cfg_file), "fetch", "GOLD", "-t", "H4"],
                mt5_module=FakeMT5()) == 0
    assert [p.name for p in (cfg_file.parent / "data").glob("*.csv")] == ["GOLD_H4.csv"]


def test_unknown_symbol_is_an_error(cfg_file, capsys):
    assert main(["--config", str(cfg_file), "fetch", "BTCUSD"], mt5_module=FakeMT5()) == 2
    assert "BTCUSD" in capsys.readouterr().err


def test_mt5_not_running(cfg_file, capsys):
    assert main(["--config", str(cfg_file), "status"], mt5_module=FakeMT5(init_ok=False)) == 2
    assert "Could not connect" in capsys.readouterr().err


def test_check_time_on_weekend(cfg_file, capsys):
    friday = datetime(2026, 10, 30, 20, 58, tzinfo=timezone.utc).timestamp()
    fake = FakeMT5(tick_time=int(friday) + 3 * 3600)
    assert main(["--config", str(cfg_file), "check-time"], mt5_module=fake) == 0
    assert "market closed" in capsys.readouterr().out


TUESDAY = datetime(2026, 11, 3, 14, 0, tzinfo=timezone.utc)   # market open, winter time


def test_check_time_match(cfg_file, capsys, monkeypatch):
    monkeypatch.setattr("market_analyst.cli.utc_now", lambda: TUESDAY)
    fake = FakeMT5(tick_time=int(TUESDAY.timestamp()) + 2 * 3600)  # server at UTC+2
    assert main(["--config", str(cfg_file), "check-time"], mt5_module=fake) == 0
    assert "OK: config matches" in capsys.readouterr().out


def test_check_time_mismatch(cfg_file, capsys, monkeypatch):
    monkeypatch.setattr("market_analyst.cli.utc_now", lambda: TUESDAY)
    fake = FakeMT5(tick_time=int(TUESDAY.timestamp()) + 9 * 3600)  # server at UTC+9
    assert main(["--config", str(cfg_file), "check-time"], mt5_module=fake) == 1
    assert "MISMATCH" in capsys.readouterr().out


# --- analysis commands (milestones 2 and 3) -------------------------------------------

def _fetched(cfg_file):
    """Fill the cache from the fake terminal, like running `fetch` on your PC."""
    fake = FakeMT5(n_bars=300)
    assert main(["--config", str(cfg_file), "fetch"], mt5_module=fake) == 0
    return fake


def test_levels_lists_zones(cfg_file, capsys):
    fake = _fetched(cfg_file)
    capsys.readouterr()
    assert main(["--config", str(cfg_file), "levels", "GOLD", "-t", "H4"], mt5_module=fake) == 0
    out = capsys.readouterr().out
    assert "=== GOLD H4" in out and "zones" in out and "price" in out


def test_chart_writes_a_png(cfg_file, capsys):
    fake = _fetched(cfg_file)
    assert main(["--config", str(cfg_file), "chart", "GOLD"], mt5_module=fake) == 0
    png = cfg_file.parent / "reports" / "GOLD_H4_levels.png"
    assert png.exists() and png.stat().st_size > 10_000


def test_analyze_prints_a_verdict_for_every_symbol(cfg_file, capsys):
    fake = _fetched(cfg_file)
    capsys.readouterr()
    assert main(["--config", str(cfg_file), "analyze", "--no-news"], mt5_module=fake) == 0
    out = capsys.readouterr().out
    assert out.count("VERDICT:") == 2
    assert "=== EURUSD analysis" in out and "=== GOLD analysis" in out
    assert "HIGH-IMPACT NEWS" not in out


def test_analyze_without_data_tells_you_to_fetch(cfg_file, capsys):
    assert main(["--config", str(cfg_file), "analyze", "--no-news"], mt5_module=FakeMT5()) == 2
    assert "market-analyst fetch" in capsys.readouterr().err


def test_news_command(cfg_file, capsys, monkeypatch):
    rows = [{"title": "CPI m/m", "country": "USD", "date": "2026-11-10T08:30:00-05:00",
             "impact": "High", "forecast": "0.3%", "previous": "0.6%"}]
    monkeypatch.setattr("market_analyst.news.load_calendar", lambda cfg, *a, **k: rows)
    assert main(["--config", str(cfg_file), "news", "GOLD"], mt5_module=FakeMT5()) == 0
    out = capsys.readouterr().out
    assert "CPI m/m" in out and "Tue 10 Nov 13:30 UTC" in out


def test_mt5_only_commands_explain_themselves_in_the_cloud(cfg_file, capsys):
    assert main(["--config", str(cfg_file), "--source", "yahoo", "status"]) == 2
    assert "needs your MT5 terminal" in capsys.readouterr().err


def test_cloud_fetch_keeps_its_own_cache_files(cfg_file, capsys, monkeypatch):
    from tests.helpers import path
    df = path([105, 110, 100, 110.1, 100.1, 105])
    monkeypatch.setattr("market_analyst.cli.fetch_yahoo",
                        lambda sym, cfg: {"H1": df, "H4": df, "D1": df})
    assert main(["--config", str(cfg_file), "--source", "yahoo", "fetch", "GOLD"]) == 0
    files = sorted(p.name for p in (cfg_file.parent / "data").glob("*.csv"))
    assert files == ["GOLD_D1.yahoo.csv", "GOLD_H1.yahoo.csv", "GOLD_H4.yahoo.csv"]
    assert "not your broker" in capsys.readouterr().out
    # ... and the analysis reads them back
    assert main(["--config", str(cfg_file), "--source", "yahoo", "analyze", "GOLD",
                 "--no-news"]) == 0
    assert "VERDICT: NO TRADE" in capsys.readouterr().out


def test_backtest_with_too_little_history_explains_itself(cfg_file, capsys):
    fake = _fetched(cfg_file)          # only 100 candles per timeframe
    assert main(["--config", str(cfg_file), "backtest", "GOLD"], mt5_module=fake) == 2
    assert "Not enough GOLD history" in capsys.readouterr().err


def test_backtest_command(cfg_file, capsys, monkeypatch):
    from tests.test_backtest import make_frames
    frames = make_frames()
    monkeypatch.setattr("market_analyst.cli.fetch_yahoo", lambda sym, cfg: frames)
    base = ["--config", str(cfg_file), "--source", "yahoo"]
    assert main(base + ["fetch", "GOLD"]) == 0
    capsys.readouterr()
    assert main(base + ["backtest", "GOLD", "-s", "h2l2", "--save"]) == 0
    out = capsys.readouterr().out
    assert "=== h2l2" in out and "out-of-sample" in out and "describes the past" in out
    assert (cfg_file.parent / "reports" / "backtest_trades.csv").exists()


def test_deals_command_lists_and_logs(cfg_file, capsys, monkeypatch):
    from market_analyst.backtest import Order
    from tests.test_backtest import make_frames
    frames = make_frames()
    monkeypatch.setattr("market_analyst.cli.fetch_yahoo", lambda sym, cfg: frames)
    base = ["--config", str(cfg_file), "--source", "yahoo"]
    assert main(base + ["fetch", "GOLD"]) == 0
    price = float(frames["H1"]["close"].iloc[-1])
    monkeypatch.setattr("market_analyst.backtest.latest_order", lambda h, s, w=400: Order(
        "stop", "buy", sl=price - 1, tp=price + 3, price=price + 0.5, info={"pattern": "H2"}))
    capsys.readouterr()
    assert main(base + ["deals", "GOLD"]) == 0
    out = capsys.readouterr().out
    assert "BUY STOP at" in out and "R:R 1:1.7" in out and "risk at 0.01 lot about $1.50" in out
    assert "not predictions" in out and "DEMO only" in out
    log = (cfg_file.parent / "reports" / "deals_log.csv").read_text()
    assert log.count("GOLD") == 3 and "taken" in log


def test_dashboard_builds_one_self_contained_page(cfg_file, capsys, monkeypatch):
    from tests.test_backtest import make_frames
    frames = make_frames()
    monkeypatch.setattr("market_analyst.cli.fetch_yahoo", lambda sym, cfg: frames)
    monkeypatch.setattr("market_analyst.news.load_calendar", lambda cfg, *a, **k: [])
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    base = ["--config", str(cfg_file), "--source", "yahoo"]
    assert main(base + ["fetch", "GOLD"]) == 0
    assert main(base + ["dashboard", "GOLD"]) == 0
    page = (cfg_file.parent / "reports" / "dashboard.html").read_text(encoding="utf-8")
    for text in ("Rulebook verdict", "Deals right now", "Market structure", "Key levels",
                 "Backtest record on GOLD", "data:image/png;base64,", "never places orders",
                 "never places orders", "not predictions", "Decision support only"):
        assert text in page
    assert "http://" not in page and "https://" not in page     # nothing loaded from the web
    assert opened and opened[0].endswith("dashboard.html")
