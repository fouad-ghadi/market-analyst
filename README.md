# Market Analyst

Personal, **read-only** market analysis assistant for EURUSD and GOLD.
It analyses. It never places orders. See `CLAUDE.md` for the full brief and rulebook.

Status: data, swings + S/R levels, structure + patterns, trade plan, backtest with
three strategies, and a research study. Not built yet: signal checker, journal.

**Read `RESEARCH.md` first.** The backtest found no strategy with a real edge on
unseen data. The shipped settings are the ones that brought the baseline to break-even.

## Two data sources

| Where you run it | Source | Notes |
|---|---|---|
| Your Windows PC with MT5 open | `mt5` | Exact XM prices. The reference. |
| Cloud (no MT5) | `yahoo` | Free public data. GOLD = COMEX futures, a few dollars away from XM spot. |

`data.source: auto` in `config.yaml` picks MT5 when its Python package is installed,
otherwise the public feed. Each source keeps its own cache files.

## Install (Windows, once)

Open **Anaconda Prompt (Miniconda3)**, then:

```bat
cd C:\Users\fouad\market-analyst
conda env create -f environment.yml
conda activate market-analyst
pip install -e .
pytest
```

`pytest` should print `168 passed`. The tests use fake data, so they need neither MT5 nor internet.

## Use

```bat
market-analyst fetch              # download candles into data\
market-analyst analyze            # structure, levels, location, candles, news -> verdict
market-analyst analyze GOLD --no-news
market-analyst levels GOLD -t H4  # list zones with touch counts (--all for every zone)
market-analyst chart GOLD -t H4 D1   # save reports\GOLD_H4_levels.png (same D1+H4 zones as `analyze`)
market-analyst news               # high-impact events this week
market-analyst dashboard --fetch  # EVERYTHING on one web page, opened in your browser
market-analyst deals              # orders the strategies want now: entry / SL / TP + past record
market-analyst review             # what happened to logged deals: taken vs skipped
market-analyst import-csv GOLD H1 C:\\path\\GOLD_H1.csv   # candles exported from MT5
market-analyst backtest           # replay history: level vs retest vs h2l2, with spread
market-analyst backtest GOLD -s level --save   # one symbol/strategy, trades saved to reports\
market-analyst research           # search settings on older data, judge on unseen data (1 min)
```

MT5 only: `market-analyst status`, `specs`, `check-time`.

The verdict is one of:

- **NO TRADE**: a rule fails (middle of the range, risk:reward below 1:2, no target level).
- **WAIT**: nothing is wrong, but price is not at a level or there is no confirmation candle yet.
- **SETUP CANDIDATE**: the chart conditions are met. It is not a trade plan and not a prediction.

## Layout

| Path | What it is |
|---|---|
| `config.yaml` | every rule and number (risk, R:R, tolerances, symbols, data source) |
| `src/market_analyst/config.py` | loads `config.yaml` and rejects dangerous values |
| `src/market_analyst/data.py` | MT5 connection (read-only), symbol specs, server time, CSV cache |
| `src/market_analyst/yahoo.py` | cloud price feed; builds H4/D1 on broker server time |
| `src/market_analyst/indicators.py` | ATR |
| `src/market_analyst/swings.py` | swing highs/lows, confirmed N candles later (no look-ahead) |
| `src/market_analyst/levels.py` | S/R zones from wick tips: touches, clear reaction, score, role reversal |
| `src/market_analyst/structure.py` | uptrend / downtrend / range, structure breaks |
| `src/market_analyst/patterns.py` | doji, hammer, shooting star, engulfing, kept only at a zone |
| `src/market_analyst/analysis.py` | combines everything into a verdict with reasons |
| `src/market_analyst/planner.py` | entry / SL / TP / risk:reward arithmetic |
| `src/market_analyst/backtest.py` | candle-by-candle replay engine, fills, exits, statistics |
| `src/market_analyst/strategies.py` | level, retest and h2l2 strategies |
| `src/market_analyst/filters.py` | extra entry conditions from `config.yaml` |
| `src/market_analyst/research.py` | in-sample search, out-of-sample verdict |
| `src/market_analyst/journal.py` | MT5 file import, review of logged deals |
| `src/market_analyst/dashboard.py` | the web page (one self-contained HTML file, nothing online) |
| `src/market_analyst/deals.py` | finds current deals and their past record (shared by `deals` and `dashboard`) |
| `src/market_analyst/news.py` | weekly economic calendar (information only) |
| `src/market_analyst/report.py` | text report and chart image |
| `src/market_analyst/cli.py` | the `market-analyst` command |
| `tests/` | pytest suite on hand-built candles and fake feeds |
| `knowledge/` | your notes and book summaries (see `knowledge/README.md`) |
| `.claude/skills/market-brief/` | tells Claude how to run a full brief (tool + notes + web news) |
