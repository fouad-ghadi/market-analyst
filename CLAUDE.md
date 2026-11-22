# Market Analyst: Project Brief for Claude Code

Put this file at the root of the project folder (e.g. `C:\Users\fouad\market-analyst\CLAUDE.md`).
It gives you (Claude Code) the full context: who I am, what I've learned, my trading rules,
and what we are building together.

---

## 1. Who I am

- Engineering student, **beginner trader** (started summer 2026). Goal = side income, 1–2 h/day.
- I practice on an **XM MetaTrader 5 demo account** (server `XMGlobal-MT5 10`, hedge mode, ~$10,000 demo balance) on **Windows**.
- Already installed on my PC: **Git**, **Miniconda** (a Python 3.12 env works), the MT5 desktop terminal.
- Also cloned: `C:\Users\fouad\TradingAgents` (TauricResearch multi-agent research framework, stocks/crypto via Yahoo Finance, needs an LLM API key, not connected to MT5). Treat it as a reference only, not a dependency.
- Basic Python. Explain what the code does as you write it, in simple terms. I learn by understanding, not by copy-pasting.
- I write in English or French. Answer in the language I use.

---

## 2. What I have learned so far (my rulebook)

This is the method I was taught step by step. The tool must **encode these rules**, not invent new ones.

### Market and timing
- Main symbols: **EURUSD** (learning pair) and **GOLD** (XM's symbol name; XAUUSD elsewhere).
- Best session: London–New York overlap (about 13:00–17:00 UTC). Market closed on weekends.
- MT5 timestamps are in **broker server time** (XM showed about UTC+3 vs my local time). Handle the offset in config.

### Reading charts
- Candlesticks: body = open→close, wicks = high/low. Key patterns: **doji, hammer, shooting star, bullish/bearish engulfing**.
- **A pattern only matters at a level.** A hammer in the middle of nowhere is noise.

### Support and Resistance (S/R)
- Draw levels on **wick tips**, not candle bodies.
- A level needs **at least 2 touches with a clear reaction** to be "validated". 1 touch = "level to watch".
- Levels are **zones**, not exact lines (price can overshoot a few pips).
- Recent touches matter more than old ones. Higher-timeframe levels are stronger.
- **Role reversal:** broken resistance becomes support, and vice versa (break and retest).
- A range can have several layers of resistance/support (inner and outer levels).

### Market structure
- Uptrend = Higher Highs + Higher Lows. Downtrend = Lower Highs + Lower Lows. Otherwise = range.
- A trend is in question when structure breaks (first lower low in an uptrend, first higher high in a downtrend).
- **Multi-timeframe:** D1/H4 = direction and main levels, H1 = confirmation, M15/M5 = entry timing only. M1 is noise.

### Strategy: "S/R + candlestick confirmation"
1. Mark validated levels on H4 (and D1).
2. Wait for price to reach a level.
3. Wait for a reversal candle **at** that level (hammer/engulfing at support, shooting star/engulfing at resistance).
4. Enter on the close of the confirmation candle, or on a retest (second entry).
5. **Price in the middle of the range = NO TRADE.**
6. Prefer trading **with** the H4 trend. Counter-trend trades = lower conviction.

### Stop Loss / Take Profit
- SL: a few pips **beyond** the level (below support for a buy, above resistance for a sell).
- TP: **just before** the next opposite level.
- **Minimum Risk:Reward = 1:2.** If the next level is too close to give 1:2, skip the trade.
- Never trade without SL. Never move SL further away. Moving SL to breakeven once in profit is allowed.

### Risk management
- Risk **1% max per trade** (learning phase: fixed 0.01 lot on demo).
- Position size: `lots = (balance × risk%) / (SL distance in pips × pip value per lot)`.
- Max 1 open trade at a time while learning. Stop for the day after 2 losses (or −3% on the day).
- No revenge trading: never increase size to recover a loss.

### Order types (MT5)
- Buy Limit = buy **below** current price. Sell Limit = sell **above** current price.
- Buy Stop = buy **above** current price (breakout). Sell Stop = sell **below** current price.
- Pending orders: set expiration to "Today" so a stale setup doesn't trigger later.

### Pre-trade checklist (every box must be "yes")
1. Is the entry at a validated level (not the middle of the range)?
2. Is the SL based on chart structure?
3. Is R:R at least 1:2?
4. Am I risking my planned %, not more?
5. Can I explain WHY I'm entering with my own analysis (not a Telegram signal)?
6. Am I calm, not chasing a loss or a move?

---

## 3. My trading record so far (demo)

| Date | Symbol | Source | Entry | SL | TP | Result | Lesson |
|---|---|---|---|---|---|---|---|
| 2026-07-20 | EURUSD buy limit | My analysis | 1.14320 | 1.14180 | 1.14700 | −1.42 (SL) | Good process, support failed. A "good loss". |
| 2026-07-20 | GOLD buy | Telegram signal | 4031.65 | 4028.00 | 4051.00 | −3.81 (SL) | SL too tight, no level behind the signal. |
| 2026-07-27 | GOLD buy ×2 | Telegram signal, market entry | 4090.72 / 4091.80 | 4078.00 | 4109.00 | +18.40 / +17.32 (TP) | Won by luck: no level, chased entry, R:R 1:1.25. |
| 2026-08-04 | GOLD buy | — | 4070.26 | 4049.00 | 4079.00 | −21.26 (SL) | **R:R 1:0.4** (risk 21, reward 9). Biggest loss. |
| 2026-08-27 | GOLD buy | — | 4621.98 | 4609.14 | 4669.96 | −3.87 | Closed manually before SL/TP. |
| 2026-10-06 | GOLD sell limit (planned) | My analysis | 4170.00 | 4179.00 | 4152.00 | not recorded | Sell at H1 double-top resistance, R:R 1:2. |

Net so far: about **+$5** on $10,000. Most losses come from **breaking my own rules** (signals, bad R:R, chasing),
not from the method. The most useful thing this tool can do is **stop me from taking trades that fail the checklist.**

---

## 4. The project: what we are building

A **personal market analysis assistant** for EURUSD and GOLD that:

1. Pulls price data from my MT5 terminal.
2. Detects S/R zones, swing structure, trend, and candlestick patterns **using the rules above**.
3. Produces a clear analysis report plus a chart image with the levels drawn.
4. Plans a trade (entry / SL / TP / R:R / lot size) **only if** the checklist passes. Otherwise it says "NO TRADE" and why.
5. Checks any external signal (e.g. from Telegram) against the chart and my rules, and gives a verdict.
6. Keeps a trade journal (imported from MT5 history) with stats and rule compliance.
7. Backtests the strategy on historical data so I know whether it has an edge **before** I trust it.
8. Later: a knowledge base built from trading books I own, so explanations can reference what I studied.

### What it is NOT
- **Not a money-printing bot.** It must never claim it can predict the market or guarantee profit.
- **Not an auto-trader.** Analysis and decision support only. I place orders myself in MT5.
- Not "training a model on books". See section 8 for the realistic approach.

---

## 5. Non-negotiable rules for you (Claude Code)

- **Never write code that sends orders** (`mt5.order_send` or similar). MT5 access is read-only. If I ask for order sending later, remind me of this rule and ask me to confirm, demo account only.
- **Never hardcode credentials.** The MT5 terminal is already logged in, and `mt5.initialize()` attaches to it. Any API key goes in a `.env` file listed in `.gitignore`.
- **No look-ahead bias** in detection or backtests: at bar *t*, only use data up to bar *t*. Swing points are only confirmed N bars later; respect that.
- Every numeric rule (touch count, tolerance, min R:R, risk %, SL buffer, etc.) lives in **one config file**, not scattered in the code.
- Build in **small milestones**. Each milestone must run end to end and have tests before moving on.
- Ask before adding a new dependency. Keep it simple.
- Report results honestly. If a backtest shows no edge, say so clearly.
- After each milestone, explain the code to me (what each module does, how to run it).

---

## 6. Suggested architecture

```
market-analyst/
├── CLAUDE.md
├── config.yaml              # all rules & parameters (symbols, timeframes, risk, R:R, tolerances)
├── .env.example             # optional API keys (never commit .env)
├── data/                    # cached OHLC (parquet/csv), journal db
├── knowledge/               # my own notes & book summaries (markdown), later
├── src/market_analyst/
│   ├── data.py              # MT5 connection, OHLC fetch, symbol specs, caching, server-time offset
│   ├── indicators.py        # ATR, simple helpers
│   ├── swings.py            # swing highs/lows (fractals), confirmed with delay
│   ├── levels.py            # S/R zone clustering from wick tips, touch counting, scoring
│   ├── structure.py         # HH/HL/LH/LL, trend / range classification, structure breaks
│   ├── patterns.py          # doji, hammer, shooting star, engulfing (flagged only at levels)
│   ├── planner.py           # entry/SL/TP/R:R/lot size + checklist → TRADE or NO TRADE (+ reasons)
│   ├── signal_check.py      # evaluate an external signal (entry/SL/TP) against chart & rules
│   ├── journal.py           # import MT5 history, store trades, stats (win rate, avg R, expectancy)
│   ├── backtest.py          # walk the strategy over history, include spread, report metrics
│   ├── report.py            # markdown/HTML report + chart image with levels drawn
│   └── cli.py               # commands: analyze, plan, check-signal, journal, backtest
└── tests/
```

**Stack:** Python 3.12 (conda env), `MetaTrader5` (official MT5 Python package, Windows only, terminal must be running),
`pandas`, `numpy`, `mplfinance` or `plotly` for charts, `sqlite3` for the journal, `pytest`.
CLI first. A small web dashboard (e.g. Streamlit) only later, if useful.
Data source = MT5 only. (TradingView "MCP" bridges are unofficial community projects; skip them.)

### Precise definitions to implement (tunable in config)
- **Swing high/low:** fractal with N bars on each side (default N = 3). Confirmed only after N more bars close.
- **S/R zone:** cluster swing wick extremes that lie within `tolerance = k × ATR(14)` (default k = 0.25).
  `touches >= 2` → validated, `touches = 1` → watch. Score = touches + recency weight + timeframe weight (D1 > H4 > H1).
- **Role reversal:** a zone broken by a candle **close** flips type (resistance → support, support → resistance).
- **Trend:** compare the last two confirmed swing highs and lows (HH+HL = up, LH+LL = down, else range).
- **Middle of range:** price in the middle 40% between the nearest validated support and resistance → NO TRADE.
- **SL:** beyond the zone edge by a buffer = `max(2 × spread, 0.1 × ATR)`.
- **TP:** just before the next opposite zone edge (minus the same buffer).
- **Lot size:** use `mt5.symbol_info()` (`trade_tick_value`, `trade_tick_size`, `volume_min`, `volume_step`).
  `lots = risk_$ / (SL_distance / tick_size × tick_value)`, rounded **down** to `volume_step`.
  If even `volume_min` exceeds the risk budget → NO TRADE.

---

## 7. Build order (milestones)

1. **Setup + data:** project skeleton, conda env, connect to MT5, fetch EURUSD and GOLD on D1/H4/H1, cache to disk, print symbol specs. Test with MT5 open.
2. **Swings + S/R levels:** detect swings and zones, print the levels with touch counts. Draw them on a chart image so I can compare with my own lines.
3. **Structure + patterns:** trend/range classification per timeframe. Candlestick patterns, flagged only near a zone.
4. **Planner + checklist:** `analyze GOLD` → report: H4 bias, key levels, where price is, and either a full trade plan (entry/SL/TP/R:R/lots) or "NO TRADE" with the reasons.
5. **Signal checker:** `check-signal GOLD buy 4087 sl 4078 tp 4109` → verdict against the rules (is the entry at a level? R:R? with or against the trend?).
6. **Journal:** import my MT5 history (`mt5.history_deals_get`), compute stats, flag which trades broke which rule.
7. **Backtest:** run the strategy on 1–3 years of H1/H4 data with spread. Report number of trades, win rate, average R, expectancy, max drawdown. Be honest about the results.
8. **Knowledge base (optional, later):** see section 8.
9. **Optional LLM layer:** use the Anthropic API (key in `.env`) to turn the computed analysis into a plain-language explanation that cites my rulebook and notes. The **numbers always come from the code**, never from the LLM. The tool must fully work without an API key.

---

## 8. "Trained with trading books": the realistic approach

We will **not** train or fine-tune a model. Instead:

1. **Rules → code.** When I read a book, I extract concrete, testable rules (e.g. "pin bar at H4 support in an uptrend"). We add them to `config.yaml` and the detection modules, then **backtest** them. A rule that doesn't survive the backtest gets dropped, whatever the book says.
2. **Notes → knowledge folder.** I write my own summaries and notes in `knowledge/*.md` (books I own, personal use; don't commit full book texts to any public repo).
3. **Retrieval (later).** The optional LLM layer searches those notes and quotes the relevant rule when explaining an analysis ("this setup matches the role-reversal rule from my notes").

---

## 9. First session: start here

1. Read this file and summarize back to me, in a few lines, what you understood (so I can correct you).
2. Propose the project skeleton and the `config.yaml` with the rules from section 2.
3. Build **milestone 1** only (setup + data), explain each file, and tell me exactly how to run it on my machine.
