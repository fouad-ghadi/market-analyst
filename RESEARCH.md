# Research log: what the backtest says

Run on 2026-11-10. Reproduce with `market-analyst backtest` and `market-analyst research`.

## Bottom line

**No strategy shows a real edge on unseen data.** The tuned baseline is about
break-even; everything else lost money in the test. Use the tool to filter out
bad trades and to keep practising on demo. Do not treat any verdict as a
prediction.

## Setup

- Data: free public feed, hourly candles. EURUSD from 2024-03, GOLD (COMEX futures, not XM spot) from 2024-08, both to 2026-11-09.
- Costs: spread 1.6 pips on EURUSD and $0.35 on GOLD, paid on every trade.
- Pessimistic rules: stop assumed hit first when stop and target are in the same candle; weekend gaps exit at the gap price.
- Split: the older 65% of history is searched; the newest 35% is never used to choose anything.
- Results in R (1R = the amount risked). "Range" = 90% bootstrap range of the average.

## 1. The three strategies with rulebook settings (both symbols)

| Strategy | Trades | Win rate | Avg per trade, all history | Avg on newest 35% | Range on newest 35% |
|---|---|---|---|---|---|
| level: reversal candle at a level, enter at close | 684 | 15% | -0.24R | -0.41R | -0.63 to -0.21 |
| retest: enter on the return to the level | 205 | 15% | -0.03R | +0.19R | -0.32 to +0.75 |
| h2l2: Brooks second entry with the H4 trend | 911 | 32% | -0.06R | -0.10R | -0.23 to +0.03 |

Why the baseline lost: stops just beyond a zone are often tiny (stopped by
normal noise within 1 to 2 candles), and on EURUSD the spread alone is about
19% of the median risk.

Retest vs h2l2: neither is reliably better. Retest was positive only on GOLD
and its range is wide (few trades). h2l2 is slightly negative on many trades.

## 2. Tuning (55 / 54 / 42 combinations tried on the older part)

| Strategy | Chosen on the older part | Older part (searched) | Newest part (UNSEEN) | Verdict |
|---|---|---|---|---|
| level | stop at least 1 ATR away, target fixed at 3R, with the D1 trend, 50 EMA sloping with the trade | +0.34R (85 trades) | +0.01R (92 trades), range -0.28 to +0.31 | BREAK-EVEN |
| retest | same stop/target, not against D1 trend, engulfing only, not a dead market | +0.37R (77) | -0.02R (56) | NO EDGE |
| h2l2 | with the D1 trend, 50 EMA slope, not a dead market | +0.13R (129) | -0.17R (56) | NO EDGE |

The gap between the two columns is overfitting: settings always look good on
the data they were picked on.

Level strategy, piece by piece, on the unseen part: rulebook -0.41R; wider stop
plus fixed 3R target only -0.16R; trend conditions only -0.54R; both together
+0.01R.

## 3. What is in config.yaml now

- `stops.min_risk_atr_mult: 1.0`, `stops.tp_mode: fixed_rr`, `stops.fixed_rr: 3.0`
- `filters.level: {trend_D1: [with trend], ema50_slope: true}`
- `filters.retest` and `filters.h2l2` stay empty: nothing held up.

These took the baseline from losing to break-even. To go back to the rulebook:
`min_risk_atr_mult: 0.0`, `tp_mode: next_level`, `filters.level: {}`.

Do not run `research` again on the same data and adopt the new winner: each
extra round fits the past more and the future less. Re-run it when there is
new history (or real MT5 data).

## Not tested

- News filter: no historical calendar is available, so it cannot be backtested.
- Breakeven stops, partial exits, M15/M5 entries.
- Machine-learning models: with under 200 filtered trades they would only memorise noise.
- Real XM prices and spreads: re-run on the PC with MT5 data before drawing conclusions.

## Round 2 (2026-11-10): five markets, trade management, regime switch

Added GBPUSD, AUDUSD and USDJPY as test-only markets (about 2,250 baseline trades
instead of 680), breakeven / partial-exit management, and a D1 regime condition.

| Strategy | Chosen on the older part | Older part (searched) | Newest part (UNSEEN) | Verdict |
|---|---|---|---|---|
| level | not inside the zone, D1 ranging, engulfing only | +0.19R (213 trades) | -0.28R (96) | NO EDGE |
| retest | stop 1.5 ATR, TP at next level, D1 ranging, zone on D1 | +0.74R (44) | -0.15R (39) | NO EDGE |
| h2l2 | with the D1 trend | -0.07R (524) | -0.03R (242) | NO EDGE |

- Trade management (stop to breakeven at +1R or +1.5R, closing half at +1R) was
  never chosen: it did not add 0.05R per trade for any strategy on the older part.
- The regime switch looked good on the older part and failed on the unseen part.
- The settings shipped in config.yaml (round 1), on all five markets:
  510 trades, +0.02R per trade overall, -0.09R on the newest part
  (range -0.27 to +0.10). Positive on EURUSD, GOLD, GBPUSD; negative on AUDUSD, USDJPY.
  Still break-even, so they stay. Nothing from round 2 was adopted.

Conclusion unchanged: these rules, on this data, have no demonstrated edge.
More tuning on the same history will not create one.

Still not tested: M15 entries (the public feed has only 60 days of M15) and a
news filter (the calendar archive started on 2026-11-10; it needs months).
