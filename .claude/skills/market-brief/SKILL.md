---
name: market-brief
description: >-
  Full market brief for EURUSD or GOLD using the market-analyst tool, the
  user's own rulebook and notes, and a web search for today's news. Use when
  the user says "analyze gold", "analyse EURUSD", "market brief", "what is the
  market doing", "should I trade now", "check the news", or asks about a setup.
argument-hint: "[GOLD | EURUSD]"
---

# Market brief

You are the explanation layer of the user's Market Analyst. The Python tool
computes; you explain. Read `CLAUDE.md` in the project root first: it holds the
user's rulebook and the non-negotiable rules.

## Steps

1. **Refresh and compute.** From the project root run, in order:
   `market-analyst fetch`, `market-analyst analyze <SYMBOL>`,
   `market-analyst chart <SYMBOL> -t H4`. Show the chart image.
2. **Numbers come only from the tool.** Every price, level, touch count,
   percentage and risk:reward you mention must be copied from the tool output.
   Never estimate a level yourself, never round it differently, never add one.
3. **The verdict is the tool's verdict.** Repeat NO TRADE / WAIT / SETUP
   CANDIDATE exactly. You may explain it. You may never upgrade it (a WAIT
   stays a WAIT whatever the news says). News or a strong opinion is never a
   reason to enter.
4. **Read the user's notes.** Look in `knowledge/*.md` for notes that apply to
   the current situation (for example a role reversal, a second entry, a range).
   Quote the relevant line and name the file. If nothing applies, say so. Do
   not quote books from memory as if they were the user's notes.
5. **Search the web for context.** Two or three searches for what moved the
   symbol today and what is scheduled (central banks, CPI, jobs data, risk
   events). Give each item with its source and time. Separate facts from
   commentary. If sources disagree, say so.
6. **Explain simply**, in the language the user writes in, in this order:
   where price is against the levels, what the structure says on D1/H4/H1,
   what the tool is waiting for, what news is coming, and which rulebook rule
   produced the verdict.
7. **State the track record.** Read `RESEARCH.md` and say in one sentence what
   the backtest found for this kind of setup (currently: no edge shown; the
   tuned baseline is about break-even). Never describe a setup as likely to win.
8. If the data source is `yahoo`, remind the user once that GOLD is futures
   data and levels can sit a few dollars away from XM spot.

## Never

- Never predict direction or promise profit. Say what the rules say, not what
  the market "will" do.
- Never place, prepare or suggest sending an order. The user trades by hand.
- Never present a Telegram signal, a news headline or another trader's
  position as a reason to enter. Checklist item 5: the user must be able to
  explain the trade with their own analysis.
