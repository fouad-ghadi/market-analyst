# Knowledge folder: your notes and book summaries

This is where what you read becomes something the tool can use.

## How a book idea gets into the tool

1. **Write the note here**, in your own words, one file per idea
   (for example `second-entry.md`, `role-reversal.md`). Short is fine.
2. **Make the rule testable.** A rule the code can check has four parts:
   - where: at which kind of level, on which timeframe
   - trigger: what exactly must the candles do (which candle, which close)
   - stop and target: where SL and TP go
   - when it is invalid: what cancels the setup
3. **It goes into `config.yaml` and the code**, as a named strategy.
4. **The backtest judges it** on 1 to 2 years of history, with spread.
   A rule that shows no edge is dropped, whatever the book says.

That loop is the realistic version of "auto-learning": the tool does not
guess or train itself. It measures every rule on history and on your own
journal, and keeps only what holds up.

## Note template

```markdown
# <Name of the idea>
Source: <book, chapter or page> (my own summary)

## The idea in one sentence

## Testable rule
- Where:
- Trigger:
- Stop loss:
- Take profit:
- Invalid when:

## My questions / what I am not sure about
```

Keep full book texts out of this folder if the project ever goes to a public
repository. Your own summaries are fine.
