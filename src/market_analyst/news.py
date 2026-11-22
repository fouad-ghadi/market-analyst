"""Economic news calendar (scheduled events such as NFP, CPI, rate decisions).

Source: the free weekly ForexFactory calendar export. It lists every
scheduled release for the current week with its currency and impact level.

This is INFORMATION, not a rulebook rule: the analysis lists high-impact
events coming up for the symbol's currencies so you are not surprised by a
spike. Whether to stand aside around news is your decision to add to the rules.

The file is cached for an hour: the publisher asks tools not to hammer it.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

from .config import cache_dir

URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


class NewsError(RuntimeError):
    """Raised when the calendar cannot be read."""


def download_calendar(opener=None) -> list[dict]:
    opener = opener or urllib.request.urlopen
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (market-analyst)"})
    try:
        with opener(req, timeout=30) as resp:
            return json.load(resp)
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise NewsError(f"Could not download the economic calendar: {exc}") from exc


def load_calendar(cfg: dict, opener=None, now_ts: float | None = None) -> list[dict]:
    """This week's raw events, from the cache if it is fresh enough."""
    path = cache_dir(cfg) / "calendar_thisweek.json"
    now_ts = now_ts or time.time()
    max_age = cfg["news"]["cache_minutes"] * 60
    if path.exists() and now_ts - path.stat().st_mtime < max_age:
        return json.loads(path.read_text(encoding="utf-8"))
    events = download_calendar(opener)
    path.write_text(json.dumps(events), encoding="utf-8")
    archive(cfg, events, now_ts)
    return events


def archive(cfg: dict, events: list[dict], now_ts: float) -> None:
    """Keep one file per week in data/calendar_archive/. The feed only offers
    the current week, so saving it every week is the only way to build the
    history needed to backtest a news filter later."""
    today = datetime.fromtimestamp(now_ts, timezone.utc).date()
    monday = today - timedelta(days=today.weekday())
    folder = cache_dir(cfg) / "calendar_archive"
    folder.mkdir(exist_ok=True)
    (folder / f"week_{monday:%Y-%m-%d}.json").write_text(json.dumps(events), encoding="utf-8")


def parse_events(raw: list[dict]) -> list[dict]:
    """Normalise raw rows: time in UTC, tidy keys. Rows without a time are skipped."""
    events = []
    for row in raw:
        try:
            when = datetime.fromisoformat(row["date"]).astimezone(timezone.utc)
        except (KeyError, ValueError, TypeError):
            continue
        events.append({
            "time_utc": when,
            "currency": row.get("country", ""),
            "impact": row.get("impact", ""),
            "title": row.get("title", ""),
            "forecast": row.get("forecast", ""),
            "previous": row.get("previous", ""),
        })
    return sorted(events, key=lambda e: e["time_utc"])


def events_for_symbol(events: list[dict], symbol: str, cfg: dict) -> list[dict]:
    """Keep events for the symbol's currencies at the configured impact levels."""
    currencies = set(cfg["symbols"][symbol].get("news_currencies", []))
    impacts = set(cfg["news"]["impact"])
    return [e for e in events if e["currency"] in currencies and e["impact"] in impacts]


def upcoming(events: list[dict], now_utc: datetime, hours: float) -> list[dict]:
    """Events between now and now + hours."""
    end = now_utc + timedelta(hours=hours)
    return [e for e in events if now_utc <= e["time_utc"] <= end]


def describe(event: dict, now_utc: datetime | None = None) -> str:
    text = f"{event['time_utc']:%a %d %b %H:%M} UTC  {event['currency']}  {event['title']}"
    if event["forecast"] or event["previous"]:
        text += f"  (forecast {event['forecast'] or '-'}, previous {event['previous'] or '-'})"
    if now_utc is not None:
        hours = (event["time_utc"] - now_utc).total_seconds() / 3600
        text += f"  [in {hours:.1f} h]" if hours >= 0 else "  [passed]"
    return text
