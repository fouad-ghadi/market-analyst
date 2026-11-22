"""Extra entry conditions ("filters"), written in config.yaml under `filters:`.

A filter spec is a small dict. Each entry is one condition on a fact that is
known when the signal appears (trend alignment, session, zone quality...):

    {"session": ["london", "overlap"]}   the fact must be one of the listed values
    {"min_zone_touches": 3}              zone_touches must be >= 3
    {"max_atr_ratio": 1.3}               atr_ratio must be <= 1.3
    {"zone_d1": True}                    the fact must equal the value

The same code is used by the backtest and by the live analysis.
"""

from __future__ import annotations


def failed_conditions(spec: dict, info: dict) -> list[str]:
    """Human-readable list of the conditions `info` does not meet (empty = all pass).
    A fact that is missing fails its condition."""
    failed = []
    for key, wanted in (spec or {}).items():
        if key.startswith("min_") and key[4:] in info:
            if info[key[4:]] < wanted:
                failed.append(f"{key[4:]} is {info[key[4:]]} (needs at least {wanted})")
        elif key.startswith("max_") and key[4:] in info:
            if info[key[4:]] > wanted:
                failed.append(f"{key[4:]} is {info[key[4:]]} (needs at most {wanted})")
        elif isinstance(wanted, list):
            if info.get(key) not in wanted:
                failed.append(f"{key} is '{info.get(key)}' (needs: {' or '.join(map(str, wanted))})")
        elif info.get(key) != wanted:
            failed.append(f"{key} is {info.get(key)} (needs {wanted})")
    return failed


def make_accept(spec: dict):
    """Turn a filter spec into a yes/no function, or None when there is no filter."""
    if not spec:
        return None
    return lambda info: not failed_conditions(spec, info)
