"""Load and check config.yaml.

The rest of the code never hard-codes a rule. It asks the config instead,
e.g. ``cfg["stops"]["min_risk_reward"]``. This module:

1. finds the project folder (the one that contains config.yaml),
2. reads the YAML file into a plain Python dict,
3. checks that the important values exist and make sense, so a typo in the
   config stops the program with a clear message instead of a silent bug.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Project root = two folders above this file: src/market_analyst/config.py -> project/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Timeframes the tool understands. M1 is deliberately absent (rulebook: "M1 is noise").
VALID_TIMEFRAMES = {"M5", "M15", "M30", "H1", "H4", "D1", "W1"}


# Rulebook settings that the code applies in exactly one way (see validate_config).
FIXED_RULES = {
    "levels.source": "wicks",                         # levels.py uses wick tips
    "levels.min_touches_watch": 1,                    # any zone below "validated" is "watch"
    "levels.role_reversal_on_close": True,            # levels.role_for uses the last CLOSE
    "structure.swings_compared": 2,                   # structure.py compares the last 2 swings
    "patterns.engulfing_body_must_engulf": True,      # patterns.py requires a full engulf
    "stops.tp_uses_same_buffer": True,                # planner.py uses one buffer for SL and TP
    "strategy.prefer_with_trend": True,               # counter-trend = "lower conviction" label
    "risk.round_lots_down": True,                     # no %-based lot sizing yet (fixed lot)
}


class ConfigError(ValueError):
    """Raised when config.yaml is missing a value or contains an impossible one."""


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read config.yaml and return it as a dict, after validation."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    validate_config(cfg)
    # Remember where the config came from, so relative paths (e.g. cache_dir)
    # are resolved next to it, whatever folder you run the command from.
    cfg["_base_dir"] = str(path.resolve().parent)
    return cfg


def _require(cfg: dict, dotted: str) -> Any:
    """Return cfg["a"]["b"] for dotted="a.b", or raise a readable ConfigError."""
    node: Any = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            raise ConfigError(f"Missing config value: '{dotted}'")
        node = node[key]
    return node


def validate_config(cfg: dict) -> None:
    """Sanity checks. Each check protects one rule from the rulebook."""
    # --- symbols ---
    symbols = _require(cfg, "symbols")
    if not symbols:
        raise ConfigError("'symbols' must list at least one symbol")
    for name, spec in symbols.items():
        if not spec or spec.get("pip_size", 0) <= 0:
            raise ConfigError(f"symbols.{name}.pip_size must be > 0")

    # --- time ---
    mode = _require(cfg, "time.server_time_mode")
    if mode not in ("new_york_close", "fixed"):
        raise ConfigError("time.server_time_mode must be 'new_york_close' or 'fixed'")
    offset = _require(cfg, "time.server_utc_offset_hours")
    if not -12 <= offset <= 14:
        raise ConfigError("time.server_utc_offset_hours must be between -12 and 14")

    # --- data ---
    if cfg["data"].get("source", "auto") not in ("auto", "mt5", "yahoo"):
        raise ConfigError("data.source must be 'auto', 'mt5' or 'yahoo'")
    bars = _require(cfg, "data.bars")
    for tf, n in bars.items():
        if tf not in VALID_TIMEFRAMES:
            raise ConfigError(f"data.bars: unknown timeframe '{tf}' (M1 is not allowed)")
        if not isinstance(n, int) or n <= 0:
            raise ConfigError(f"data.bars.{tf} must be a positive integer")

    # --- rules that protect your money ---
    if _require(cfg, "stops.min_risk_reward") < 1.0:
        raise ConfigError("stops.min_risk_reward below 1.0 makes no sense (rulebook says 2.0)")
    risk_pct = _require(cfg, "risk.risk_pct_per_trade")
    if not 0 < risk_pct <= 1.0:
        raise ConfigError("risk.risk_pct_per_trade must be > 0 and <= 1.0 (rulebook: 1% max)")
    if _require(cfg, "risk.max_open_trades") < 1:
        raise ConfigError("risk.max_open_trades must be >= 1")
    if _require(cfg, "stops.allow_widen_sl"):
        raise ConfigError("stops.allow_widen_sl must stay false (never move the SL further away)")
    if _require(cfg, "risk.allow_size_increase_after_loss"):
        raise ConfigError("risk.allow_size_increase_after_loss must stay false (no revenge trading)")
    if cfg["stops"].get("tp_mode", "next_level") not in ("next_level", "fixed_rr"):
        raise ConfigError("stops.tp_mode must be 'next_level' or 'fixed_rr'")
    if cfg["stops"].get("tp_mode") == "fixed_rr" and \
            cfg["stops"].get("fixed_rr", 0) < cfg["stops"]["min_risk_reward"]:
        raise ConfigError("stops.fixed_rr must be at least stops.min_risk_reward")
    if cfg["stops"].get("min_risk_atr_mult", 0) < 0:
        raise ConfigError("stops.min_risk_atr_mult cannot be negative")
    pct = _require(cfg, "strategy.middle_of_range_pct")
    if not 0 <= pct < 100:
        raise ConfigError("strategy.middle_of_range_pct must be between 0 and 100")
    if not _require(cfg, "stops.allow_move_sl_to_breakeven") and \
            cfg["stops"].get("manage_at_r", 0) > 0 and cfg["stops"].get("breakeven_after_manage"):
        raise ConfigError("stops.breakeven_after_manage moves the stop to breakeven, but "
                          "stops.allow_move_sl_to_breakeven is false")
    modes = set(_require(cfg, "strategy.entry_modes"))
    if not modes <= {"confirmation_close", "retest"}:
        raise ConfigError("strategy.entry_modes can only contain confirmation_close and retest")

    # --- settings the code implements in one way only ---
    # They are in config.yaml to document the rule. Changing one would do
    # nothing, which is worse than an error, so we refuse other values.
    for dotted, value in FIXED_RULES.items():
        if _require(cfg, dotted) != value:
            raise ConfigError(f"{dotted} must stay {value!r}: the code only implements that "
                              "rule. To change it, the detection code has to change too.")


def output_dir(cfg: dict) -> Path:
    """Absolute path of the folder for reports and chart images (created if missing)."""
    p = Path(cfg.get("report", {}).get("output_dir", "reports"))
    if not p.is_absolute():
        p = Path(cfg.get("_base_dir", PROJECT_ROOT)) / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir(cfg: dict) -> Path:
    """Absolute path of the data cache folder (created if missing)."""
    p = Path(cfg["data"]["cache_dir"])
    if not p.is_absolute():
        p = Path(cfg.get("_base_dir", PROJECT_ROOT)) / p
    p.mkdir(parents=True, exist_ok=True)
    return p
