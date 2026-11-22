"""Guard rails from CLAUDE.md section 5. These tests fail if a rule is broken."""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

# Any MT5 function that can create, change or close a trade.
TRADING_CALLS = re.compile(
    r"\.\s*(order_send|order_check)\b"          # mt5.order_send(...)
    r"|\b(order_send|order_check)\("            # order_send(...)
    r"|[\"'](order_send|order_check)[\"']"      # getattr(mt5, "order_send")
)

# Things that look like a hard-coded login.
CREDENTIALS = re.compile(r"\b(password|login)\s*=\s*[\"'\d]", re.IGNORECASE)


def _source_files():
    return sorted(SRC.rglob("*.py"))


def test_source_files_exist():
    assert _source_files(), "no source files found: test is misconfigured"


def test_no_code_sends_orders():
    offenders = [f"{p.name}:{i}" for p in _source_files()
                 for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
                 if TRADING_CALLS.search(line)]
    assert not offenders, f"Order-sending code found (forbidden): {offenders}"


def test_no_hardcoded_credentials():
    offenders = [f"{p.name}:{i}" for p in _source_files()
                 for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
                 if CREDENTIALS.search(line)]
    assert not offenders, f"Possible hard-coded credentials: {offenders}"


def test_env_file_is_gitignored():
    gitignore = (SRC.parent / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in gitignore
