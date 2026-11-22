"""Allows `python -m market_analyst ...` to run the CLI."""

import sys

from .cli import main

sys.exit(main())
