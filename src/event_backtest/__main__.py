"""python -m event_backtest 的入口。"""
from __future__ import annotations

import sys

from event_backtest.cli import main

if __name__ == "__main__":
    sys.exit(main())
