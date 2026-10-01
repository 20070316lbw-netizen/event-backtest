from __future__ import annotations

from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent
logger.info(f"项目根目录是 {PROJECT_ROOT}")

