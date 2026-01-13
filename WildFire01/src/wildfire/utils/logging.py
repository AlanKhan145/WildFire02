"""Logging helpers."""

from __future__ import annotations

import logging
from typing import Optional


def setup_logger(name: str = "wildfire", level: int = logging.INFO) -> logging.Logger:
    """Create or fetch a logger with a stream handler."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get logger with default name."""
    return setup_logger(name or "wildfire")
