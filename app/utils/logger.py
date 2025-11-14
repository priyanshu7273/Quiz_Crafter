"""Structured logging helpers used across the project."""
from __future__ import annotations

import json
import logging
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import get_settings

_LOGGER_CACHE: Dict[str, Logger] = {}


class JsonFormatter(logging.Formatter):
    """Format log records as JSON preserving original message for humans."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - standard override
        payload: Dict[str, Any] = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        for key, value in record.__dict__.items():
            if key.startswith("_extra_"):
                payload[key.replace("_extra_", "")] = value
        return json.dumps(payload, ensure_ascii=False)


def configure_logger(name: str) -> Logger:
    """Create or reuse a logger with console + JSON file handlers."""

    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]

    try:
        settings = get_settings()
        level = settings.log_level
        log_path = settings.log_path
    except Exception:
        level = "INFO"
        log_path = "logs/app.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (JSON)
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    _LOGGER_CACHE[name] = logger
    return logger


def log_extra(**kwargs: Any) -> Dict[str, Any]:
    """Prepare a dictionary that will be serialised into the JSON log file."""

    return {f"_extra_{key}": value for key, value in kwargs.items()}
