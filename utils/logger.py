"""
utils/logger.py - Structured logging with structlog.
Writes JSON to stdout AND to logs/app.log for persistent file logging.
"""
import logging
import sys
from pathlib import Path

import structlog


def setup_logging() -> None:
    """Configure structlog for structured JSON logging to both stdout and file."""
    from config.settings import config
    log_dir = Path(config["app"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, config["app"]["log_level"].upper(), logging.INFO)

    # Root logger with console + file handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root_logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)

    log_file = log_dir / "app.log"
    if not any(isinstance(h, logging.FileHandler) and str(log_file) in str(getattr(h, 'baseFilename', ''))
               for h in root_logger.handlers):
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    setup_logging()
    return structlog.get_logger(name)
