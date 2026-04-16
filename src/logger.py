"""
Logger Module.

Centralized logging configuration for the Stock Price Predictor application.
Provides a consistent logging setup with console and file handlers.

Usage:
    from src.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Starting process...")
    logger.error("An error occurred")
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


# Log format with color support for console
CONSOLE_FORMAT: str = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)

# File format (more detailed)
FILE_FORMAT: str = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(funcName)s | %(message)s"
)

# ANSI color codes for console output
COLORS: dict = {
    "DEBUG": "\033[36m",     # Cyan
    "INFO": "\033[32m",      # Green
    "WARNING": "\033[33m",   # Yellow
    "ERROR": "\033[31m",     # Red
    "CRITICAL": "\033[35m",  # Magenta
    "RESET": "\033[0m"       # Reset
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels for console output."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with color if level is known."""
        color = COLORS.get(record.levelname, COLORS["RESET"])
        reset = COLORS["RESET"]

        # Add color to levelname
        record.levelname = f"{color}{record.levelname}{reset}"

        return super().format(record)


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_dir: Optional[str] = None
) -> logging.Logger:
    """
    Get or create a logger with the specified name.

    Configures the logger with:
    - Console handler with colored output
    - Optional file handler for persistent logs

    Args:
        name: Logger name (typically use __name__).
        level: Logging level (default: INFO).
        log_to_file: Whether to log to file (default: True).
        log_dir: Directory for log files (default: ./logs).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Console handler with colored formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = ColoredFormatter(CONSOLE_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_to_file:
        if log_dir is None:
            log_dir = str(Path(__file__).parent.parent / "logs")

        # Create logs directory if it doesn't exist
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # Create log file with timestamp in name
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = Path(log_dir) / f"stock_predictor_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(FILE_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    return logger


# Create a default logger for module-level use
default_logger: logging.Logger = get_logger("stock_predictor")


def get_default_logger() -> logging.Logger:
    """
    Get the default logger instance.

    Returns:
        Default logger configured for the application.
    """
    return default_logger
