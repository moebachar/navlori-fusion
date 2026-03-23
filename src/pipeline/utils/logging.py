"""Structured logging setup for NavLoRI Fusion.

Three levels of visibility:
  - High: MLflow (experiment comparison, post-hoc)
  - Mid: TensorBoard (live training curves, attention maps)
  - Low: Python logging via Rich (debug prints, tensor shapes, timing)
"""

import logging

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure structured logging with Rich handler."""
    logger = logging.getLogger("navlori")
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the navlori namespace."""
    return logging.getLogger(f"navlori.{name}")
