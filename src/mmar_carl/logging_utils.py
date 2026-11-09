"""
Logging utilities for CARL reasoning system.

Provides structured logging with configurable levels and formats.
"""

import logging
import re
import sys
from typing import Optional

# Default logger name for CARL
LOGGER_NAME = "mmar_carl"

# Create module-level logger
_logger: Optional[logging.Logger] = None

_MAX_ERROR_CHARS = 500


def _safe_error_summary(value: object) -> str:
    """Return a compact, journal-safe error without leaking bearer credentials."""
    text = " ".join(str(value or "").split())
    text = re.sub(r"(?i)(authorization\s*[:=]\s*bearer\s+)\S+", r"\1[REDACTED]", text)
    text = re.sub(
        r"\b[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\b",
        "[REDACTED_TOKEN]",
        text,
    )
    if len(text) > _MAX_ERROR_CHARS:
        text = text[: _MAX_ERROR_CHARS - 1].rstrip() + "…"
    return text


def get_logger(level: Optional[int] = None) -> logging.Logger:
    """
    Get or create the CARL logger.

    Args:
        level: Optional logging level (e.g., logging.DEBUG, logging.INFO).
               If not provided, uses existing level or INFO default.

    Returns:
        Configured logger instance

    Example:
        ```python
        from mmar_carl.logging_utils import get_logger, set_log_level
        import logging

        # Set debug level for development
        set_log_level(logging.DEBUG)

        # Use logger
        logger = get_logger()
        logger.info("Starting chain execution")
        logger.debug("Step dependencies: %s", dependencies)
        ```
    """
    global _logger

    if _logger is None:
        _logger = logging.getLogger(LOGGER_NAME)

        # Only configure if not already configured
        if not _logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            _logger.addHandler(handler)
            _logger.setLevel(logging.INFO)

    if level is not None:
        _logger.setLevel(level)

    return _logger


def set_log_level(level: int) -> None:
    """
    Set the logging level for CARL.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING)

    Example:
        ```python
        import logging
        from mmar_carl.logging_utils import set_log_level

        # Enable debug logging
        set_log_level(logging.DEBUG)

        # Disable most logging
        set_log_level(logging.WARNING)
        ```
    """
    logger = get_logger()
    logger.setLevel(level)


def log_chain_start(chain_name: str, total_steps: int, max_workers: int | str) -> None:
    """Log chain execution start."""
    logger = get_logger()
    logger.info("Starting chain '%s' with %d steps (max_workers=%s)", chain_name, total_steps, max_workers)


def log_chain_complete(success: bool, total_time: float, steps_completed: int, steps_total: int) -> None:
    """Log chain execution completion."""
    logger = get_logger()
    status = "completed successfully" if success else "failed"
    logger.info(
        "Chain execution %s in %.2fs (%d/%d steps)",
        status,
        total_time,
        steps_completed,
        steps_total,
    )


def log_step_start(step_number: int, step_title: str, step_type: str) -> None:
    """Log step execution start."""
    logger = get_logger()
    logger.debug("Starting step %d: %s (type=%s)", step_number, step_title, step_type)


def log_step_complete(
    step_number: int,
    success: bool,
    execution_time: float,
    *,
    step_title: str | None = None,
    step_type: object | None = None,
    error_message: str | None = None,
) -> None:
    """Log step execution completion."""
    logger = get_logger()
    if success:
        logger.debug("Step %d completed in %.2fs", step_number, execution_time)
        return

    identity = f"Step {step_number}"
    if step_title:
        identity += f" {step_title!r}"
    if step_type is not None:
        type_name = getattr(step_type, "value", step_type)
        identity += f" ({type_name})"
    reason = _safe_error_summary(error_message)
    suffix = f" — {reason}" if reason else " — no error detail was provided by the step"
    logger.warning("%s failed in %.2fs%s", identity, execution_time, suffix)


def log_batch_start(batch_num: int, steps_in_batch: int) -> None:
    """Log batch execution start."""
    logger = get_logger()
    logger.debug("Executing batch %d with %d steps in parallel", batch_num, steps_in_batch)


def log_error(message: str, exc: Optional[Exception] = None) -> None:
    """Log an error with optional exception."""
    logger = get_logger()
    if exc:
        logger.error("%s: %s", message, str(exc), exc_info=True)
    else:
        logger.error("%s", message)


def log_warning(message: str, *args) -> None:
    """Log a warning message."""
    logger = get_logger()
    logger.warning(message, *args)


def log_debug(message: str, *args) -> None:
    """Log a debug message."""
    logger = get_logger()
    logger.debug(message, *args)


def log_info(message: str, *args) -> None:
    """Log an info message."""
    logger = get_logger()
    logger.info(message, *args)
