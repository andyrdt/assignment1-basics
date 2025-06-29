from datetime import datetime
from functools import partial

# ---------------------------------------------------------------------------
# Simple logging helper configuration
# Choose the minimum visible level by editing `LOG_LEVEL` below. Any calls with
# a level *below* this threshold will be ignored.
#
# Available options (case-insensitive):
#   "DEBUG"  – show everything
#   "INFO"   – default, hide DEBUG messages
#   "WARN"   – show only warnings & errors
#   "ERROR"  – show only errors
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"

_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}

def _timestamp() -> str:
    """Return current local time formatted with millisecond precision."""
    # %f gives microseconds; keep the first three digits for milliseconds.
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def log_print(*msg, level: str = "INFO", sep: str = " ", **print_kwargs):
    """
    A lightweight replacement for logging.* calls.

    Example
    -------
    >>> log_print("Loading file", level="DEBUG")
    2025-06-29 14:37:51.123 | DEBUG | Loading file
    """
    level = level.upper()
    # Respect the global threshold; skip message if below the selected level
    if _LEVELS.get(level, 100) < _LEVELS.get(LOG_LEVEL.upper(), 20):
        return

    prefix = f"{_timestamp()} | {level:<5} |"
    print(prefix, *msg, sep=sep, **print_kwargs)