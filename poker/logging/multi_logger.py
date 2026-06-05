"""Utilities for composing multiple loggers."""

from poker.logging.events import Event
from poker.logging.logger import Logger


class MultiLogger:
    """Fan-out logger that forwards events to multiple loggers."""

    def __init__(self, loggers: list[Logger]) -> None:
        self._loggers = loggers

    def log_event(self, event: Event) -> None:
        for logger in self._loggers:
            logger.log_event(event)

    def flush(self) -> None:
        for logger in self._loggers:
            logger.flush()

