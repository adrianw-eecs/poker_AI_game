"""Game logging interface and implementation."""

import json
from pathlib import Path
from typing import Protocol

from poker.logging.events import Event, EventEncoder


class Logger(Protocol):
    """Protocol for logging game events.

    A logger receives events during hand play and can process them
    (e.g., write to file, update stats, send to monitoring service).
    Loggers are injected into the game engine so they can be swapped
    out (e.g., NullLogger for training runs that don't need logs).
    """

    def log_event(self, event: Event) -> None:
        """Log a single game event.

        Args:
            event: The event to log.
        """
        ...

    def flush(self) -> None:
        """Flush any buffered writes to ensure data is persisted.

        Called at the end of a hand (HandEnded event) to ensure
        all events for that hand are written to storage.
        """
        ...


class NullLogger:
    """Logger that discards all events (useful for training)."""

    def log_event(self, event: Event) -> None:
        """Discard the event."""
        pass

    def flush(self) -> None:
        """No-op flush."""
        pass


class GameLogger:
    """Writes game events to a JSONL file with buffering.

    Each event is stored as a JSON object on a single line, making
    the log easy to parse, replay, and analyze. Buffering improves
    performance by batching writes; flushing happens on HandEnded
    events to ensure hand data is persisted atomically.
    """

    def __init__(self, filepath: Path | str) -> None:
        """Initialize the logger with a JSONL file path.

        Args:
            filepath: Path where the JSONL log will be written.
                      Created if it doesn't exist; appended to if it does.
        """
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: list[str] = []

    def log_event(self, event: Event) -> None:
        """Buffer an event for writing.

        Args:
            event: The event to log.
        """
        data = EventEncoder.to_dict(event)
        json_line = json.dumps(data)
        self._buffer.append(json_line)

    def flush(self) -> None:
        """Write all buffered events to the JSONL file."""
        if not self._buffer:
            return

        with open(self.filepath, "a") as f:
            for line in self._buffer:
                f.write(line + "\n")

        self._buffer.clear()
