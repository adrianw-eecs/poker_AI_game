"""Human-readable session transcript logger."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from poker.logging.events import Event, HandEnded, SessionEnded
from poker.logging.transcript import render_event


class SessionTextLogger:
    """Writes a single human-readable transcript for an entire session.

    Lines are rendered from the same events as the JSONL log. The buffer
    is flushed when a hand or the session ends.
    """

    def __init__(self, filepath: Path | str, *, header: str | None = None, seed: int | None = None) -> None:
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: list[str] = []

        banner = header or "=== POKER SESSION TRANSCRIPT ==="
        created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._buffer.append(f"{banner}\nCreated: {created}\n")
        if seed is not None:
            self._buffer.append(f"Random Seed: {seed}\n")
        self._buffer.append("\n")
        self.flush()

    def log_event(self, event: Event) -> None:
        text = render_event(event)
        if text:
            if not text.endswith("\n"):
                text += "\n"
            self._buffer.append(text)
        if isinstance(event, (HandEnded, SessionEnded)):
            self.flush()

    def log_text(self, text: str) -> None:
        """Log raw text directly to the transcript file.

        Args:
            text: The text to log.
        """
        self._buffer.append(text)
        if not text.endswith("\n"):
            self._buffer.append("\n")
        self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.writelines(self._buffer)
        self._buffer.clear()
