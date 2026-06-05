"""Human-readable session transcript logger."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from poker.logging.events import ActionTaken, Event, HandEnded, HandStarted, StreetEnded


class SessionTextLogger:
    """Writes a single human-readable transcript for an entire session."""

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
        if isinstance(event, HandStarted):
            self._buffer.append(
                f"{'=' * 60}\n"
                f"Hand #{event.hand_number + 1} started | dealer_seat={event.dealer_seat} "
                f"| blinds={event.small_blind}/{event.big_blind}\n"
                f"{'=' * 60}\n\n"
            )
            self.flush()  # Flush after hand start
            return

        if isinstance(event, ActionTaken):
            if event.log_line:
                self._buffer.append(f"{event.log_line}\n")
            else:
                amt = f" (+{event.chips_added})" if event.chips_added else ""
                self._buffer.append(
                    f"[Hand #{event.hand_number + 1} | {event.street.upper()} | "
                    f"#{event.action_num}] seat {event.seat}: "
                    f"{event.action_type.upper()} {event.amount}{amt} "
                    f"| Pot: {event.pot_after}\n"
                )
            self.flush()  # Flush after each action
            return

        if isinstance(event, StreetEnded):
            if event.street == "REPLAY":
                self._buffer.append(event.snapshot)
                if not event.snapshot.endswith("\n"):
                    self._buffer.append("\n")
            else:
                self._buffer.append(f"[End of {event.street}]\n")
                self._buffer.append(event.snapshot)
                if not event.snapshot.endswith("\n"):
                    self._buffer.append("\n")
            self.flush()  # Flush after each street ends
            return

        if isinstance(event, HandEnded):
            dist = " ".join(
                f"seat{seat}={chips}" for seat, chips in sorted(event.chip_distribution.items())
            )
            self._buffer.append(f"Hand ended | chip_distribution: {dist}\n\n")
            self.flush()  # Flush after hand ends
            return

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
