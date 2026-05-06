"""Blind level schedules for tournament-style progression."""

from dataclasses import dataclass

from poker.exceptions import ConfigError


@dataclass(frozen=True)
class BlindLevel:
    """A single blind level in a schedule.

    Attributes:
        small: The small blind amount in chips.
        big: The big blind amount in chips.
        ante: The ante amount per player in chips.
    """

    small: int
    big: int
    ante: int = 0


@dataclass
class BlindSchedule:
    """A schedule of escalating or fixed blind levels.

    Attributes:
        levels: List of BlindLevel entries.
        hands_per_level: Number of hands before advancing to the next level.
        fixed: If True, always use levels[0] (no escalation).
    """

    levels: list[BlindLevel]
    hands_per_level: int
    fixed: bool = False

    def __post_init__(self) -> None:
        """Validate the blind schedule."""
        if not self.levels:
            raise ConfigError("Blind schedule must have at least one level")

        if self.hands_per_level <= 0:
            raise ConfigError("hands_per_level must be positive")

        # In escalating mode, verify big blinds are non-decreasing
        if not self.fixed:
            for i in range(len(self.levels) - 1):
                if self.levels[i].big > self.levels[i + 1].big:
                    raise ConfigError(
                        f"Big blinds must be non-decreasing in escalating mode. "
                        f"Level {i} has {self.levels[i].big}, "
                        f"but level {i + 1} has {self.levels[i + 1].big}."
                    )

    def level_for_hand(self, hand_number: int) -> BlindLevel:
        """Get the blind level for a given hand number.

        Args:
            hand_number: The hand number (0-indexed).

        Returns:
            The appropriate BlindLevel for this hand.
        """
        if self.fixed:
            return self.levels[0]

        level_index = hand_number // self.hands_per_level
        # Clamp to the last level if we've exhausted the schedule
        level_index = min(level_index, len(self.levels) - 1)
        return self.levels[level_index]
