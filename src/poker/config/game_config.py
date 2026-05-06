"""Game configuration."""

from dataclasses import dataclass
from enum import Enum
from typing import cast

from poker.config.blind_schedule import BlindSchedule
from poker.exceptions import ConfigError


class TournamentMode(Enum):
    """Tournament game mode.

    Attributes:
        SNG: Sit-n-Go tournament.
        MTT: Multi-Table Tournament.
    """

    SNG = "sng"
    MTT = "mtt"


@dataclass(frozen=True)
class GameConfig:
    """Immutable game configuration.

    Attributes:
        num_players: Number of players (2-10).
        starting_stack: Starting chip stack per player.
        small_blind: Small blind amount in chips.
        big_blind: Big blind amount in chips.
        ante: Ante amount per player in chips.
        rake_percent: Rake as percentage of pot (0-100).
        rake_cap: Maximum rake in chips per pot, or None for uncapped.
        blind_schedule: The blind schedule (escalating or fixed).
        run_it_twice: Whether to enable run-it-twice for all-in situations.
        tournament_mode: Tournament mode (SNG, MTT) or None for cash game.
        final_table_size: Number of players for final table notification in MTT (default 9).
    """

    num_players: int
    starting_stack: int
    small_blind: int
    big_blind: int
    ante: int
    rake_percent: float
    rake_cap: int | None
    blind_schedule: BlindSchedule
    run_it_twice: bool = False
    tournament_mode: TournamentMode | None = None
    final_table_size: int = 9

    def __post_init__(self) -> None:
        """Validate game configuration."""
        if not 2 <= self.num_players <= 10:
            raise ConfigError(f"num_players must be 2-10, got {self.num_players}")

        if self.starting_stack < 2 * self.big_blind:
            raise ConfigError(
                f"starting_stack ({self.starting_stack}) must be at least "
                f"2 x big_blind ({2 * self.big_blind})"
            )

        if not 0 <= self.rake_percent <= 100:
            raise ConfigError(
                f"rake_percent must be 0-100, got {self.rake_percent}"
            )

        if self.rake_cap is not None and self.rake_cap < 0:
            raise ConfigError(f"rake_cap must be non-negative, got {self.rake_cap}")

        if self.ante < 0:
            raise ConfigError(f"ante must be non-negative, got {self.ante}")

        if self.small_blind < 0:
            raise ConfigError(f"small_blind must be non-negative, got {self.small_blind}")

        if self.big_blind <= 0:
            raise ConfigError(f"big_blind must be positive, got {self.big_blind}")

    def to_dict(self) -> dict[str, object]:
        """Convert config to a dictionary (JSON-safe).

        Returns:
            A dictionary representation of the config.
        """
        return {
            "num_players": self.num_players,
            "starting_stack": self.starting_stack,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "ante": self.ante,
            "rake_percent": self.rake_percent,
            "rake_cap": self.rake_cap,
            "run_it_twice": self.run_it_twice,
            # blind_schedule is not serialized here; caller handles separately
        }

    @classmethod
    def from_dict(cls, data: dict[str, object], blind_schedule: BlindSchedule) -> "GameConfig":
        """Create config from a dictionary.

        Args:
            data: The dictionary representation.
            blind_schedule: The blind schedule instance.

        Returns:
            A GameConfig instance.
        """
        return cls(
            num_players=cast(int, data["num_players"]),
            starting_stack=cast(int, data["starting_stack"]),
            small_blind=cast(int, data["small_blind"]),
            big_blind=cast(int, data["big_blind"]),
            ante=cast(int, data.get("ante", 0)),
            rake_percent=cast(float, data.get("rake_percent", 0.0)),
            rake_cap=cast(int | None, data.get("rake_cap")),
            blind_schedule=blind_schedule,
            run_it_twice=cast(bool, data.get("run_it_twice", False)),
        )
