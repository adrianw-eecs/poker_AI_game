"""Player actions in poker."""

from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    """Enumeration of legal action types."""

    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"
    ALL_IN = "all_in"


@dataclass(frozen=True)
class Action:
    """An action taken by a player.

    Attributes:
        type: The type of action (fold, check, call, raise, all-in).
        amount: The total chips committed by this action. Zero for fold/check.
    """

    type: ActionType
    amount: int = 0

    def __post_init__(self) -> None:
        """Validate action consistency."""
        if self.amount < 0:
            raise ValueError(f"Action amount cannot be negative: {self.amount}")

        if self.type in (ActionType.RAISE, ActionType.ALL_IN) and self.amount == 0:
            raise ValueError(
                f"{self.type.value} action requires a positive amount, got {self.amount}"
            )

        if self.type in (ActionType.FOLD, ActionType.CHECK) and self.amount != 0:
            raise ValueError(
                f"{self.type.value} action must have zero amount, got {self.amount}"
            )

    @classmethod
    def fold(cls) -> "Action":
        """Create a fold action."""
        return cls(type=ActionType.FOLD, amount=0)

    @classmethod
    def check(cls) -> "Action":
        """Create a check action."""
        return cls(type=ActionType.CHECK, amount=0)

    @classmethod
    def call(cls, amount: int) -> "Action":
        """Create a call action.

        Args:
            amount: The amount to call (total chips committed).

        Returns:
            A call action.
        """
        return cls(type=ActionType.CALL, amount=amount)

    @classmethod
    def raise_to(cls, amount: int) -> "Action":
        """Create a raise action.

        Args:
            amount: The total chips to raise to.

        Returns:
            A raise action.
        """
        return cls(type=ActionType.RAISE, amount=amount)

    @classmethod
    def all_in(cls, amount: int) -> "Action":
        """Create an all-in action.

        Args:
            amount: The total chips going all-in.

        Returns:
            An all-in action.
        """
        return cls(type=ActionType.ALL_IN, amount=amount)

    def __str__(self) -> str:
        """Return a human-readable action description."""
        if self.type == ActionType.FOLD:
            return "folds"
        elif self.type == ActionType.CHECK:
            return "checks"
        elif self.type == ActionType.CALL:
            return f"calls {self.amount}"
        elif self.type == ActionType.RAISE:
            return f"raises to {self.amount}"
        # ALL_IN
        return f"goes all-in for {self.amount}"
