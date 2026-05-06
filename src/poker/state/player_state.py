"""Player state snapshot."""

from dataclasses import dataclass

from poker.domain.card import Card


@dataclass(frozen=True)
class PlayerState:
    """Immutable snapshot of a player's per-hand state.

    Attributes:
        seat: The player's seat number.
        name: The player's name.
        stack: The player's current chip stack.
        hole_cards: The player's hole cards (empty if not dealt).
        committed_this_street: Chips committed in the current street only.
        committed_this_hand: Total chips committed across all streets this hand.
        has_folded: Whether the player has folded.
        is_all_in: Whether the player is all-in.
        is_eliminated: Whether the player has been eliminated.
    """

    seat: int
    name: str
    stack: int
    hole_cards: tuple[Card, ...]
    committed_this_street: int
    committed_this_hand: int
    has_folded: bool
    is_all_in: bool
    is_eliminated: bool

    @property
    def is_active(self) -> bool:
        """Check if the player is active (not folded, not eliminated, has cards).

        Returns:
            True if the player is active in the current hand.
        """
        return not self.has_folded and not self.is_eliminated and len(self.hole_cards) > 0

    def effective_stack(self, other: "PlayerState") -> int:
        """Get the effective stack size relative to another player.

        The effective stack is the minimum of the two players' stacks,
        since no one can win more than that from their opponent.

        Args:
            other: The other player.

        Returns:
            The effective stack size.
        """
        return min(self.stack, other.stack)

    def with_stack(self, stack: int) -> "PlayerState":
        """Return a new PlayerState with the stack updated.

        Args:
            stack: The new stack size.

        Returns:
            A new PlayerState instance.
        """
        return PlayerState(
            seat=self.seat,
            name=self.name,
            stack=stack,
            hole_cards=self.hole_cards,
            committed_this_street=self.committed_this_street,
            committed_this_hand=self.committed_this_hand,
            has_folded=self.has_folded,
            is_all_in=self.is_all_in,
            is_eliminated=self.is_eliminated,
        )

    def with_hole_cards(self, hole_cards: tuple[Card, ...]) -> "PlayerState":
        """Return a new PlayerState with hole cards set.

        Args:
            hole_cards: The new hole cards.

        Returns:
            A new PlayerState instance.
        """
        return PlayerState(
            seat=self.seat,
            name=self.name,
            stack=self.stack,
            hole_cards=hole_cards,
            committed_this_street=self.committed_this_street,
            committed_this_hand=self.committed_this_hand,
            has_folded=self.has_folded,
            is_all_in=self.is_all_in,
            is_eliminated=self.is_eliminated,
        )

    def with_committed_this_street(self, amount: int) -> "PlayerState":
        """Return a new PlayerState with committed_this_street updated.

        Args:
            amount: The new committed this street amount.

        Returns:
            A new PlayerState instance.
        """
        return PlayerState(
            seat=self.seat,
            name=self.name,
            stack=self.stack,
            hole_cards=self.hole_cards,
            committed_this_street=amount,
            committed_this_hand=self.committed_this_hand,
            has_folded=self.has_folded,
            is_all_in=self.is_all_in,
            is_eliminated=self.is_eliminated,
        )

    def with_committed_this_hand(self, amount: int) -> "PlayerState":
        """Return a new PlayerState with committed_this_hand updated.

        Args:
            amount: The new committed this hand amount.

        Returns:
            A new PlayerState instance.
        """
        return PlayerState(
            seat=self.seat,
            name=self.name,
            stack=self.stack,
            hole_cards=self.hole_cards,
            committed_this_street=self.committed_this_street,
            committed_this_hand=amount,
            has_folded=self.has_folded,
            is_all_in=self.is_all_in,
            is_eliminated=self.is_eliminated,
        )

    def with_folded(self, has_folded: bool) -> "PlayerState":
        """Return a new PlayerState with folded status updated.

        Args:
            has_folded: Whether the player has folded.

        Returns:
            A new PlayerState instance.
        """
        return PlayerState(
            seat=self.seat,
            name=self.name,
            stack=self.stack,
            hole_cards=self.hole_cards,
            committed_this_street=self.committed_this_street,
            committed_this_hand=self.committed_this_hand,
            has_folded=has_folded,
            is_all_in=self.is_all_in,
            is_eliminated=self.is_eliminated,
        )

    def with_all_in(self, is_all_in: bool) -> "PlayerState":
        """Return a new PlayerState with all-in status updated.

        Args:
            is_all_in: Whether the player is all-in.

        Returns:
            A new PlayerState instance.
        """
        return PlayerState(
            seat=self.seat,
            name=self.name,
            stack=self.stack,
            hole_cards=self.hole_cards,
            committed_this_street=self.committed_this_street,
            committed_this_hand=self.committed_this_hand,
            has_folded=self.has_folded,
            is_all_in=is_all_in,
            is_eliminated=self.is_eliminated,
        )

    def with_eliminated(self, is_eliminated: bool) -> "PlayerState":
        """Return a new PlayerState with eliminated status updated.

        Args:
            is_eliminated: Whether the player is eliminated.

        Returns:
            A new PlayerState instance.
        """
        return PlayerState(
            seat=self.seat,
            name=self.name,
            stack=self.stack,
            hole_cards=self.hole_cards,
            committed_this_street=self.committed_this_street,
            committed_this_hand=self.committed_this_hand,
            has_folded=self.has_folded,
            is_all_in=self.is_all_in,
            is_eliminated=is_eliminated,
        )
