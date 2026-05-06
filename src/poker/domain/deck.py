"""Deck of cards with deterministic shuffling."""

from poker.domain.card import Card, Rank, Suit
from poker.exceptions import EngineStateError
from poker.rng import RNG


class Deck:
    """A standard 52-card deck with seeded shuffling."""

    def __init__(self, rng: RNG) -> None:
        """Initialize a new deck.

        Args:
            rng: The RNG instance for shuffling (injected for reproducibility).
        """
        self._rng = rng
        self._cards = self._create_deck()
        self._shuffle()

    @staticmethod
    def _create_deck() -> list[Card]:
        """Create a new unshuffled deck."""
        return [Card(rank, suit) for rank in Rank for suit in Suit]

    def _shuffle(self) -> None:
        """Shuffle the deck."""
        self._rng.shuffle(self._cards)

    def deal(self, n: int) -> list[Card]:
        """Deal n cards from the deck.

        Args:
            n: Number of cards to deal.

        Returns:
            A list of n cards.

        Raises:
            EngineStateError: If not enough cards remain.
        """
        if n > len(self._cards):
            raise EngineStateError(
                f"Cannot deal {n} cards; only {len(self._cards)} remain"
            )
        cards = self._cards[:n]
        self._cards = self._cards[n:]
        return cards

    def remaining(self) -> int:
        """Return the number of cards remaining in the deck.

        Returns:
            Number of cards left.
        """
        return len(self._cards)

    def reset(self) -> None:
        """Reset the deck to a full, shuffled state."""
        self._cards = self._create_deck()
        self._shuffle()

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"Deck({self.remaining()} cards)"
