"""Card representation and enums."""

from enum import IntEnum
from typing import Self

from poker.exceptions import EvaluationError

# ANSI color codes for terminal output
_ANSI_RED = "\033[91m"
_ANSI_BLACK = "\033[30m"
_ANSI_RESET = "\033[0m"


def _colorize_card(rank_str: str, suit: "Suit") -> str:
    """Return a colored card string with ANSI codes.

    Args:
        rank_str: The rank string (e.g., 'A', '2', 'K').
        suit: The Suit enum value.

    Returns:
        A string with rank, colored suit symbol, and ANSI reset code.
    """
    suit_symbol = str(suit)
    # Red suits: HEARTS (2) and DIAMONDS (1)
    if suit in (Suit.HEARTS, Suit.DIAMONDS):
        return f"{rank_str}{_ANSI_RED}{suit_symbol}{_ANSI_RESET}"
    # Black suits: CLUBS (0) and SPADES (3)
    else:
        return f"{rank_str}{_ANSI_BLACK}{suit_symbol}{_ANSI_RESET}"


class Suit(IntEnum):
    """Card suit."""

    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3

    def __str__(self) -> str:
        """Return the suit symbol as Unicode character."""
        return ["♣", "♦", "♥", "♠"][self.value]


class Rank(IntEnum):
    """Card rank."""

    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    def __str__(self) -> str:
        """Return the rank symbol."""
        return ["", "", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"][
            self.value
        ]


class Card:
    """An immutable playing card."""

    __slots__ = ("_rank", "_suit")

    def __init__(self, rank: Rank, suit: Suit) -> None:
        """Initialize a card.

        Args:
            rank: The rank of the card.
            suit: The suit of the card.
        """
        self._rank = rank
        self._suit = suit

    @property
    def rank(self) -> Rank:
        """Return the rank."""
        return self._rank

    @property
    def suit(self) -> Suit:
        """Return the suit."""
        return self._suit

    def __str__(self) -> str:
        """Return colored string representation like 'A♠' or '2♦' with ANSI color codes."""
        return _colorize_card(str(self.rank), self.suit)

    def __repr__(self) -> str:
        """Return unambiguous representation."""
        return f"Card({self.rank.name}, {self.suit.name})"

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self) -> int:
        """Make cards hashable so they can be in sets/dicts."""
        return hash((self.rank, self.suit))

    def __lt__(self, other: object) -> bool:
        """Compare by rank first, then suit."""
        if not isinstance(other, Card):
            return NotImplemented
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.suit < other.suit

    def __le__(self, other: object) -> bool:
        """Compare by rank first, then suit."""
        if not isinstance(other, Card):
            return NotImplemented
        return self == other or self < other

    def __gt__(self, other: object) -> bool:
        """Compare by rank first, then suit."""
        if not isinstance(other, Card):
            return NotImplemented
        return not self <= other

    def __ge__(self, other: object) -> bool:
        """Compare by rank first, then suit."""
        if not isinstance(other, Card):
            return NotImplemented
        return not self < other

    @classmethod
    def from_string(cls, s: str) -> Self:
        """Parse a card from a string like 'As' or '2d'.

        Args:
            s: The card string.

        Returns:
            The parsed card.

        Raises:
            EvaluationError: If the string is not a valid card.
        """
        if len(s) != 2:
            raise EvaluationError(f"Invalid card string: {s!r}")

        rank_str, suit_str = s[0], s[1]

        # Parse rank
        rank_map = {
            "2": Rank.TWO,
            "3": Rank.THREE,
            "4": Rank.FOUR,
            "5": Rank.FIVE,
            "6": Rank.SIX,
            "7": Rank.SEVEN,
            "8": Rank.EIGHT,
            "9": Rank.NINE,
            "T": Rank.TEN,
            "J": Rank.JACK,
            "Q": Rank.QUEEN,
            "K": Rank.KING,
            "A": Rank.ACE,
        }
        if rank_str not in rank_map:
            raise EvaluationError(f"Invalid rank: {rank_str!r}")
        rank = rank_map[rank_str]

        # Parse suit (case-insensitive)
        suit_map = {
            "c": Suit.CLUBS,
            "d": Suit.DIAMONDS,
            "h": Suit.HEARTS,
            "s": Suit.SPADES,
            "♣": Suit.CLUBS,
            "♦": Suit.DIAMONDS,
            "♥": Suit.HEARTS,
            "♠": Suit.SPADES,
        }
        suit_lower = suit_str.lower()
        if suit_lower not in suit_map:
            raise EvaluationError(f"Invalid suit: {suit_str!r}")
        suit = suit_map[suit_lower]

        return cls(rank, suit)
