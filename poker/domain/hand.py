"""Hand types and hand ranking."""

from dataclasses import dataclass
from enum import IntEnum

# Mapping rank integers to names for __str__ display
_RANK_NAMES = {
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine",
    10: "Ten",
    11: "Jack",
    12: "Queen",
    13: "King",
    14: "Ace",
}

_RANK_ABBREVS = {
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "T",
    11: "J",
    12: "Q",
    13: "K",
    14: "A",
}


class HandType(IntEnum):
    """Enumeration of poker hand types, ordered from worst to best."""

    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9


@dataclass(frozen=True, order=True)
class HandRank:
    """Comparable hand strength.

    Stores the hand type and ordered kickers. Comparisons are lexicographic:
    first by hand type, then by kickers in descending order.
    """

    type: HandType
    kickers: tuple[int, ...]

    def __str__(self) -> str:
        """Return a human-readable hand description."""
        if self.type == HandType.HIGH_CARD:
            kicker_strs = [_RANK_NAMES[k] for k in self.kickers]
            return f"High card, {', '.join(kicker_strs[:3])}"

        if self.type == HandType.PAIR:
            pair_rank = self.kickers[0]
            return f"Pair of {_RANK_NAMES[pair_rank]}s, {_RANK_NAMES[self.kickers[1]]} kicker"

        if self.type == HandType.TWO_PAIR:
            high = self.kickers[0]
            low = self.kickers[1]
            kicker = self.kickers[2]
            return (
                f"Two pair, {_RANK_NAMES[high]}s and {_RANK_NAMES[low]}s, "
                f"{_RANK_NAMES[kicker]} kicker"
            )

        if self.type == HandType.THREE_OF_A_KIND:
            trips = self.kickers[0]
            return f"Three of a kind, {_RANK_NAMES[trips]}s"

        if self.type == HandType.STRAIGHT:
            high = self.kickers[0]
            return f"Straight, {_RANK_NAMES[high]}-high"

        if self.type == HandType.FLUSH:
            high = self.kickers[0]
            return f"Flush, {_RANK_NAMES[high]}-high"

        if self.type == HandType.FULL_HOUSE:
            trips = self.kickers[0]
            pair = self.kickers[1]
            return f"Full house, {_RANK_NAMES[trips]}s full of {_RANK_NAMES[pair]}s"

        if self.type == HandType.FOUR_OF_A_KIND:
            quads = self.kickers[0]
            kicker = self.kickers[1]
            return f"Four of a kind, {_RANK_NAMES[quads]}s, {_RANK_NAMES[kicker]} kicker"

        # STRAIGHT_FLUSH
        high = self.kickers[0]
        return f"Straight flush, {_RANK_NAMES[high]}-high"
