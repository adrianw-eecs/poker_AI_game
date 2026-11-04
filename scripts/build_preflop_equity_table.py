"""Build the 169-class preflop equity table used by observation v2.

Each row is one starting hand. Columns are win-or-chop equity against 1 through
8 random opponents, estimated with ``monte_carlo_equity``.
"""

import argparse
from pathlib import Path

import numpy as np

from poker.domain.card import Card, Rank, Suit
from poker.evaluation.equity import monte_carlo_equity
from poker.ml.observation_v2 import preflop_combo_index
from poker.rng import RNG

_RANKS = list(Rank)
_SUITS = list(Suit)
_DEFAULT_OUT = (
    Path(__file__).resolve().parents[1] / "poker" / "evaluation" / "data" / "preflop_equity.npy"
)


def _canonical_cards() -> list[tuple[Card, Card]]:
    """One representative holding for each of the 169 classes, in index order."""
    hands: list[tuple[Card, Card] | None] = [None] * 169
    for high in range(len(_RANKS)):
        for low in range(high + 1):
            if high == low:
                cards = (Card(_RANKS[high], Suit.SPADES), Card(_RANKS[low], Suit.HEARTS))
            else:
                suited = (Card(_RANKS[high], Suit.SPADES), Card(_RANKS[low], Suit.SPADES))
                offsuit = (Card(_RANKS[high], Suit.SPADES), Card(_RANKS[low], Suit.HEARTS))
                hands[preflop_combo_index(*suited)] = suited
                hands[preflop_combo_index(*offsuit)] = offsuit
                continue
            hands[preflop_combo_index(*cards)] = cards
    if any(hand is None for hand in hands):
        raise RuntimeError("Preflop class map did not cover all 169 hands")
    return [hand for hand in hands if hand is not None]


def build_table(num_trials: int, seed: int) -> np.ndarray:
    """Return a float32 array of shape (169, 8)."""
    table = np.zeros((169, 8), dtype=np.float32)
    hands = _canonical_cards()
    for index, hole in enumerate(hands):
        for opponents in range(1, 9):
            rng = RNG(seed + index * 16 + opponents)
            table[index, opponents - 1] = monte_carlo_equity(
                hole, (), opponents, num_trials, rng
            )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Write poker/evaluation/data/preflop_equity.npy")
    parser.add_argument("--trials", type=int, default=200, help="Monte Carlo trials per cell")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args()
    table = build_table(args.trials, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, table)
    print(f"wrote {table.shape} table to {args.out}")


if __name__ == "__main__":
    main()
