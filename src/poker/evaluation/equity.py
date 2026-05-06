"""Monte Carlo equity calculator for poker hands."""

from collections.abc import Sequence

from poker.domain.card import Card
from poker.domain.deck import Deck
from poker.evaluation.evaluator import evaluate
from poker.rng import RNG


def monte_carlo_equity(
    hole: Sequence[Card],
    board: Sequence[Card],
    num_opponents: int,
    num_trials: int,
    rng: RNG,
) -> float:
    """Estimate win probability via Monte Carlo simulation.

    Args:
        hole: Player's 2 hole cards.
        board: Community cards (0-5 cards).
        num_opponents: Number of opponents with unknown cards.
        num_trials: Number of simulations to run.
        rng: Seeded RNG for reproducibility.

    Returns:
        Win probability in [0, 1]. Chops are counted as 0.5 (shared win).
    """
    if len(hole) != 2:
        raise ValueError(f"Expected 2 hole cards, got {len(hole)}")
    if len(board) not in (0, 3, 4, 5):
        raise ValueError(f"Board must have 0, 3, 4, or 5 cards, got {len(board)}")
    if num_opponents < 1:
        raise ValueError(f"Must have at least 1 opponent, got {num_opponents}")
    if num_trials < 1:
        raise ValueError(f"Must run at least 1 trial, got {num_trials}")

    known_cards = set(hole) | set(board)
    wins = 0.0

    for _ in range(num_trials):
        # Create a deck excluding known cards
        full_deck = Deck(rng)
        remaining = [card for card in full_deck._create_deck() if card not in known_cards]

        # Shuffle remaining cards
        rng.shuffle(remaining)

        # Deal remaining board cards to complete to 5
        cards_needed = 5 - len(board)
        final_board = list(board) + remaining[:cards_needed]
        idx = cards_needed

        # Deal opponent hole cards
        opponent_hole_cards = []
        for _ in range(num_opponents):
            opponent_hole_cards.append([remaining[idx], remaining[idx + 1]])
            idx += 2

        # Evaluate all hands
        our_hand = evaluate(list(hole) + final_board)
        opponent_hands = [
            evaluate(opp_hole + final_board)
            for opp_hole in opponent_hole_cards
        ]

        # Count wins (chops)
        better_opponents = sum(1 for opp_hand in opponent_hands if opp_hand > our_hand)
        equal_opponents = sum(1 for opp_hand in opponent_hands if opp_hand == our_hand)

        if better_opponents == 0:
            # We win or chop
            if equal_opponents == 0:
                wins += 1.0
            else:
                # We chop with equal_opponents + ourselves
                wins += 1.0 / (equal_opponents + 1)

    return wins / num_trials


def exact_equity_river(
    hole: Sequence[Card], board: Sequence[Card], opponent_hole: Sequence[Card]
) -> float:
    """Calculate exact heads-up equity on river (all 5 community cards known).

    Args:
        hole: Player's 2 hole cards.
        board: All 5 community cards.
        opponent_hole: Opponent's 2 hole cards.

    Returns:
        0.0 if opponent wins, 0.5 if tie, 1.0 if player wins.
    """
    if len(hole) != 2 or len(board) != 5 or len(opponent_hole) != 2:
        raise ValueError("exact_equity_river requires 2 hole cards, 5 board cards")

    our_hand = evaluate(list(hole) + list(board))
    opp_hand = evaluate(list(opponent_hole) + list(board))

    if our_hand > opp_hand:
        return 1.0
    elif our_hand == opp_hand:
        return 0.5
    else:
        return 0.0
