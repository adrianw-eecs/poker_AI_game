"""Monte Carlo equity calculator for poker hands."""

from collections.abc import Sequence

import numpy as np

from poker.domain.card import Card
from poker.evaluation.evaluator import evaluate
from poker.evaluation.fast_evaluator import card_index, score_hands
from poker.rng import RNG


def monte_carlo_equity(
    hole: Sequence[Card],
    board: Sequence[Card],
    num_opponents: int,
    num_trials: int,
    rng: RNG,
) -> float:
    """Estimate win probability via Monte Carlo simulation.

    The remaining deck is built once. Every trial's cards are drawn together
    as one index array, and hands are scored with the rank-multiset lookup
    rather than ``evaluate``. Chops count as ``1 / (partners + 1)``.

    Args:
        hole: Player's 2 hole cards.
        board: Community cards (0-5 cards).
        num_opponents: Number of opponents with unknown cards.
        num_trials: Number of simulations to run.
        rng: Seeded RNG for reproducibility. ``rng.seed`` selects the numpy
            stream. An unseeded RNG contributes one ``randint`` draw.

    Returns:
        Win probability in [0, 1].
    """
    if len(hole) != 2:
        raise ValueError(f"Expected 2 hole cards, got {len(hole)}")
    if len(board) not in (0, 3, 4, 5):
        raise ValueError(f"Board must have 0, 3, 4, or 5 cards, got {len(board)}")
    if num_opponents < 1:
        raise ValueError(f"Must have at least 1 opponent, got {num_opponents}")
    if num_trials < 1:
        raise ValueError(f"Must run at least 1 trial, got {num_trials}")

    hole_ids = np.asarray([card_index(card) for card in hole], dtype=np.int16)
    board_ids = np.asarray([card_index(card) for card in board], dtype=np.int16)
    known = set(int(card_id) for card_id in hole_ids)
    known.update(int(card_id) for card_id in board_ids)
    remaining = np.asarray(
        [card_id for card_id in range(52) if card_id not in known], dtype=np.int16
    )

    board_need = 5 - int(board_ids.shape[0])
    per_trial = board_need + 2 * num_opponents
    if per_trial > int(remaining.shape[0]):
        raise ValueError(
            f"Need {per_trial} cards but only {int(remaining.shape[0])} remain"
        )

    seed = rng.seed if rng.seed is not None else rng.randint(0, 2**31 - 1)
    generator = np.random.default_rng(seed)
    keys = generator.random((num_trials, int(remaining.shape[0])))
    # Prefix of a random permutation: a uniform sample without replacement.
    order = np.argsort(keys, axis=1)[:, :per_trial]
    drawn = remaining[order]

    full_board = np.empty((num_trials, 5), dtype=np.int16)
    n_board = int(board_ids.shape[0])
    if n_board:
        full_board[:, :n_board] = board_ids
    if board_need:
        full_board[:, n_board:] = drawn[:, :board_need]

    hero = np.empty((num_trials, 7), dtype=np.int16)
    hero[:, 0:2] = hole_ids
    hero[:, 2:] = full_board
    hero_scores = score_hands(hero)

    better = np.zeros(num_trials, dtype=np.int32)
    equal = np.zeros(num_trials, dtype=np.int32)
    opp = np.empty((num_trials, 7), dtype=np.int16)
    opp[:, 2:] = full_board
    offset = board_need
    for _ in range(num_opponents):
        opp[:, 0:2] = drawn[:, offset : offset + 2]
        offset += 2
        opp_scores = score_hands(opp)
        better += opp_scores > hero_scores
        equal += opp_scores == hero_scores

    share = np.zeros(num_trials, dtype=np.float64)
    no_better = better == 0
    pure = no_better & (equal == 0)
    chop = no_better & (equal > 0)
    share[pure] = 1.0
    share[chop] = 1.0 / (equal[chop].astype(np.float64) + 1.0)
    return float(share.mean())


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
