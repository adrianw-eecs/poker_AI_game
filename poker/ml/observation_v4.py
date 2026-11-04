"""Observation v4: v3 plus hand strength and pot structure.

The vector is v3's 354 floats followed by 18 scalars. Layout:

    0    .. 354   identical to observation v3
    354  .. 364   strength (equity and kickers)
    364  .. 372   pot structure

Keeping the v3 prefix byte-identical means a v4 network differs from v3 only
by the input layer that reads the new tail.

Strength block, absolute indices:

    354  equity against every opponent still in the hand
    355  that equity divided by the opponent count (heads-up leaves it unchanged)
    356  ``HandRank.kickers[1]`` scaled from deuce to ace
    357  rank of the kicker tuple inside its hand category, in [0, 1]
    358  pair rank placed between the board's low and high card
    359  primary rank (``kickers[0]``) scaled from deuce to ace
    360  hand category, high-card 0 through straight-flush 1
    361  ``kickers[2]`` when the hand has one
    362  board class: 1 top pair or overpair, 0.5 middle, 0.25 bottom, 0 otherwise
    363  1 when the pair rank is above every board card

Preflop equity is the 169-class table already used by v2. Postflop equity is
a fast Monte Carlo. The table is the better preflop estimate, and dealing a
full board on top of it would spend the per-decision budget twice.

Structure block:

    364  fraction of the pot the hero is eligible to win
    365  all-in opponents / 8
    366  largest stack any opponent can still put in, saturated against the pot
    367  implied-odds proxy, pot / (pot + chips left after calling)
    368  fraction of the pot that sits in side pots
    369  1 when the hero is all-in
    370  opponents who can still bet / 8
    371  fraction of the hero's stack left after calling
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from poker.domain.card import Card
from poker.domain.hand import HandRank, HandType
from poker.evaluation.equity import monte_carlo_equity
from poker.evaluation.fast_evaluator import card_index, fast_evaluate, intra_category
from poker.ml.observation_v2 import _preflop_equity, _ratio, _sat
from poker.ml.observation_v2 import MoneyUnit
from poker.ml.observation_v3 import OBS_V3_DIM, build_observation_v3
from poker.ml.tendencies import TendencyTracker
from poker.rng import RNG
from poker.state.game_state import GameState

STRENGTH_DIM = 10
STRUCTURE_DIM = 8
OBS_V4_DIM = OBS_V3_DIM + STRENGTH_DIM + STRUCTURE_DIM

# 100 trials is the accuracy operating point (~0.6 ms heads-up). Dividing by
# the number of opponents holds the evaluation count roughly constant, which
# is what keeps a 6-max decision under the 2 ms observation budget.
OBS_EQUITY_TRIALS = 100

EQ_EQUITY = OBS_V3_DIM
EQ_EFFECTIVE = OBS_V3_DIM + 1
EQ_KICKER = OBS_V3_DIM + 2
EQ_INTRA = OBS_V3_DIM + 3
EQ_BOARD_RELATIVE = OBS_V3_DIM + 4
EQ_PRIMARY = OBS_V3_DIM + 5
EQ_CATEGORY = OBS_V3_DIM + 6
EQ_SECOND_KICKER = OBS_V3_DIM + 7
EQ_BOARD_CLASS = OBS_V3_DIM + 8
EQ_OVERPAIR = OBS_V3_DIM + 9

ST_ELIGIBLE = OBS_V3_DIM + STRENGTH_DIM
ST_ALL_IN = ST_ELIGIBLE + 1
ST_CEILING = ST_ELIGIBLE + 2
ST_IMPLIED = ST_ELIGIBLE + 3
ST_SIDE = ST_ELIGIBLE + 4
ST_HERO_ALL_IN = ST_ELIGIBLE + 5
ST_CAN_BET = ST_ELIGIBLE + 6
ST_STACK_LEFT = ST_ELIGIBLE + 7

# The loss reads this slot. It is the equity feature above.
EQUITY_INDEX = EQ_EQUITY

_PAIR_TYPES = frozenset(
    {
        HandType.PAIR,
        HandType.TWO_PAIR,
        HandType.THREE_OF_A_KIND,
        HandType.FULL_HOUSE,
        HandType.FOUR_OF_A_KIND,
    }
)


def equity_trials(opponents: int) -> int:
    """Monte Carlo trials for a decision against ``opponents`` live players."""
    return max(32, OBS_EQUITY_TRIALS // max(opponents, 1))


def _live_opponents(state: GameState, hero: int) -> int:
    return sum(
        1
        for seat, other in enumerate(state.players)
        if seat != hero and not other.has_folded and not other.is_eliminated
    )


def _equity_seed(
    hole: Sequence[Card], board: Sequence[Card], opponents: int
) -> int:
    """Card-derived seed so the same decision always encodes the same equity."""
    seed = 0xA5A55A5A
    for card in hole:
        seed = (seed * 16777619 + card_index(card)) & 0xFFFFFFFF
    for card in board:
        seed = (seed * 16777619 + card_index(card)) & 0xFFFFFFFF
    return seed ^ ((opponents * 0x9E3779B1) & 0xFFFFFFFF)


def _equity(hole: tuple[Card, ...], board: tuple[Card, ...], live: int) -> float:
    if len(hole) != 2:
        return 0.0
    if live <= 0:
        return 1.0
    if len(board) == 0:
        return _preflop_equity(hole, live)
    if len(board) not in (3, 4, 5):
        return _preflop_equity(hole, live)
    trials = equity_trials(live)
    return monte_carlo_equity(hole, board, live, trials, RNG(_equity_seed(hole, board, live)))


def _norm_kicker(hand: HandRank, index: int) -> float:
    if index >= len(hand.kickers):
        return 0.0
    return (hand.kickers[index] - 2) / 12.0


def _hole_only(hole: tuple[Card, ...]) -> tuple[float, float, float, float, float]:
    """Kicker, intra, primary, category, second kicker before a board exists."""
    if len(hole) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    high = max(int(hole[0].rank), int(hole[1].rank))
    low = min(int(hole[0].rank), int(hole[1].rank))
    primary = (high - 2) / 12.0
    if high == low:
        return 0.0, primary, primary, (int(HandType.PAIR) - 1) / 8.0, 0.0
    return (low - 2) / 12.0, primary, primary, 0.0, 0.0


def _board_views(hand: HandRank, board: tuple[Card, ...]) -> tuple[float, float, float]:
    """``(relative, class, overpair)`` for pair-style hands. Zeros otherwise."""
    if not board or hand.type not in _PAIR_TYPES or not hand.kickers:
        return 0.0, 0.0, 0.0
    pair_rank = hand.kickers[0]
    ranks = [int(card.rank) for card in board]
    high = max(ranks)
    low = min(ranks)
    if pair_rank > high:
        return 1.0, 1.0, 1.0
    if pair_rank == high:
        return 1.0, 1.0, 0.0
    if pair_rank <= low:
        # Bottom pair sits on the low card; an underpair sits strictly below it.
        board_class = 0.25 if pair_rank == low else 0.0
        return 0.0, board_class, 0.0
    span = high - low
    relative = (pair_rank - low) / span if span else 0.0
    return relative, 0.5, 0.0


def _strength_block(state: GameState, hero: int) -> list[float]:
    player = state.players[hero]
    hole = tuple(player.hole_cards)
    board = tuple(state.community_cards)
    live = _live_opponents(state, hero)
    equity = _equity(hole, board, live)
    effective = equity if live <= 1 else equity / float(live)
    cards = hole + board
    if len(cards) >= 5:
        hand = fast_evaluate(cards)
        kicker = _norm_kicker(hand, 1)
        intra = intra_category(hand)
        primary = _norm_kicker(hand, 0)
        category = (int(hand.type) - 1) / 8.0
        second = _norm_kicker(hand, 2)
        relative, board_class, overpair = _board_views(hand, board)
    else:
        kicker, intra, primary, category, second = _hole_only(hole)
        relative, board_class, overpair = 0.0, 0.0, 0.0
    return [
        equity,
        effective,
        kicker,
        intra,
        relative,
        primary,
        category,
        second,
        board_class,
        overpair,
    ]


def _structure_block(state: GameState, hero: int) -> list[float]:
    player = state.players[hero]
    pot_amounts = [pot.amount for pot in state.pots]
    built = sum(pot_amounts)
    total = built if built > 0 else sum(other.committed_this_hand for other in state.players)
    if player.has_folded:
        eligible = 0.0
    elif built <= 0:
        eligible = 1.0
    else:
        eligible_chips = sum(
            pot.amount for pot in state.pots if hero in pot.eligible_seats
        )
        eligible = eligible_chips / built

    all_in = 0
    can_bet = 0
    max_stack = 0
    for seat, other in enumerate(state.players):
        if seat == hero or other.has_folded or other.is_eliminated:
            continue
        if other.is_all_in:
            all_in += 1
        elif other.stack > 0:
            can_bet += 1
        if other.stack > max_stack:
            max_stack = other.stack

    ceiling = _sat(float(max_stack), float(max(total, 1)))
    call = max(0, state.current_bet_to_call - player.committed_this_street)
    remaining = max(0, player.stack - call)
    raw = total / max(1, remaining)
    implied = raw / (1.0 + raw)
    side = 0.0 if built <= 0 else (built - pot_amounts[0]) / built
    return [
        eligible,
        all_in / 8.0,
        ceiling,
        implied,
        side,
        1.0 if player.is_all_in else 0.0,
        can_bet / 8.0,
        _ratio(float(remaining), float(max(player.stack, 1))),
    ]


def build_observation_v4(
    state: GameState,
    seat: int,
    tendencies: TendencyTracker | None = None,
    money_unit: MoneyUnit = "buyin",
) -> npt.NDArray[np.float32]:
    """Build the v4 vector for ``seat``.

    The first ``OBS_V3_DIM`` entries are exactly ``build_observation_v3`` for
    the same state, seat, and tendency tracker.
    """
    base = build_observation_v3(state, seat, tendencies, money_unit)
    block = np.asarray(
        _strength_block(state, seat) + _structure_block(state, seat),
        dtype=np.float32,
    )
    if block.shape[0] != STRENGTH_DIM + STRUCTURE_DIM:
        raise RuntimeError(
            f"observation v4 tail is {block.shape[0]} floats, "
            f"expected {STRENGTH_DIM + STRUCTURE_DIM}"
        )
    np.clip(block, 0.0, 1.0, out=block)
    out = np.empty(OBS_V4_DIM, dtype=np.float32)
    out[:OBS_V3_DIM] = base
    out[OBS_V3_DIM:] = block
    return out
