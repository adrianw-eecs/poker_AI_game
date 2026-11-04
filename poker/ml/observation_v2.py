"""Hero-relative observation for NFSP v2.

The vector is one flat float32 buffer. Card ids sit in their own block so a
network can cast them to embedding indices. Every other block is in ``[0, 1]``.
Seat slot 0 is the hero and slot ``k`` is ``k`` seats clockwise.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from poker.domain.action import ActionType
from poker.domain.card import Card
from poker.domain.hand import HandType
from poker.engine.rules import blind_seats
from poker.evaluation.evaluator import evaluate
from poker.ml.action_space import min_raise_to
from poker.ml.encoder import card_to_index
from poker.state.game_state import GameState, Street

MAX_SEATS = 9
CARD_SLOTS = 7
DERIVED_DIM = 20
HERO_DIM = 16
SEAT_DIM = 15
STREET_DIM = 10
N_STREET_SUMMARIES = 4
ACTION_SLOTS = 6
ACTION_DIM = 14
SCALAR_DIM = 16
EMPTY_CARD = 52

_CARD_ID_END = CARD_SLOTS
_DERIVED_END = _CARD_ID_END + DERIVED_DIM
_HERO_END = _DERIVED_END + HERO_DIM
_SEAT_END = _HERO_END + MAX_SEATS * SEAT_DIM
_STREET_END = _SEAT_END + N_STREET_SUMMARIES * STREET_DIM
_ACTION_END = _STREET_END + ACTION_SLOTS * ACTION_DIM
OBS_V2_DIM = _ACTION_END + SCALAR_DIM

_EQUITY_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "data" / "preflop_equity.npy"
_EQUITY: np.ndarray | None = None
_EQUITY_LOADED = False


@dataclass(frozen=True)
class ObsSpecV2:
    """Byte offsets of each block inside the flat vector."""

    dim: int = OBS_V2_DIM
    card_ids: tuple[int, int] = (0, _CARD_ID_END)
    derived: tuple[int, int] = (_CARD_ID_END, _DERIVED_END)
    hero: tuple[int, int] = (_DERIVED_END, _HERO_END)
    seats: tuple[int, int] = (_HERO_END, _SEAT_END)
    seat_dim: int = SEAT_DIM
    street_hist: tuple[int, int] = (_SEAT_END, _STREET_END)
    street_dim: int = STREET_DIM
    action_hist: tuple[int, int] = (_STREET_END, _ACTION_END)
    action_dim: int = ACTION_DIM
    action_slots: int = ACTION_SLOTS
    scalars: tuple[int, int] = (_ACTION_END, OBS_V2_DIM)
    empty_card: int = EMPTY_CARD


OBS_SPEC_V2 = ObsSpecV2()


def preflop_combo_index(card_a: Card, card_b: Card) -> int:
    """Map two hole cards onto the 169-class preflop index.

    Pairs occupy 0-12 (deuce pair through ace pair). Suited non-pairs occupy
    13-90 and offsuit non-pairs occupy 91-168, each in high-rank-major order.
    """
    high = max(int(card_a.rank), int(card_b.rank))
    low = min(int(card_a.rank), int(card_b.rank))
    hi = high - 2
    lo = low - 2
    if hi == lo:
        return hi
    combo = 0
    found = 0
    for rank_hi in range(1, 13):
        for rank_lo in range(rank_hi):
            if rank_hi == hi and rank_lo == lo:
                found = combo
            combo += 1
    base = 13 if card_a.suit == card_b.suit else 13 + 78
    return base + found


def load_preflop_equity() -> np.ndarray | None:
    """Load the 169 x 8 equity table, or None when it has not been built."""
    global _EQUITY, _EQUITY_LOADED
    if not _EQUITY_LOADED:
        _EQUITY_LOADED = True
        if _EQUITY_PATH.is_file():
            _EQUITY = np.load(_EQUITY_PATH)
    return _EQUITY


def _clip01(value: float) -> float:
    """Clamp a scalar into ``[0, 1]``.

    Deliberately avoids ``np.clip`` on a Python scalar: that costs ~2.9 us per
    call, and one observation makes ~216 such calls, so it accounted for the
    bulk of the 233 us build time. Plain comparisons are ~57x faster and behave
    identically for ``nan`` (propagates) and ``+-inf`` (clamp to 1.0 / 0.0).
    """
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _ratio(numer: float, denom: float) -> float:
    # ``not (denom > 0)`` rather than ``denom <= 0`` so a NaN denominator keeps
    # the old ``np.clip`` behaviour of returning 0.0 instead of propagating NaN.
    if not (denom > 0):
        return 0.0
    value = numer / denom
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _sat(numer: float, denom: float) -> float:
    """Map a non-negative ratio into ``[0, 1]`` without a hard ceiling of 1."""
    if numer <= 0:
        return 0.0
    value = numer / (numer + max(denom, 1.0))
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _connectivity(ranks: list[int]) -> float:
    if not ranks:
        return 0.0
    unique = set(ranks)
    if 14 in unique:
        unique.add(1)
    best = 1
    for start in range(1, 11):
        window = sum(1 for rank in range(start, start + 5) if rank in unique)
        best = max(best, window)
    return best / 5.0


def _derived_cards(hole: tuple[Card, ...], board: tuple[Card, ...]) -> list[float]:
    feats = [0.0] * DERIVED_DIM
    if len(hole) >= 2:
        high, low = sorted(hole[:2], key=lambda card: int(card.rank), reverse=True)
        feats[0] = float(high.suit == low.suit)
        feats[1] = float(high.rank == low.rank)
        feats[2] = (int(high.rank) - int(low.rank)) / 12.0
        feats[3] = (int(high.rank) - 2) / 12.0
        feats[4] = (int(low.rank) - 2) / 12.0
    if board:
        ranks = [int(card.rank) for card in board]
        suits = [card.suit for card in board]
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)
        max_rank = max(rank_counts.values())
        max_suit = max(suit_counts.values())
        feats[5] = float(max_rank >= 2)
        feats[6] = float(max_rank >= 3)
        feats[7] = float(max_suit >= 3)
        feats[8] = float(max_suit >= 4)
        feats[9] = float(max_suit >= 5)
        feats[10] = _connectivity(ranks)
        feats[11] = (max(ranks) - 2) / 12.0
        feats[12] = len(board) / 5.0
        feats[13] = float(len(board) >= 3)
        feats[14] = float(len(board) >= 4)
        feats[15] = float(len(board) >= 5)
        feats[16] = float(len(board) >= 3 and max_suit == len(board))
        feats[17] = float(len(board) >= 3 and len(suit_counts) == 2)
        feats[18] = float(len(board) >= 3 and max_suit == 1)
        feats[19] = (min(ranks) - 2) / 12.0
    return feats


def _made_hand_index(hole: tuple[Card, ...], board: tuple[Card, ...]) -> int:
    cards = list(hole) + list(board)
    if len(cards) >= 5:
        return int(evaluate(cards).type) - 1
    if len(hole) < 2:
        return 0
    counts = Counter(int(card.rank) for card in cards)
    if 4 in counts.values():
        return int(HandType.FOUR_OF_A_KIND) - 1
    if 3 in counts.values() and 2 in counts.values():
        return int(HandType.FULL_HOUSE) - 1
    if 3 in counts.values():
        return int(HandType.THREE_OF_A_KIND) - 1
    pairs = sum(1 for count in counts.values() if count == 2)
    if pairs >= 2:
        return int(HandType.TWO_PAIR) - 1
    if pairs == 1:
        return int(HandType.PAIR) - 1
    return int(HandType.HIGH_CARD) - 1


def _straight_draw_flags(cards: list[Card]) -> tuple[float, float]:
    ranks = {int(card.rank) for card in cards}
    if 14 in ranks:
        ranks.add(1)
    made = any(all((start + offset) in ranks for offset in range(5)) for start in range(1, 11))
    if made:
        return 0.0, 0.0
    oesd = False
    for start in range(2, 11):
        run = [start + offset for offset in range(4)]
        if all(rank in ranks for rank in run) and (start - 1) <= 14 and (start + 4) <= 14:
            oesd = True
            break
    gut = False
    if not oesd:
        for start in range(1, 11):
            window = [start + offset for offset in range(5)]
            if sum(1 for rank in window if rank in ranks) == 4:
                gut = True
                break
    return (1.0 if oesd else 0.0, 1.0 if gut else 0.0)


def _draw_flags(hole: tuple[Card, ...], board: tuple[Card, ...]) -> tuple[float, float, float, float]:
    if len(hole) < 2 or len(board) < 3:
        return 0.0, 0.0, 0.0, 0.0
    cards = list(hole) + list(board)
    suit_counts = Counter(card.suit for card in cards)
    hole_suits = Counter(card.suit for card in hole)
    flush_draw = 0.0
    if not any(count >= 5 for count in suit_counts.values()):
        for suit, count in suit_counts.items():
            if count == 4 and hole_suits[suit] >= 1:
                flush_draw = 1.0
    oesd, gutshot = _straight_draw_flags(cards)
    combo = 1.0 if flush_draw and (oesd or gutshot) else 0.0
    return flush_draw, oesd, gutshot, combo


def _preflop_equity(hole: tuple[Card, ...], opponents: int) -> float:
    if len(hole) != 2:
        return 0.0
    table = load_preflop_equity()
    if table is None:
        return 0.0
    column = min(max(opponents, 1), table.shape[1]) - 1
    return _clip01(float(table[preflop_combo_index(hole[0], hole[1]), column]))


def _relative(seat: int, hero: int, num_players: int) -> int:
    return (seat - hero) % num_players


def _split_streets(
    history: list[tuple[int, object]], ends: tuple[int, ...]
) -> list[list[tuple[int, object]]]:
    segments: list[list[tuple[int, object]]] = []
    start = 0
    bounds = [*ends, len(history)]
    for end in bounds:
        segments.append(history[start:end])
        start = end
        if len(segments) == N_STREET_SUMMARIES:
            break
    while len(segments) < N_STREET_SUMMARIES:
        segments.append([])
    return segments


def _street_index(action_index: int, ends: tuple[int, ...]) -> int:
    for street, end in enumerate(ends):
        if action_index < end:
            return street
    return min(len(ends), N_STREET_SUMMARIES - 1)


def _hero_block(state: GameState, hero: int) -> list[float]:
    player = state.players[hero]
    feats = [0.0] * HERO_DIM
    made = _made_hand_index(player.hole_cards, state.community_cards)
    if 0 <= made < 9:
        feats[made] = 1.0
    flush_draw, oesd, gutshot, combo = _draw_flags(player.hole_cards, state.community_cards)
    feats[9] = flush_draw
    feats[10] = oesd
    feats[11] = gutshot
    feats[12] = combo
    opponents = sum(
        1
        for seat, other in enumerate(state.players)
        if seat != hero and not other.has_folded and not other.is_eliminated
    )
    if state.street == Street.PREFLOP:
        feats[13] = _preflop_equity(player.hole_cards, opponents)
    if len(player.hole_cards) >= 2 and player.hole_cards[0].rank == player.hole_cards[1].rank:
        feats[14] = 1.0
    hole_ranks = {card.rank for card in player.hole_cards}
    if hole_ranks and any(card.rank in hole_ranks for card in state.community_cards):
        feats[15] = 1.0
    return feats


def _seat_block(state: GameState, hero: int, table_seat: int, pot: int) -> list[float]:
    num_players = len(state.players)
    feats = [0.0] * SEAT_DIM
    if table_seat < 0 or table_seat >= num_players:
        return feats
    player = state.players[table_seat]
    sb_seat, bb_seat = blind_seats(state)
    feats[0] = 1.0
    feats[1] = float(not player.has_folded and not player.is_eliminated)
    feats[2] = float(player.is_all_in)
    big_blind = max(state.config.big_blind, 1)
    feats[3] = _clip01(float(np.log1p(player.stack / big_blind) / np.log1p(200.0)))
    feats[4] = _ratio(player.committed_this_street, max(pot, 1))
    feats[5] = _ratio(player.committed_this_hand, max(state.config.starting_stack, 1))
    feats[6] = float(table_seat == state.dealer_seat)
    feats[7] = float(table_seat == sb_seat)
    feats[8] = float(table_seat == bb_seat)
    feats[9] = float(any(seat == table_seat for seat, _ in state.action_history_this_street))
    last = None
    for seat, action in reversed(state.action_history_this_street):
        if seat == table_seat:
            last = action
            break
    if last is not None:
        mapping = {
            ActionType.FOLD: 10,
            ActionType.CHECK: 11,
            ActionType.CALL: 12,
            ActionType.RAISE: 13,
            ActionType.ALL_IN: 13,
        }
        slot = mapping.get(last.type)
        if slot is not None:
            feats[slot] = 1.0
    raises = sum(
        1
        for seat, action in state.action_history_this_hand
        if seat == table_seat and action.type in (ActionType.RAISE, ActionType.ALL_IN)
    )
    feats[14] = _ratio(raises, MAX_SEATS)
    return feats


def _street_summary(
    state: GameState,
    hero: int,
    actions: list[tuple[int, object]],
    pot_entering: int,
    reached: bool,
) -> list[float]:
    feats = [0.0] * STREET_DIM
    if not reached and not actions:
        return feats
    num_players = len(state.players)
    raises = calls = folds = checks = 0
    last_aggressor = None
    largest = 0
    for seat, action in actions:
        if action.type == ActionType.FOLD:
            folds += 1
        elif action.type == ActionType.CHECK:
            checks += 1
        elif action.type == ActionType.CALL:
            calls += 1
        elif action.type in (ActionType.RAISE, ActionType.ALL_IN):
            raises += 1
            last_aggressor = seat
            largest = max(largest, action.amount)
    pot = max(sum(pot.amount for pot in state.pots), 1)
    feats[0] = _ratio(raises, MAX_SEATS)
    feats[1] = _ratio(calls, MAX_SEATS)
    feats[2] = _ratio(folds, MAX_SEATS)
    feats[3] = _ratio(checks, MAX_SEATS)
    if last_aggressor is not None:
        feats[4] = (_relative(last_aggressor, hero, num_players) + 1) / MAX_SEATS
        feats[8] = float(last_aggressor == hero)
    feats[5] = _ratio(largest, pot)
    feats[6] = _ratio(pot_entering, max(state.config.starting_stack, 1))
    feats[7] = 1.0 if reached or actions else 0.0
    feats[9] = _ratio(len(actions), MAX_SEATS * 2)
    return feats


def _action_slot(
    state: GameState, hero: int, item: tuple[int, object] | None, street: int
) -> list[float]:
    feats = [0.0] * ACTION_DIM
    if item is None:
        return feats
    seat, action = item
    num_players = len(state.players)
    pot = max(sum(pot.amount for pot in state.pots), 1)
    feats[0] = _relative(seat, hero, num_players) / max(MAX_SEATS - 1, 1)
    type_index = {
        ActionType.FOLD: 1,
        ActionType.CHECK: 2,
        ActionType.CALL: 3,
        ActionType.RAISE: 4,
        ActionType.ALL_IN: 5,
    }.get(action.type)
    if type_index is not None:
        feats[type_index] = 1.0
    feats[6] = _ratio(action.amount, pot)
    if 0 <= street < 4:
        feats[7 + street] = 1.0
    feats[11] = float(action.type == ActionType.ALL_IN)
    feats[12] = street / 3.0
    feats[13] = 1.0
    return feats


def _scalars(state: GameState, hero: int) -> list[float]:
    player = state.players[hero]
    pot = sum(pot.amount for pot in state.pots)
    if pot <= 0:
        pot = sum(other.committed_this_hand for other in state.players)
    call_chips = max(0, state.current_bet_to_call - player.committed_this_street)
    feats = [0.0] * SCALAR_DIM
    street_index = {
        Street.PREFLOP: 0,
        Street.FLOP: 1,
        Street.TURN: 2,
        Street.RIVER: 3,
        Street.SHOWDOWN: 4,
    }[state.street]
    feats[street_index] = 1.0
    feats[5] = _ratio(pot, max(state.config.starting_stack, 1))
    feats[6] = _ratio(call_chips, max(pot, 1))
    feats[7] = _ratio(call_chips, pot + call_chips) if call_chips else 0.0
    spr = player.stack / max(pot, 1)
    feats[8] = _clip01(float(np.log1p(spr) / np.log(100.0)))
    opponents = [
        other.stack
        for seat, other in enumerate(state.players)
        if seat != hero and not other.has_folded and not other.is_eliminated
    ]
    effective = min([player.stack, *opponents]) if opponents else player.stack
    feats[9] = _sat(effective, max(pot, 1))
    try:
        min_to = min_raise_to(state, hero)
    except Exception:
        min_to = state.current_bet_to_call + state.config.big_blind
    feats[10] = _sat(min_to, max(pot, 1))
    feats[11] = _sat(player.stack + player.committed_this_street, max(pot, 1))
    active = sum(1 for other in state.players if not other.has_folded and not other.is_eliminated)
    feats[12] = active / MAX_SEATS
    feats[13] = len(state.players) / MAX_SEATS
    feats[14] = _relative(hero, state.dealer_seat, len(state.players)) / max(MAX_SEATS - 1, 1)
    feats[15] = float(player.committed_this_street >= state.current_bet_to_call and player.stack > 0)
    return feats


def build_observation_v2(state: GameState, seat: int) -> npt.NDArray[np.float32]:
    """Build the hero-relative v2 vector for ``seat``."""
    num_players = len(state.players)
    if not 0 <= seat < num_players:
        raise ValueError(f"Seat {seat} out of range [0, {num_players - 1}]")

    player = state.players[seat]
    hole = tuple(sorted(player.hole_cards, key=lambda card: (int(card.rank), int(card.suit)), reverse=True))
    card_ids = [float(EMPTY_CARD)] * CARD_SLOTS
    for index, card in enumerate(hole[:2]):
        card_ids[index] = float(card_to_index(card))
    for index, card in enumerate(state.community_cards[:5]):
        card_ids[2 + index] = float(card_to_index(card))

    pot = sum(pot.amount for pot in state.pots)
    features: list[float] = []
    features.extend(card_ids)
    features.extend(_derived_cards(hole, state.community_cards))
    features.extend(_hero_block(state, seat))

    for offset in range(MAX_SEATS):
        table_seat = (seat + offset) % num_players if offset < num_players else -1
        features.extend(_seat_block(state, seat, table_seat, pot))

    segments = _split_streets(state.action_history_this_hand, state.street_history_ends)
    current_street = {
        Street.PREFLOP: 0,
        Street.FLOP: 1,
        Street.TURN: 2,
        Street.RIVER: 3,
        Street.SHOWDOWN: 4,
    }[state.street]
    for street in range(N_STREET_SUMMARIES):
        entering = state.pots_entering_street[street] if street < len(state.pots_entering_street) else 0
        reached = street <= current_street
        features.extend(_street_summary(state, seat, segments[street], entering, reached))

    indexed = list(enumerate(state.action_history_this_hand))
    tail = indexed[-ACTION_SLOTS:]
    padding = ACTION_SLOTS - len(tail)
    slots: list[tuple[int, object] | None] = [None] * padding
    streets: list[int] = [0] * padding
    for action_index, item in tail:
        slots.append(item)
        streets.append(_street_index(action_index, state.street_history_ends))
    for item, street in zip(slots, streets):
        features.extend(_action_slot(state, seat, item, street))

    features.extend(_scalars(state, seat))
    if len(features) != OBS_V2_DIM:
        raise RuntimeError(f"observation v2 built {len(features)} floats, expected {OBS_V2_DIM}")
    vector = np.asarray(features, dtype=np.float32)
    vector[OBS_SPEC_V2.card_ids[0] : OBS_SPEC_V2.card_ids[1]] = np.clip(
        vector[OBS_SPEC_V2.card_ids[0] : OBS_SPEC_V2.card_ids[1]], 0, EMPTY_CARD
    )
    rest = vector.copy()
    rest[OBS_SPEC_V2.card_ids[0] : OBS_SPEC_V2.card_ids[1]] = 0
    vector_rest = np.clip(rest, 0.0, 1.0)
    vector_rest[OBS_SPEC_V2.card_ids[0] : OBS_SPEC_V2.card_ids[1]] = vector[
        OBS_SPEC_V2.card_ids[0] : OBS_SPEC_V2.card_ids[1]
    ]
    return vector_rest
