"""Game state to observation conversion for ML agents."""

import numpy as np
import numpy.typing as npt

from poker.domain.action import ActionType
from poker.domain.card import Card
from poker.domain.hand import HandType
from poker.engine.rules import blind_seats
from poker.evaluation.evaluator import evaluate as evaluate_hand
from poker.ml.encoder import cards_to_indices
from poker.state.game_state import GameState, Street

# Maximum seats supported; absent seats are zero-filled so the vector is always
# the same length regardless of how many players are actually in the game.
MAX_PLAYERS = 9
# 2 + 5 + MAX_PLAYERS*(2+4+1) + 5+4+2+8+1+4 + 1+1+1+1
OBS_DIM = 2 + 5 + MAX_PLAYERS * 7 + 5 + 4 + 2 + 8 + 1 + 4 + 1 + 1 + 1 + 1  # = 98


def _classify_hand_strength(hole_cards: tuple[Card, ...], community_cards: tuple[Card, ...]) -> int:
    """Classify hand strength into 8 buckets using the real evaluator.

    Returns bucket index [0-7]:
    0: High card, no draw
    1: One pair
    2: Pair + draw (flush or straight draw)
    3: Two pair
    4: Straight draw (4-card, no made hand)
    5: Flush draw (4-card, no made hand)
    6: Made straight or flush
    7: Three of a kind or better (trips, full house, quads, straight flush)
    """
    from collections import Counter

    if len(hole_cards) < 2:
        return 0

    all_cards = list(hole_cards) + list(community_cards)

    # Use the real evaluator when we have enough cards for a 5-card hand.
    if len(all_cards) >= 5:
        hand_rank = evaluate_hand(all_cards)
        ht = hand_rank.type
        if ht in (HandType.STRAIGHT_FLUSH, HandType.FOUR_OF_A_KIND,
                  HandType.FULL_HOUSE, HandType.THREE_OF_A_KIND):
            return 7
        if ht in (HandType.FLUSH, HandType.STRAIGHT):
            return 6
        if ht == HandType.TWO_PAIR:
            return 3
        if ht == HandType.PAIR:
            # Check for a draw to categorize as pair+draw vs bare pair.
            return _check_pair_with_draw(all_cards)
        # High card — check for draws.
        return _check_high_card_draws(all_cards)

    # Pre-flop or early streets: classify without evaluator.
    ranks = [card.rank for card in all_cards]
    suits = [card.suit for card in all_cards]
    rank_counts = Counter(ranks)

    if 3 in rank_counts.values() or 4 in rank_counts.values():
        return 7
    pairs = sum(1 for v in rank_counts.values() if v == 2)
    if pairs >= 2:
        return 3
    has_pair = pairs == 1
    # Flush draw needs at least 4 cards of same suit.
    suit_counts = Counter(suits)
    has_flush_draw = max(suit_counts.values()) >= 4
    # Straight draw: any 4 of 5 consecutive ranks (gap-1 span check).
    has_straight_draw = _has_straight_draw(ranks)

    if has_pair and (has_flush_draw or has_straight_draw):
        return 2
    if has_flush_draw:
        return 5
    if has_straight_draw:
        return 4
    return 1 if has_pair else 0


def _has_straight_draw(ranks: list) -> bool:
    """Return True if there are 4 cards within a span of 4 (open-ended or gutshot)."""
    unique = sorted({int(r) for r in ranks})
    # Ace plays low as 1 too.
    if 14 in unique:
        unique = [1] + unique
    for i in range(len(unique) - 3):
        if unique[i + 3] - unique[i] <= 4:
            return True
    return False


def _check_pair_with_draw(all_cards: list[Card]) -> int:
    """Return 2 if pair + draw, else 1 (bare pair)."""
    from collections import Counter
    ranks = [card.rank for card in all_cards]
    suits = [card.suit for card in all_cards]
    suit_counts = Counter(suits)
    has_flush_draw = max(suit_counts.values()) >= 4
    has_straight_draw = _has_straight_draw([int(r) for r in ranks])
    return 2 if (has_flush_draw or has_straight_draw) else 1


def _check_high_card_draws(all_cards: list[Card]) -> int:
    """Return 5 (flush draw), 4 (straight draw), or 0 (high card)."""
    from collections import Counter
    ranks = [card.rank for card in all_cards]
    suits = [card.suit for card in all_cards]
    suit_counts = Counter(suits)
    if max(suit_counts.values()) >= 4:
        return 5
    if _has_straight_draw([int(r) for r in ranks]):
        return 4
    return 0


def _compute_player_aggression(state: GameState, seat: int) -> float:
    """Compute aggression % for a specific player.

    Returns aggression rate in [0, 1] where 1 = all actions are raises/all-ins.
    """
    actions = [a for s, a in state.action_history_this_street if s == seat]
    if not actions:
        return 0.0

    aggressive = sum(1 for a in actions if a.type in [ActionType.RAISE, ActionType.ALL_IN])
    return float(aggressive) / len(actions)


def _compute_table_aggression(state: GameState, seat: int) -> float:
    """Compute average aggression % for opponents.

    Returns average aggression rate in [0, 1].
    """
    opponent_seats = [i for i in range(len(state.players)) if i != seat]
    if not opponent_seats:
        return 0.0

    agg_rates = []
    for opp_seat in opponent_seats:
        actions = [a for s, a in state.action_history_this_street if s == opp_seat]
        if actions:
            agg = sum(1 for a in actions if a.type in [ActionType.RAISE, ActionType.ALL_IN])
            agg_rates.append(float(agg) / len(actions))

    return float(np.mean(agg_rates)) if agg_rates else 0.0


def build_observation(state: GameState, seat: int) -> npt.NDArray[np.float32]:
    """Build a fixed-length observation vector from game state.

    Always produces a vector of length OBS_DIM (98) regardless of player count.
    Per-player sections always cover MAX_PLAYERS (9) slots; absent seats are zero-filled,
    which is equivalent to a player with an empty stack (eliminated/not present).

    Layout (98 features total):
    - Hole cards:                   2   (card index / 51)
    - Community cards:              5   (card index / 51, 0 if not yet dealt)
    - Stacks + committed:          18   (MAX_PLAYERS × 2)
    - Position flags:              36   (MAX_PLAYERS × 4: is_self, is_button, is_sb, is_bb)
    - Folded flags:                 9   (MAX_PLAYERS × 1)
    - Street one-hot:               5
    - Action counts:                4   (bets, raises, calls, folds this street)
    - Pot + bet to call:            2
    - Hand strength one-hot:        8
    - SPR:                          1
    - Aggression metrics:           4   (own, table, 2 spare)
    - Player count:                 1   (num_players / MAX_PLAYERS)  [NEW]
    - Action-on flag:               1
    - Deck remaining:               1
    - Small blind:                  1

    Args:
        state: The current game state.
        seat: The player's seat number (0-indexed).

    Returns:
        A normalized numpy float32 array of shape (OBS_DIM,).

    Raises:
        ValueError: If seat is not in valid player range.
    """
    if not (0 <= seat < len(state.players)):
        raise ValueError(f"Seat {seat} out of range [0, {len(state.players) - 1}]")

    player = state.players[seat]
    num_players = len(state.players)

    max_stack = max(1, max(p.stack for p in state.players) * 2)
    max_stack = max(max_stack, state.config.starting_stack * 50)

    features: list[float] = []

    # === Hole cards (2) ===
    hole_indices = cards_to_indices(player.hole_cards, num_cards=2)
    features.extend((hole_indices.clip(0) / 51.0).tolist())

    # === Community cards (5) ===
    community_indices = cards_to_indices(state.community_cards, num_cards=5)
    features.extend((community_indices.clip(0) / 51.0).tolist())

    # === Stacks and committed — MAX_PLAYERS fixed slots (18) ===
    for i in range(MAX_PLAYERS):
        if i < num_players:
            p = state.players[i]
            features.extend([float(p.stack) / max_stack, float(p.committed_this_street) / max_stack])
        else:
            features.extend([0.0, 0.0])

    # === Position flags — MAX_PLAYERS fixed slots (36) ===
    # Heads-up the button posts the small blind. Multiway both blinds sit left of it.
    sb_seat, bb_seat = blind_seats(state)
    for i in range(MAX_PLAYERS):
        if i < num_players:
            is_self = 1.0 if i == seat else 0.0
            is_button = 1.0 if i == state.dealer_seat else 0.0
            is_sb = 1.0 if i == sb_seat else 0.0
            is_bb = 1.0 if i == bb_seat else 0.0
            features.extend([is_self, is_button, is_sb, is_bb])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

    # === Folded flags — MAX_PLAYERS fixed slots (9) ===
    for i in range(MAX_PLAYERS):
        features.append(float(state.players[i].has_folded) if i < num_players else 0.0)

    # === Street one-hot (5) ===
    street_encoding = [0.0] * 5
    street_map = {
        Street.PREFLOP: 0,
        Street.FLOP: 1,
        Street.TURN: 2,
        Street.RIVER: 3,
        Street.SHOWDOWN: 4,
    }
    street_encoding[street_map[state.street]] = 1.0
    features.extend(street_encoding)

    # === Action counts this street (4) ===
    num_bets = 0
    num_raises = 0
    num_calls = 0
    num_folds = 0
    for _, action in state.action_history_this_street:
        if action.type == ActionType.FOLD:
            num_folds += 1
        elif action.type in (ActionType.CHECK, ActionType.CALL):
            num_calls += 1
        elif action.type in (ActionType.RAISE, ActionType.ALL_IN):
            num_raises += 1

    max_actions_per_street = MAX_PLAYERS * 2 + 5  # fixed constant = 23
    features.extend([
        float(num_bets) / max_actions_per_street,
        float(num_raises) / max_actions_per_street,
        float(num_calls) / max_actions_per_street,
        float(num_folds) / max_actions_per_street,
    ])

    # === Pot and bet to call (2) ===
    pot_amount = sum(p.amount for p in state.pots)
    features.extend([float(pot_amount) / max_stack, float(state.current_bet_to_call) / max_stack])

    # === Hand strength one-hot (8) ===
    hand_strength = _classify_hand_strength(player.hole_cards, state.community_cards)
    hand_encoding = [0.0] * 8
    hand_encoding[min(hand_strength, 7)] = 1.0
    features.extend(hand_encoding)

    # === SPR — Stack-to-Pot Ratio (1) ===
    spr = player.stack / max(pot_amount, 1.0)
    features.append(float(np.clip(np.log(spr + 1.0) / np.log(100.0), 0.0, 1.0)))

    # === Aggression metrics (4) ===
    features.extend([_compute_player_aggression(state, seat), _compute_table_aggression(state, seat), 0.0, 0.0])

    # === Player count (1) — new; disambiguates empty slot from absent seat ===
    features.append(float(num_players) / MAX_PLAYERS)

    # === Action-on flag, deck remaining, small blind (3) ===
    features.append(1.0 if state.action_on_seat == seat else 0.0)
    features.append(float(state.deck_remaining_count) / 52.0)
    features.append(float(state.blind_level.small) / max_stack)

    assert len(features) == OBS_DIM, f"Expected {OBS_DIM} features, got {len(features)}"
    return np.array(features, dtype=np.float32)


def observation_spec(version: int = 1) -> tuple[tuple[int, ...], str]:
    """Return the observation shape and dtype for a versioned encoder.

    Version 1 is the fixed 98-d hero-absolute vector. Version 2 is the
    hero-relative encoder in ``observation_v2``.
    """
    if version == 1:
        return ((OBS_DIM,), "float32")
    if version == 2:
        from poker.ml.observation_v2 import OBS_V2_DIM

        return ((OBS_V2_DIM,), "float32")
    raise ValueError(f"Unknown observation version {version}")


# Production NFSP checkpoints were trained on a packed 155-d vector: live seats
# only, then zero-pad to index 139, then action-on / deck / small-blind.
LEGACY_OBS_DIM = 155
_LEGACY_PREFIX_PAD = 139


def adapt_observation(obs: npt.NDArray[np.float32], target_dim: int) -> npt.NDArray[np.float32]:
    """Match an observation vector to the width a loaded network expects.

    Current env observations are fixed-width ``OBS_DIM``. Checkpoints trained
    before that layout expect ``LEGACY_OBS_DIM`` with absent seats omitted
    rather than zero-filled.
    """
    vector = np.asarray(obs, dtype=np.float32).reshape(-1)
    if vector.shape[0] == target_dim:
        return vector
    if vector.shape[0] == OBS_DIM and target_dim == LEGACY_OBS_DIM:
        return _pack_fixed_obs_to_legacy(vector)
    raise ValueError(
        f"Cannot adapt observation of length {vector.shape[0]} to {target_dim}"
    )


def _pack_fixed_obs_to_legacy(obs: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Repack a fixed 9-seat observation into the legacy 155-d layout."""
    hole = obs[0:2]
    community = obs[2:7]
    stacks = obs[7:25].reshape(MAX_PLAYERS, 2)
    positions = obs[25:61].reshape(MAX_PLAYERS, 4)
    folded = obs[61:70]
    street = obs[70:75]
    actions = obs[75:79]
    pot = obs[79:81]
    hand = obs[81:89]
    spr = obs[89:90]
    aggression = obs[90:94]
    num_players = int(np.clip(np.rint(float(obs[94]) * MAX_PLAYERS), 2, MAX_PLAYERS))
    tail = obs[95:98]

    packed = np.concatenate(
        [
            hole,
            community,
            stacks[:num_players].reshape(-1),
            positions[:num_players].reshape(-1),
            folded[:num_players],
            street,
            actions,
            pot,
            hand,
            spr,
            aggression,
        ]
    )
    if packed.shape[0] < _LEGACY_PREFIX_PAD:
        packed = np.pad(packed, (0, _LEGACY_PREFIX_PAD - packed.shape[0]))
    else:
        packed = packed[:_LEGACY_PREFIX_PAD]

    legacy = np.concatenate([packed, tail])
    if legacy.shape[0] < LEGACY_OBS_DIM:
        legacy = np.pad(legacy, (0, LEGACY_OBS_DIM - legacy.shape[0]))
    return legacy.astype(np.float32)
