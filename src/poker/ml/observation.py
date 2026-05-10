"""Game state to observation conversion for ML agents."""

import numpy as np
import numpy.typing as npt

from poker.domain.action import ActionType
from poker.domain.card import Card
from poker.ml.encoder import cards_to_indices
from poker.state.game_state import GameState, Street


def _classify_hand_strength(hole_cards: tuple[Card, ...], community_cards: tuple[Card, ...]) -> int:
    """Classify hand strength into buckets.

    Returns bucket index [0-7]:
    0: Unpaired, no draw
    1: Pair, no draw
    2: Pair + draw
    3: Two-pair or better
    4: Straight draw (4-card)
    5: Flush draw (4-card)
    6: Made straight/flush
    7: Best hand (trips+)
    """
    if len(hole_cards) < 2:
        return 0

    # Count card ranks
    all_cards = list(hole_cards) + list(community_cards)
    ranks = [card.rank for card in all_cards]
    from collections import Counter
    rank_counts = Counter(ranks)

    # Check for trips or better
    if 3 in rank_counts.values() or 4 in rank_counts.values():
        return 7

    # Check for two-pair
    if list(rank_counts.values()).count(2) >= 2:
        return 3

    # Check for pair
    has_pair = 2 in rank_counts.values()

    # Check for made straight/flush (simplified)
    if len(community_cards) >= 3:
        # Very simplified: just check if we have all high cards
        high_cards = sum(1 for r in ranks if r >= 10)
        if high_cards >= 5:
            return 6

    # Check for draws (simplified)
    if len(community_cards) >= 3:
        # 4-card straight draw: 4 consecutive ranks
        unique_ranks = sorted(set(ranks))
        for i in range(len(unique_ranks) - 3):
            if unique_ranks[i + 3] - unique_ranks[i] == 3:
                return 4 if not has_pair else 2

        # 4-card flush draw: check suits
        suits = [card.suit for card in all_cards]
        from collections import Counter as SuitCounter
        suit_counts = SuitCounter(suits)
        if 4 in suit_counts.values():
            return 5 if not has_pair else 2

    # Pair with no draw, or unpaired
    return 1 if has_pair else 0


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

    Observation contains ~140 normalized features:
    - Hole cards: 2 card indices (0-51) normalized to [0, 1]
    - Community cards: 5 card indices (0-51) normalized to [0, 1]
    - Stack sizes: 9 normalized values (own + 8 opponents)
    - Committed amounts: 9 normalized values
    - Positions: 9 one-hot encodings (is_self, relative positions)
    - Folded flags: 9 binary values
    - Action counts: Number of bets, raises, calls, folds this street
    - Current bet to call: Normalized value
    - Pot amount: Normalized value
    - Street encoding: 4 one-hot bits (preflop, flop, turn, river)
    - Button/blinds: 3 binary flags (is_button, is_sb, is_bb)
    - Action on: Binary flag (is_action_on)

    Args:
        state: The current game state.
        seat: The player's seat number (0-indexed).

    Returns:
        A normalized numpy float32 array of shape (142,).

    Raises:
        ValueError: If seat is not in valid player range.
    """
    if not (0 <= seat < len(state.players)):
        raise ValueError(f"Seat {seat} out of range [0, {len(state.players) - 1}]")

    player = state.players[seat]
    num_players = len(state.players)

    # Normalize by max stack (50x starting stack is a reasonable upper bound)
    max_stack = max(1, max(p.stack for p in state.players) * 2)
    max_stack = max(max_stack, state.config.starting_stack * 50)

    features: list[float] = []

    # === Hole cards (2 features) ===
    hole_indices = cards_to_indices(player.hole_cards, num_cards=2)
    features.extend((hole_indices.clip(0) / 51.0).tolist())  # Clip -1 to 0, normalize to [0, 1]

    # === Community cards (5 features) ===
    community_indices = cards_to_indices(state.community_cards, num_cards=5)
    features.extend((community_indices.clip(0) / 51.0).tolist())  # -1 becomes 0 after clipping

    # === Stacks and committed (9 players * 2 features = 18) ===
    for p in state.players:
        stack_norm = float(p.stack) / max_stack
        committed_norm = float(p.committed_this_street) / max_stack
        features.extend([stack_norm, committed_norm])

    # === Position encodings: relative position and action flags (9 * 4 = 36) ===
    for i, p in enumerate(state.players):
        # Is self
        is_self = 1.0 if i == seat else 0.0

        # Relative position from button
        rel_pos = (i - state.dealer_seat) % num_players

        # Is button, small blind, big blind
        is_button = 1.0 if i == state.dealer_seat else 0.0
        is_sb = 1.0 if (i == (state.dealer_seat + 1) % num_players) else 0.0
        is_bb = 1.0 if (i == (state.dealer_seat + 2) % num_players) else 0.0

        features.extend([is_self, is_button, is_sb, is_bb])

    # === Folded flags (9 features) ===
    for p in state.players:
        features.append(float(p.has_folded))

    # === Street encoding (5 one-hot) ===
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

    # === Action history this street ===
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

    # Normalize by max reasonable actions per street
    max_actions_per_street = num_players * 2 + 5
    features.extend([
        float(num_bets) / max_actions_per_street,
        float(num_raises) / max_actions_per_street,
        float(num_calls) / max_actions_per_street,
        float(num_folds) / max_actions_per_street,
    ])

    # === Pot and bet to call ===
    pot_amount = sum(p.amount for p in state.pots)
    pot_norm = float(pot_amount) / max_stack
    bet_to_call_norm = float(state.current_bet_to_call) / max_stack

    features.extend([pot_norm, bet_to_call_norm])

    # === Hand strength bucket (8 features, one-hot) ===
    hand_strength = _classify_hand_strength(player.hole_cards, state.community_cards)
    hand_encoding = [0.0] * 8
    hand_encoding[min(hand_strength, 7)] = 1.0
    features.extend(hand_encoding)

    # === SPR - Stack-to-Pot Ratio (1 feature) ===
    spr = (player.stack / max(pot_amount, 1.0)) if pot_amount > 0 else 1.0
    spr_norm = np.log(spr + 1.0) / np.log(100.0)  # Log-scale, cap at 100:1
    features.append(float(np.clip(spr_norm, 0.0, 1.0)))

    # === Aggression metrics (4 features) ===
    own_agg = _compute_player_aggression(state, seat)
    opp_agg = _compute_table_aggression(state, seat)
    features.extend([own_agg, opp_agg, 0.0, 0.0])  # Last 2 are spare for future use

    # === Pad to required size ===
    # Current size = 2 + 5 + 18 + 36 + 9 + 5 + 4 + 2 + 8 + 1 + 4 = 94
    # Add back the earlier action history padding
    target_size_before_final = 139
    while len(features) < target_size_before_final:
        features.append(0.0)

    # === Final features: Action on, Deck, Blinds (indices 139-141) ===
    # Action on flag
    is_action_on = 1.0 if state.action_on_seat == seat else 0.0
    features.append(is_action_on)

    # Deck remaining (normalized)
    deck_norm = float(state.deck_remaining_count) / 52.0
    features.append(deck_norm)

    # Small/Big blind amounts (normalized) - will pad one more
    sb_norm = float(state.blind_level.small) / max_stack
    features.append(sb_norm)

    # === Final pad to exactly 155 (142 + 13 new features) ===
    target_size = 155
    while len(features) < target_size:
        features.append(0.0)

    if len(features) > target_size:
        features = features[:target_size]

    return np.array(features, dtype=np.float32)


def observation_spec() -> tuple[tuple[int, ...], str]:
    """Return the specification of observations.

    Returns:
        A tuple of (shape, dtype_name) for gym/gymnasium compatibility.
        Enhanced from 142 to 155 features (added hand strength, SPR, aggression).
    """
    return ((155,), "float32")
