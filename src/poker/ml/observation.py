"""Game state to observation conversion for ML agents."""

import numpy as np
import numpy.typing as npt

from poker.domain.action import ActionType
from poker.ml.encoder import cards_to_indices
from poker.state.game_state import GameState, Street


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

    # === Pad to 139 features ===
    # Keep last 3 features (indices 139-141) for important game state
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

    # === Final pad to exactly 142 ===
    target_size = 142
    while len(features) < target_size:
        features.append(0.0)

    if len(features) > target_size:
        features = features[:target_size]

    return np.array(features, dtype=np.float32)


def observation_spec() -> tuple[tuple[int, ...], str]:
    """Return the specification of observations.

    Returns:
        A tuple of (shape, dtype_name) for gym/gymnasium compatibility.
    """
    return ((142,), "float32")
