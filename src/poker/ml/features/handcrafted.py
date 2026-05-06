"""Handcrafted feature extraction for poker agents.

Features designed to be interpretable and useful for linear/tree models.
"""

import numpy as np
import numpy.typing as npt

from poker.state.game_state import GameState


def extract_handcrafted_features(state: GameState, seat: int) -> npt.NDArray[np.float32]:
    """Extract ~15 handcrafted features from game state.

    Features:
    1-3: Hole card strength (high rank, low rank, is_pair)
    4: Is suited
    5: Street (0-3 normalized)
    6: Pot odds
    7: Stack to pot ratio (SPR)
    8: Position relative to button (normalized)
    9: Number of active opponents
    10: Bet to call (normalized)
    11: Committed this street (normalized)
    12: Number of raises this street
    13: Has pair (post-flop only)
    14: Board paired (post-flop only)
    15: Effective stack (normalized)

    Args:
        state: Current game state.
        seat: Player's seat.

    Returns:
        Feature vector of shape (15,) normalized to [0, 1].
    """
    features: list[float] = []
    player = state.players[seat]

    # Normalize by max stack (50x starting is reasonable upper bound)
    max_stack = max(1, max(p.stack for p in state.players) * 2)
    max_stack = max(max_stack, state.config.starting_stack * 50)

    # 1-3: Hole card strength
    if len(player.hole_cards) >= 2:
        card1, card2 = player.hole_cards[0], player.hole_cards[1]
        high_rank = max(card1.rank.value, card2.rank.value)  # 2-14
        low_rank = min(card1.rank.value, card2.rank.value)
        is_pair = 1.0 if card1.rank == card2.rank else 0.0

        features.append(high_rank / 14.0)
        features.append(low_rank / 14.0)
        features.append(is_pair)
    else:
        features.extend([0.0, 0.0, 0.0])

    # 4: Is suited
    if len(player.hole_cards) >= 2:
        is_suited = 1.0 if player.hole_cards[0].suit == player.hole_cards[1].suit else 0.0
        features.append(is_suited)
    else:
        features.append(0.0)

    # 5: Street (0-3 normalized, ignoring SHOWDOWN)
    street_map = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
    street_val = street_map.get(state.street.value, 0)
    features.append(street_val / 3.0)

    # 6: Pot odds (bet_to_call / (pot + bet_to_call))
    total_pot = sum(p.amount for p in state.pots)
    if state.current_bet_to_call + total_pot > 0:
        pot_odds = state.current_bet_to_call / (total_pot + state.current_bet_to_call)
    else:
        pot_odds = 0.0
    features.append(min(1.0, pot_odds))

    # 7: Stack to pot ratio (SPR) - own_stack / pot
    if total_pot > 0:
        spr = min(1.0, player.stack / total_pot)
    else:
        spr = 1.0
    features.append(spr)

    # 8: Position relative to button (normalized)
    num_players = len(state.players)
    position = (seat - state.dealer_seat) % num_players
    features.append(position / num_players)

    # 9: Number of active opponents (normalized)
    num_active = sum(1 for p in state.players if not p.has_folded and p.seat != seat)
    features.append(num_active / (num_players - 1))

    # 10: Bet to call (normalized)
    bet_to_call_norm = state.current_bet_to_call / max_stack
    features.append(min(1.0, bet_to_call_norm))

    # 11: Committed this street (normalized)
    committed_norm = player.committed_this_street / max_stack
    features.append(min(1.0, committed_norm))

    # 12: Number of raises this street (normalized)
    num_raises = sum(
        1 for _, action in state.action_history_this_street
        if action.type.value in ("raise", "all_in")
    )
    features.append(min(1.0, num_raises / 5.0))

    # 13-14: Board features (post-flop only)
    has_pair = 0.0
    board_paired = 0.0
    if len(state.community_cards) >= 3:
        # Has pair (hole card matches board)
        board_ranks = [c.rank for c in state.community_cards]
        if len(player.hole_cards) >= 2:
            has_pair = 1.0 if any(c.rank in board_ranks for c in player.hole_cards) else 0.0

        # Board paired (two of same rank on board)
        board_paired = 1.0 if len(board_ranks) != len(set(board_ranks)) else 0.0

    features.append(has_pair)
    features.append(board_paired)

    # 15: Effective stack (min of own and largest opponent, normalized)
    if len(state.players) > 1:
        opponent_stacks = [p.stack for p in state.players if p.seat != seat]
        if opponent_stacks:
            effective_stack = min(player.stack, max(opponent_stacks))
        else:
            effective_stack = player.stack
    else:
        effective_stack = player.stack

    eff_stack_norm = effective_stack / max_stack
    features.append(min(1.0, eff_stack_norm))

    return np.array(features, dtype=np.float32)


def feature_names() -> list[str]:
    """Return names of the 15 features."""
    return [
        "high_card_rank",
        "low_card_rank",
        "is_pair",
        "is_suited",
        "street",
        "pot_odds",
        "spr",
        "position",
        "num_opponents",
        "bet_to_call",
        "committed_this_street",
        "num_raises",
        "has_pair",
        "board_paired",
        "effective_stack",
    ]
