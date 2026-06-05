"""Action space definition and masking for PokerEnv."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from poker.domain.action import Action, ActionType
from poker.engine.action_validator import legal_actions
from poker.state.game_state import GameState


@dataclass
class ActionSpace:
    """Discrete action space for poker agent.

    Supports 7 discrete actions:
    - 0: Fold
    - 1: Check/Call
    - 2-6: Raise to 5 different quantized amounts (min, 20%, 40%, 60%, 80%, max)

    The agent selects an action index, which is validated against the
    legal action mask before execution.
    """

    num_actions: int = 7

    def sample(self, mask: npt.NDArray[np.int32]) -> int:
        """Sample a random legal action index.

        Args:
            mask: Action validity mask of shape (7,) with 1 for legal, 0 for illegal.

        Returns:
            A legal action index.

        Raises:
            ValueError: If no actions are legal.
        """
        legal_indices = np.where(mask)[0]
        if len(legal_indices) == 0:
            raise ValueError("No legal actions available (empty mask)")
        return int(np.random.choice(legal_indices))

    def contains(self, action: int) -> bool:
        """Check if an action index is in the space.

        Args:
            action: The action index.

        Returns:
            True if 0 <= action < 7.
        """
        return 0 <= action < self.num_actions


def build_action_mask(state: GameState, seat: int) -> npt.NDArray[np.int32]:
    """Build a legal action mask for a player.

    Returns a binary mask of shape (7,) where mask[i] = 1 iff action i is legal.

    Args:
        state: The current game state.
        seat: The seat of the player.

    Returns:
        A mask array with 1 for legal actions, 0 for illegal.

    Raises:
        ValueError: If no legal actions exist.
    """
    legal = legal_actions(state, seat)
    mask = np.zeros(7, dtype=np.int32)

    for action in legal:
        if action.type == ActionType.FOLD:
            mask[0] = 1
        elif action.type == ActionType.CHECK:
            mask[1] = 1
        elif action.type == ActionType.CALL:
            mask[1] = 1
        elif action.type == ActionType.RAISE:
            # Raise template: amount is minimum. Agent can choose 5 buckets.
            # All 5 raise buckets are legal if one raise is legal.
            mask[2:7] = 1
        elif action.type == ActionType.ALL_IN:
            # All-in is technically a raise. Allows choosing bucket 5 (max).
            mask[2:7] = 1

    if mask.sum() == 0:
        raise ValueError(f"No legal actions computed for seat {seat}")

    return mask


def action_index_to_action(
    action_index: int,
    state: GameState,
    seat: int,
) -> Action:
    """Convert a discrete action index to a concrete Action.

    Args:
        action_index: An integer 0-6 from the discrete space.
        state: The current game state.
        seat: The seat taking the action.

    Returns:
        A concrete Action with amount set.

    Raises:
        ValueError: If the action is illegal or invalid.
    """
    if action_index < 0 or action_index >= 7:
        raise ValueError(f"Invalid action index {action_index}, must be 0-6")

    legal = legal_actions(state, seat)

    if action_index == 0:  # Fold
        fold_actions = [a for a in legal if a.type == ActionType.FOLD]
        if not fold_actions:
            raise ValueError("Fold is not legal")
        return fold_actions[0]

    elif action_index == 1:  # Check/Call
        # Prefer call if available, otherwise check
        call_actions = [a for a in legal if a.type == ActionType.CALL]
        if call_actions:
            return call_actions[0]
        check_actions = [a for a in legal if a.type == ActionType.CHECK]
        if check_actions:
            return check_actions[0]
        raise ValueError("Check/Call is not legal")

    elif 2 <= action_index <= 6:  # Raise buckets
        raise_actions = [a for a in legal if a.type in (ActionType.RAISE, ActionType.ALL_IN)]
        if not raise_actions:
            raise ValueError("Raise is not legal")

        # Quantize the raise amounts
        raise_amounts = sorted([a.amount for a in raise_actions])
        num_raises = len(raise_amounts)

        if num_raises == 1:
            # Only one raise available (e.g., all-in)
            return raise_actions[0]

        # Map action_index (2-6) to bucket index (0-4)
        bucket = action_index - 2  # 0-4
        max_bucket = 4
        bucket_index = min(int(bucket * num_raises / (max_bucket + 1)), num_raises - 1)

        selected_amount = raise_amounts[bucket_index]
        matching_actions = [a for a in raise_actions if a.amount == selected_amount]
        return matching_actions[0]

    raise ValueError(f"Invalid action index {action_index}")


def action_to_action_index(action: Action, state: GameState, seat: int) -> int:
    """Convert a concrete Action to a discrete action index.

    This is primarily useful for recording expert demonstrations.

    Args:
        action: The action to convert.
        state: The current game state.
        seat: The seat taking the action.

    Returns:
        An integer 0-6.

    Raises:
        ValueError: If the action cannot be represented.
    """
    if action.type == ActionType.FOLD:
        return 0
    elif action.type == ActionType.CHECK:
        return 1
    elif action.type == ActionType.CALL:
        return 1
    elif action.type in (ActionType.RAISE, ActionType.ALL_IN):
        # Get all legal raises and find which bucket this amount falls into
        legal = legal_actions(state, seat)
        raise_actions = [a for a in legal if a.type in (ActionType.RAISE, ActionType.ALL_IN)]
        if not raise_actions:
            raise ValueError("Raise is not legal for this state")

        raise_amounts = sorted([a.amount for a in raise_actions])
        num_raises = len(raise_amounts)

        if num_raises == 1:
            return 2  # Default to first raise bucket

        # Find closest raise amount
        distances = [abs(amt - action.amount) for amt in raise_amounts]
        closest_idx = min(range(len(distances)), key=distances.__getitem__)

        # Map back to bucket index 2-6
        max_bucket = 4
        bucket = int(closest_idx * (max_bucket + 1) / num_raises)
        return min(2 + bucket, 6)

    raise ValueError(f"Unknown action type: {action.type}")
