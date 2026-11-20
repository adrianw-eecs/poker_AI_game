"""Action space definition and masking for PokerEnv."""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

from poker.domain.action import Action, ActionType
from poker.engine.action_validator import legal_actions, validate
from poker.engine.rules import may_raise
from poker.ml.encoder import dequantize_raise
from poker.state.game_state import GameState, Street


class BetScheme(Enum):
    """How discrete raise buckets become chip amounts.

    ``LEGACY`` is the original min-raise / all-in quantization. SD-CFR stays
    on this scheme. ``POT_FRACTION`` sizes raises off the pot after a call.
    """

    LEGACY = "legacy"
    POT_FRACTION = "pot_fraction"


# Bucket 3, 4, 5 under POT_FRACTION. Bucket 2 is the minimum (or a 2.5bb open).
_POT_FRACTIONS: dict[int, float] = {3: 0.5, 4: 1.0, 5: 2.0}


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


def min_raise_to(state: GameState, seat: int) -> int:
    """Smallest legal raise-to for this seat, matching the action validator."""
    player = state.players[seat]
    to_call = state.current_bet_to_call
    if player.committed_this_street >= to_call and to_call == 0:
        return state.config.big_blind
    increment = max(state.config.big_blind, state.last_raise_size)
    return to_call + increment


def capped_raise_bounds(
    state: GameState, seat: int, max_bet: int
) -> tuple[int, int] | None:
    """Return (min_raise_to, max_raise_to) under an optional street-commitment cap.

    ``max_bet`` of 0 leaves the ceiling at the player's stack. A positive cap
    is the most chips this player may be committed for on the current street.
    Returns None when no raise fits under that ceiling.
    """
    player = state.players[seat]
    if player.stack <= 0 or not may_raise(state, seat):
        return None
    stack_to = player.stack + player.committed_this_street
    ceiling = stack_to if max_bet <= 0 else min(stack_to, max_bet)
    min_to = min_raise_to(state, seat)
    if ceiling < min_to or min_to <= state.current_bet_to_call:
        return None
    return min_to, ceiling


def cap_action(state: GameState, seat: int, action: Action, max_bet: int) -> Action:
    """Shrink a raise or all-in so the street commitment stays within ``max_bet``.

    A cap of 0 leaves the action unchanged. When the cap is too small for a
    legal raise, the action becomes a call or a check.
    """
    if max_bet <= 0 or action.type not in (ActionType.RAISE, ActionType.ALL_IN):
        return action
    player = state.players[seat]
    stack_to = player.stack + player.committed_this_street
    to_call = state.current_bet_to_call
    bounds = capped_raise_bounds(state, seat, max_bet)
    if bounds is None:
        if player.committed_this_street >= to_call:
            return Action.check()
        return Action.call(min(to_call, stack_to))
    min_to, ceiling = bounds
    amount = min(action.amount, ceiling)
    if amount < min_to:
        if player.committed_this_street >= to_call:
            return Action.check()
        return Action.call(min(to_call, stack_to))
    if amount >= stack_to:
        return Action.all_in(stack_to)
    return Action.raise_to(amount)


def _pot_size(state: GameState) -> int:
    """Chips in the pots, falling back to commitments when pots are stale."""
    pot = sum(pot.amount for pot in state.pots)
    if pot > 0:
        return pot
    return sum(player.committed_this_hand for player in state.players)


def _unopened_preflop(state: GameState) -> bool:
    """True when the only live bet is the big blind."""
    return (
        state.street == Street.PREFLOP
        and state.current_bet_to_call == state.config.big_blind
        and state.last_raise_size == state.config.big_blind
    )


def pot_fraction_raise_to(
    state: GameState, seat: int, bucket: int, max_bet: int = 0
) -> int | None:
    """Target raise-to for a pot-fraction bucket, or None when no raise fits.

    Bucket 2 is the minimum legal raise, except an unopened preflop where it
    opens to 2.5 big blinds. Buckets 3-5 are 50%, 100% and 200% of the pot
    after the hero calls. Bucket 6 is all-in. The result is clamped to
    ``[min_raise_to, ceiling]``.
    """
    player = state.players[seat]
    if player.stack <= 0 or bucket < 2 or bucket > 6:
        return None
    bounds = capped_raise_bounds(state, seat, max_bet)
    stack_to = player.stack + player.committed_this_street
    if bucket == 6:
        if player.stack <= 0:
            return None
        # A shove above the current bet is a raise. It is illegal when a
        # short all-in left betting closed. A shove that only calls stays.
        if stack_to > state.current_bet_to_call and not may_raise(state, seat):
            return None
        if bounds is None:
            return stack_to
        return bounds[1]
    if bounds is None:
        return None
    min_to, ceiling = bounds
    if bucket == 2 and _unopened_preflop(state):
        target = int(round(2.5 * state.config.big_blind))
    elif bucket == 2:
        target = min_to
    else:
        fraction = _POT_FRACTIONS[bucket]
        call_chips = max(0, state.current_bet_to_call - player.committed_this_street)
        pot_after_call = _pot_size(state) + call_chips
        target = state.current_bet_to_call + int(round(fraction * pot_after_call))
    # An unopened 2.5bb open must not sit above the larger pot-fraction
    # buckets. Heads-up the 50% pot size is often only a min-raise, which
    # made bucket 3 smaller than bucket 2.
    if _unopened_preflop(state) and bucket < 6:
        target = max(target, int(round(2.5 * state.config.big_blind)))
    return min(ceiling, max(min_to, target))


def _raise_action_for_amount(state: GameState, seat: int, amount: int) -> Action:
    """Turn a clamped raise-to into a raise or an all-in."""
    stack_to = state.players[seat].stack + state.players[seat].committed_this_street
    if amount >= stack_to:
        return Action.all_in(stack_to)
    return Action.raise_to(amount)


def _dedup_raise_mask(
    mask: npt.NDArray[np.int32],
    state: GameState,
    seat: int,
    max_bet: int,
    scheme: BetScheme,
) -> None:
    """Clear a raise bucket whose clamped amount matches a cheaper bucket."""
    seen: set[tuple[str, int]] = set()
    for bucket in range(5):
        index = 2 + bucket
        if mask[index] == 0:
            continue
        try:
            action = action_index_to_action(
                index, state, seat, max_bet=max_bet, scheme=scheme
            )
        except ValueError:
            mask[index] = 0
            continue
        key = (action.type.value, action.amount)
        if key in seen:
            mask[index] = 0
        else:
            seen.add(key)


def _mask_pot_fraction(
    state: GameState, seat: int, max_bet: int
) -> npt.NDArray[np.int32]:
    """Legal mask for pot-fraction buckets, with duplicate amounts removed."""
    legal = legal_actions(state, seat)
    mask = np.zeros(7, dtype=np.int32)
    for action in legal:
        if action.type == ActionType.FOLD:
            mask[0] = 1
        elif action.type in (ActionType.CHECK, ActionType.CALL):
            mask[1] = 1

    seen: set[int] = set()
    call_actions = [action for action in legal if action.type == ActionType.CALL]
    if call_actions:
        seen.add(call_actions[0].amount)

    player = state.players[seat]
    shove = player.stack + player.committed_this_street if player.stack > 0 else None
    for bucket in range(2, 7):
        amount = pot_fraction_raise_to(state, seat, bucket, max_bet=max_bet)
        if amount is None or amount in seen:
            continue
        # The full-stack size belongs to bucket 6. Leaving it on an earlier
        # bucket cleared the shove index at 5-10bb, where every size clamps.
        if bucket < 6 and shove is not None and amount >= shove:
            continue
        action = _raise_action_for_amount(state, seat, amount)
        try:
            validate(state, seat, action)
        except Exception:
            continue
        seen.add(amount)
        mask[bucket] = 1

    if mask.sum() == 0:
        raise ValueError(f"No legal actions computed for seat {seat}")
    return mask


def build_action_mask(
    state: GameState,
    seat: int,
    max_bet: int = 0,
    scheme: BetScheme = BetScheme.LEGACY,
    dedup: bool = False,
) -> npt.NDArray[np.int32]:
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
    if scheme == BetScheme.POT_FRACTION:
        return _mask_pot_fraction(state, seat, max_bet)

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

    if max_bet > 0:
        mask[2:7] = 0
        bounds = capped_raise_bounds(state, seat, max_bet)
        if bounds is not None:
            min_to, ceiling = bounds
            seen: set[int] = set()
            for bucket in range(5):
                amount = dequantize_raise(bucket, min_to, ceiling)
                if amount in seen or amount < min_to or amount > ceiling:
                    continue
                seen.add(amount)
                mask[2 + bucket] = 1

    if dedup:
        _dedup_raise_mask(mask, state, seat, max_bet, scheme)

    if mask.sum() == 0:
        raise ValueError(f"No legal actions computed for seat {seat}")

    return mask


def action_index_to_action(
    action_index: int,
    state: GameState,
    seat: int,
    max_bet: int = 0,
    scheme: BetScheme = BetScheme.LEGACY,
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
        if scheme == BetScheme.POT_FRACTION:
            amount = pot_fraction_raise_to(state, seat, action_index, max_bet=max_bet)
            if amount is None:
                raise ValueError("Raise is not legal")
            return _raise_action_for_amount(state, seat, amount)

        if max_bet > 0:
            bounds = capped_raise_bounds(state, seat, max_bet)
            if bounds is None:
                raise ValueError("Raise is not legal")
            min_to, ceiling = bounds
            stack_to = state.players[seat].stack + state.players[seat].committed_this_street
            amount = min(ceiling, max(min_to, dequantize_raise(action_index - 2, min_to, ceiling)))
            if amount >= stack_to:
                return Action.all_in(stack_to)
            return Action.raise_to(amount)

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
