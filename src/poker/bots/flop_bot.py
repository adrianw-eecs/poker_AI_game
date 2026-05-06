"""Flop-focused bot strategy with simplified pair detection."""

from poker.domain.action import Action, ActionType
from poker.rng import RNG
from poker.state.game_state import GameState, Street


class FlopBot:
    """Bot that plays a simplified flop-focused strategy.

    Pre-flop: Check or call to see the flop cheaply.
    Flop+: Raise with any pair, fold with nothing.
    Turn/River: Same logic (raise on pair, fold otherwise).

    Strategy is deliberately simple to minimize computation:
    - Pair detection is O(n) rank checking, not expensive hand evaluation.
    - No complex hand ranking needed.
    """

    def __init__(self, name: str = "FlopBot", seed: int | None = None) -> None:
        """Initialize the flop bot.

        Args:
            name: The display name for this bot.
            seed: Optional seed for reproducible raise amounts.
        """
        self._name = name
        self._rng = RNG(seed)

    @property
    def name(self) -> str:
        """Return the bot's name."""
        return self._name

    def act(self, state: GameState, legal: list[Action]) -> Action:
        """Choose an action based on simplified flop-focused strategy.

        Pre-flop: Check or call to see the flop.
        Flop+: Raise with pair, fold without.

        Args:
            state: The current game state.
            legal: List of legal actions available to this player.

        Returns:
            The chosen action.
        """
        # Build action map for easy lookup
        action_by_type: dict[ActionType, Action] = {}
        for action in legal:
            action_by_type[action.type] = action

        if state.street == Street.PREFLOP:
            return self._act_preflop(action_by_type)
        else:  # FLOP, TURN, RIVER
            return self._act_postflop(state, action_by_type)

    def _act_preflop(self, action_by_type: dict[ActionType, Action]) -> Action:
        """Pre-flop action: check or call to enter the flop.

        Never raise pre-flop; goal is to see flop cheaply.

        Args:
            action_by_type: Map of action types to available actions.

        Returns:
            A check or call action if available.
        """
        if ActionType.CHECK in action_by_type:
            return action_by_type[ActionType.CHECK]
        if ActionType.CALL in action_by_type:
            return action_by_type[ActionType.CALL]
        # Last resort
        return action_by_type.get(ActionType.FOLD, list(action_by_type.values())[0])

    def _act_postflop(
        self, state: GameState, action_by_type: dict[ActionType, Action]
    ) -> Action:
        """Post-flop action: raise with pair, otherwise fold.

        If fewer than 3 community cards (shouldn't happen post-flop but handle it):
        Check or call.

        If 3+ community cards: Check for pair.
        - Pair found: Raise 10-150 chips (clamped to legal range).
        - No pair: Fold.

        Args:
            state: The current game state.
            action_by_type: Map of action types to available actions.

        Returns:
            The chosen action.
        """
        if state.action_on_seat is None:
            return action_by_type.get(ActionType.FOLD, list(action_by_type.values())[0])

        player = state.players[state.action_on_seat]

        # If not enough community cards yet, check or call
        if len(state.community_cards) < 3:
            if ActionType.CHECK in action_by_type:
                return action_by_type[ActionType.CHECK]
            if ActionType.CALL in action_by_type:
                return action_by_type[ActionType.CALL]
            return action_by_type.get(ActionType.FOLD, list(action_by_type.values())[0])

        # Flop or later: check for pair
        if self._has_pair(player.hole_cards, state.community_cards):
            # Has pair: try to raise
            if ActionType.RAISE in action_by_type:
                return self._get_raise_action(action_by_type)

        # No pair: fold
        if ActionType.FOLD in action_by_type:
            return action_by_type[ActionType.FOLD]
        # Fallback if fold not available (shouldn't happen)
        if ActionType.CHECK in action_by_type:
            return action_by_type[ActionType.CHECK]
        if ActionType.CALL in action_by_type:
            return action_by_type[ActionType.CALL]
        return list(action_by_type.values())[0]

    def _has_pair(self, hole_cards: tuple, community_cards: tuple) -> bool:
        """Check if hole cards + community cards contain a pair.

        Simple O(n) operation: count ranks and look for duplicates.
        Much faster than full hand evaluation.

        Args:
            hole_cards: Player's two hole cards.
            community_cards: Community cards on board (3, 4, or 5).

        Returns:
            True if any rank appears 2+ times, False otherwise.
        """
        all_cards = hole_cards + community_cards
        ranks = [card.rank for card in all_cards]

        # Check if any rank appears 2 or more times
        for rank in set(ranks):
            if ranks.count(rank) >= 2:
                return True
        return False

    def _get_raise_action(self, action_by_type: dict[ActionType, Action]) -> Action:
        """Generate a raise action between 10-150 chips, clamped to legal range.

        Args:
            action_by_type: Map of action types to available actions.

        Returns:
            A raise action with amount between 10-150 (or legal range).
        """
        raise_action = action_by_type[ActionType.RAISE]
        min_raise = raise_action.amount

        # Get max from all-in if available
        all_in_action = action_by_type.get(ActionType.ALL_IN)
        max_raise = all_in_action.amount if all_in_action else min_raise

        # Target range: 10-150, clamped to legal [min_raise, max_raise]
        target_min = max(10, min_raise)
        target_max = min(150, max_raise)

        if target_min <= target_max:
            amount = self._rng.randint(target_min, target_max)
            return Action.raise_to(amount)
        else:
            # Legal range is above 150, use minimum legal raise
            return Action.raise_to(min_raise)

    def observe_result(self, final_state: GameState, reward: float) -> None:
        """Ignore the hand outcome (flop bot does not learn).

        Args:
            final_state: The final game state after the hand.
            reward: The normalized chip delta.
        """
        pass
