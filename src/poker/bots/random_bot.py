"""Random bot implementation."""

from poker.domain.action import Action, ActionType
from poker.rng import RNG
from poker.state.game_state import GameState


class RandomBot:
    """Bot that selects actions uniformly at random from legal options.

    When raising, picks a random raise amount between the minimum valid
    raise and the all-in amount. Accepts a seed for reproducibility.
    """

    def __init__(self, name: str = "RandomBot", seed: int | None = None) -> None:
        """Initialize the random bot.

        Args:
            name: The display name for this bot.
            seed: Optional seed for reproducible action selection.
        """
        self._name = name
        self._rng = RNG(seed)

    @property
    def name(self) -> str:
        """Return the bot's name."""
        return self._name

    def act(self, state: GameState, legal: list[Action]) -> Action:
        """Choose an action using weighted probabilities from legal action types.

        Base weights are: Fold 20%, Check 35%, Call 15%, Raise 25%, All-in 5%.
        When an action is unavailable, its weight is redistributed to other available actions
        proportionally. If Check is disabled, its weight goes to Call; if Call is disabled,
        its weight goes to Check.
        If the chosen type is RAISE, picks a random raise amount between the minimum
        valid raise (from RAISE action) and the all-in amount.

        Args:
            state: The current game state (ignored; random bot is stateless).
            legal: List of legal actions available to this player.

        Returns:
            A randomly chosen legal action.
        """
        # Index legal actions by type
        action_by_type: dict[ActionType, Action] = {}
        for action in legal:
            action_by_type[action.type] = action

        # Base weights for each action type (before redistribution)
        base_weights = {
            "fold": 0.20,
            "check": 0.35,
            "call": 0.15,
            "raise": 0.25,
            "all_in": 0.05,
        }

        # Handle Check/Call fallback: if one is missing, give its weight to the other
        check_available = ActionType.CHECK in action_by_type
        call_available = ActionType.CALL in action_by_type

        if not check_available and call_available:
            # Check is unavailable, move its weight to call
            base_weights["call"] += base_weights["check"]
            base_weights["check"] = 0.0

        if not call_available and check_available:
            # Call is unavailable, move its weight to check
            base_weights["check"] += base_weights["call"]
            base_weights["call"] = 0.0

        # Map categories to action types
        categories = {
            "fold": [ActionType.FOLD],
            "check": [ActionType.CHECK],
            "call": [ActionType.CALL],
            "raise": [ActionType.RAISE],
            "all_in": [ActionType.ALL_IN],
        }

        # Determine which categories are available
        available_categories = {}
        unavailable_weight = 0.0

        for category, action_types in categories.items():
            is_available = any(at in action_by_type for at in action_types)
            if is_available and base_weights[category] > 0:
                available_categories[category] = base_weights[category]
            else:
                unavailable_weight += base_weights[category]

        # Redistribute unavailable weight proportionally to available categories
        if available_categories and unavailable_weight > 0:
            total_available = sum(available_categories.values())
            for category in available_categories:
                redistribute = (available_categories[category] / total_available) * unavailable_weight
                available_categories[category] += redistribute

        # Normalize weights to sum to 1.0
        total_weight = sum(available_categories.values())
        normalized_weights = {
            cat: weight / total_weight for cat, weight in available_categories.items()
        }

        # Pick a category using normalized weights
        categories_list = list(normalized_weights.keys())
        weights_list = [normalized_weights[cat] for cat in categories_list]
        chosen_category = self._rng.choices(categories_list, weights=weights_list, k=1)[0]

        # Pick a specific action from the chosen category
        available_in_category = [
            at for at in categories[chosen_category] if at in action_by_type
        ]
        chosen_type = self._rng.choice(available_in_category)

        if chosen_type == ActionType.RAISE:
            # Extract min/max bounds from legal actions (respects game state constraints)
            raise_action = action_by_type.get(ActionType.RAISE)
            all_in_action = action_by_type.get(ActionType.ALL_IN)

            if raise_action and all_in_action:
                min_raise_amount = raise_action.amount
                max_raise_amount = all_in_action.amount
                amount = self._rng.randint(min_raise_amount, max_raise_amount)
                return Action.raise_to(amount)
            else:
                # Fallback: use the raise action as-is if we can't extract bounds
                return raise_action if raise_action else action_by_type[chosen_type]

        return action_by_type[chosen_type]

    def observe_result(self, final_state: GameState, reward: float) -> None:
        """Ignore the hand outcome (random bot does not learn).

        Args:
            final_state: The final game state after the hand.
            reward: The normalized chip delta.
        """
        pass
