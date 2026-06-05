"""SD-CFR poker bot for inference against trained SDCFRModel."""

import numpy as np
import numpy.typing as npt

from poker.domain.action import Action
from poker.engine.action_validator import legal_actions as compute_legal_actions
from poker.ml.action_space import action_index_to_action, action_to_action_index, build_action_mask
from poker.ml.models.sdcfr_model import SDCFRModel
from poker.ml.observation import build_observation
from poker.state.game_state import GameState


class SDCFRBot:
    """Poker bot that plays according to a trained SD-CFR strategy.

    Uses stochastic action sampling from the regret-matched strategy rather
    than argmax, which is required for Nash equilibrium strategies.
    """

    def __init__(
        self,
        name: str = "SDCFRBot",
        model: SDCFRModel | None = None,
        **model_kwargs: object,
    ) -> None:
        self._name = name
        self.model = model if model is not None else SDCFRModel(**model_kwargs)
        self._seat_cache: int | None = None
        self._last_state_id: int | None = None

    @property
    def name(self) -> str:
        """Return the bot's display name."""
        return self._name

    def act(self, state: GameState, legal: list[Action]) -> Action:
        """Choose an action by sampling from the SD-CFR strategy.

        Args:
            state: Current game state (from this bot's perspective).
            legal: List of legal actions available.

        Returns:
            Sampled action from the mixed Nash strategy.
        """
        seat = self._find_seat(state)
        obs = build_observation(state, seat)
        legal_mask = self._build_action_mask(state, legal, seat).astype(np.float32)
        strategy = self.model.get_strategy(obs, legal_mask)

        # Stochastic sampling — Nash strategies are mixed; never use argmax.
        action_idx = int(np.random.choice(7, p=strategy))

        # Clamp to legal action if sampled index is somehow illegal
        if legal_mask[action_idx] == 0:
            legal_indices = np.where(legal_mask)[0]
            action_idx = int(np.random.choice(legal_indices))

        return action_index_to_action(action_idx, state, seat)

    def observe_result(self, final_state: GameState, reward: float) -> None:
        """No-op: SD-CFR trains offline via traversals, not online feedback."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_seat(self, state: GameState) -> int:
        """Return this bot's seat number, using a cache for efficiency.

        Args:
            state: Current game state.

        Returns:
            Seat index (0-indexed).

        Raises:
            RuntimeError: If seat cannot be determined.
        """
        state_id = id(state)
        if self._last_state_id == state_id and self._seat_cache is not None:
            return self._seat_cache

        for i, player in enumerate(state.players):
            if len(player.hole_cards) > 0:
                self._seat_cache = i
                self._last_state_id = state_id
                return i

        raise RuntimeError("Cannot determine bot seat from game state")

    def _build_action_mask(
        self,
        state: GameState,
        legal: list[Action],
        seat: int,
    ) -> npt.NDArray[np.int32]:
        """Build a (7,) binary legal-action mask.

        Args:
            state: Current game state.
            legal: Legal actions for this seat.
            seat: The seat number.

        Returns:
            Binary mask of shape (7,).
        """
        mask = np.zeros(7, dtype=np.int32)
        for action in legal:
            idx = action_to_action_index(action, state, seat)
            mask[idx] = 1
        return mask
