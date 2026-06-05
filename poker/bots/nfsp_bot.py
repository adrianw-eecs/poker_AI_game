"""NFSP-based poker bot."""

from poker.bots.base_ml_bot import BaseMLBot
from poker.domain.action import Action
from poker.ml.action_space import action_index_to_action, action_to_action_index
from poker.ml.models.nfsp_model import NFSPModel
from poker.ml.observation import build_observation
from poker.state.game_state import GameState


class NFSPBot(BaseMLBot):
    """Poker bot powered by Neural Fictitious Self-Play.

    Mixes between an average (Nash) policy and a best-response policy
    during training. At inference time uses the average policy only.
    """

    def __init__(
        self,
        name: str = "NFSPBot",
        model: NFSPModel | None = None,
        training: bool = False,
        **model_kwargs,
    ) -> None:
        model = model if model is not None else NFSPModel(**model_kwargs)
        super().__init__(name=name, model=model)
        self.training = training

    def act(self, state: GameState, legal: list[Action]) -> Action:
        if not legal:
            raise ValueError("No legal actions available")

        seat = self._find_seat(state)
        obs = build_observation(state, seat)
        mask = self._build_action_mask(state, legal, seat)
        action_idx = self.model.select_action(obs, mask, training=self.training)
        return action_index_to_action(action_idx, state, seat)

    def observe_result(self, final_state: GameState, reward: float) -> None:
        pass
