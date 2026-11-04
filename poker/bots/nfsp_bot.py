"""NFSP-based poker bot."""

from poker.bots.base_ml_bot import BaseMLBot
from poker.bots.progress import ensure_progressing_action
from poker.domain.action import Action
from poker.ml.action_space import BetScheme, action_index_to_action, build_action_mask
from poker.ml.models.nfsp_model import NFSPModel
from poker.ml.observation import build_observation
from poker.ml.observation_v2 import build_observation_v2
from poker.ml.observation_v3 import build_observation_v3
from poker.ml.observation_v4 import build_observation_v4
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
        self._pending_episode = True

    def act(self, state: GameState, legal: list[Action]) -> Action:
        if not legal:
            raise ValueError("No legal actions available")

        if self.training and self._pending_episode:
            self.model.begin_episode()
            self._pending_episode = False

        seat = self._find_seat(state)
        version = int(getattr(self.model, "obs_version", 1))
        if version >= 4:
            obs = build_observation_v4(state, seat)
        elif version == 3:
            obs = build_observation_v3(state, seat)
        elif version == 2:
            obs = build_observation_v2(state, seat)
        else:
            obs = build_observation(state, seat)
        scheme = getattr(self.model, "bet_scheme", BetScheme.LEGACY)
        if not isinstance(scheme, BetScheme):
            scheme = BetScheme.LEGACY
        mask = build_action_mask(state, seat, scheme=scheme, dedup=True)
        action_idx = self.model.select_action(obs, mask, training=self.training)
        chosen = action_index_to_action(action_idx, state, seat, scheme=scheme)
        return ensure_progressing_action(state, chosen, legal)

    def observe_result(self, final_state: GameState, reward: float) -> None:
        self._pending_episode = True
