"""External-sampling MCCFR traversal using PokerEnv."""

import numpy as np
import numpy.typing as npt
import torch

from poker.bots.random_bot import RandomBot
from poker.config.game_config import GameConfig
from poker.ml.action_space import BetScheme, build_action_mask
from poker.ml.buffers import WeightedCircularBuffer
from poker.ml.cfr.regret_matching import compute_strategy_from_network
from poker.ml.env import PokerEnv
from poker.ml.models.nfsp_networks import PokerNetwork

# Minimum action probability to avoid division by zero in importance weights.
_MIN_PROB = 1e-6
# Cap on importance weight to prevent gradient explosion from rare actions.
_MAX_IS_WEIGHT = 10.0


class ExternalSamplingTraverser:
    """Outcome-sampling MCCFR traversals using PokerEnv.

    Algorithm (per traversal):
    1. Run a full hand with the traverser at learning_seat. Opponents are
       provided by the caller — any mix of bots (RandomBot, FlopBot, SDCFRBot,
       etc.) is supported.
    2. At each traverser decision point, sample one action from the current
       strategy network. Record (obs, sampled_action, strategy_prob).
    3. At the terminal state, back-fill advantage targets using outcome
       sampling with importance weighting:
           advantages[sampled_action] = clip(terminal_reward / strategy_prob)
           advantages[unsampled actions] = 0
    4. Store each (obs ‖ advantages) vector in the regret buffer with weight
       = cfr_iteration so that later iterations contribute more.

    Over many traversals this gives unbiased advantage estimates that the
    network is trained to predict. Regret matching on those predictions then
    produces the strategy.
    """

    def __init__(
        self,
        config: GameConfig,
        advantage_network: PokerNetwork,
        device: torch.device,
        regret_buffer: WeightedCircularBuffer,
        obs_version: int = 1,
        bet_scheme: BetScheme = BetScheme.LEGACY,
    ) -> None:
        if obs_version not in (1, 2):
            raise ValueError(f"obs_version must be 1 or 2, got {obs_version}")
        self.config = config
        self.advantage_network = advantage_network
        self.device = device
        self.regret_buffer = regret_buffer
        self.obs_version = obs_version
        self.bet_scheme = bet_scheme

    def pack(
        self,
        obs: npt.NDArray[np.float32],
        advantages: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Pack ``[obs | advantages]`` into one buffer row.

        The obs width depends on the encoder (98 for v1, 318 for v2), so this
        cannot be a module-level constant. A mismatched width here is what the
        regret buffer would otherwise reject at sample time.
        """
        row = np.concatenate([obs, advantages]).astype(np.float32)
        expected = self.regret_buffer.obs_dim
        if row.shape[0] != expected:
            raise ValueError(
                f"packed row is {row.shape[0]} wide but the regret buffer expects "
                f"{expected}. Traverser obs_version={self.obs_version} "
                f"(obs {obs.shape[0]} + 7 actions)."
            )
        return row

    def traverse(
        self,
        traverser_seat: int,
        cfr_iteration: int,
        opponent_bots: list | None = None,
        num_players: int | None = None,
    ) -> float:
        """Run one traversal and store advantage samples in the regret buffer.

        Args:
            traverser_seat: Seat index being trained this iteration.
            cfr_iteration: Current CFR iteration (used as sample weight ≥ 1).
            opponent_bots: Bots for the non-traverser seats. Length must equal
                num_players - 1. Defaults to RandomBot for all opponents.
            num_players: Override config.num_players for this traversal.
                Allows variable player-count training from the same traverser.

        Returns:
            Terminal reward for the traverser (0.0 if traverser never acted).
        """
        n = num_players if num_players is not None else self.config.num_players
        # Clamp seat to valid range (caller may pass a fixed rotation index)
        traverser_seat = traverser_seat % n
        num_opponents = n - 1

        if opponent_bots is None:
            opponent_bots = [RandomBot() for _ in range(num_opponents)]

        env = PokerEnv(
            num_players=n,
            starting_stack=self.config.starting_stack,
            small_blind=self.config.small_blind,
            big_blind=self.config.big_blind,
            ante=self.config.ante,
            learning_seat=traverser_seat,
            opponent_bots=opponent_bots,
            obs_version=self.obs_version,
            bet_scheme=self.bet_scheme,
        )

        obs, _ = env.reset()

        # Hand may be over before the traverser acts (e.g. everyone folds
        # before reaching their seat in the preflop order).
        if env.state is None or env.state.action_on_seat != traverser_seat:
            return 0.0

        # (obs, sampled_action, strategy_prob_for_that_action, legal_mask)
        decisions: list[tuple[npt.NDArray[np.float32], int, float, npt.NDArray[np.float32]]] = []
        terminated = False
        terminal_reward = 0.0

        while not terminated:
            if env.state is None or env.state.action_on_seat != traverser_seat:
                break

            legal_mask = build_action_mask(
                env.state, traverser_seat, scheme=self.bet_scheme
            ).astype(np.float32)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            strategy = compute_strategy_from_network(
                self.advantage_network, obs_tensor, legal_mask, self.device
            )

            action = int(np.random.choice(7, p=strategy))
            decisions.append((obs.copy(), action, float(strategy[action]), legal_mask))

            obs, reward, terminated, _, _ = env.step(action)
            if terminated:
                terminal_reward = reward

        if not decisions:
            return terminal_reward

        # Back-fill outcome-sampling advantage targets for each recorded decision.
        # Only the taken action gets a non-zero estimate (importance weighted);
        # unsampled actions receive 0, which the per-batch normalisation in
        # train_iteration() will treat as the mean baseline.
        # sqrt weighting slows unbounded growth: linear weights reach 5000x by
        # iteration 5000; sqrt stays below 71x, keeping early data in budget (Fix #11).
        weight = float(np.sqrt(max(1, cfr_iteration)))
        for obs_i, action_i, prob_i, mask_i in decisions:
            advantages = np.zeros(7, dtype=np.float32)
            is_weight = min(1.0 / max(prob_i, _MIN_PROB), _MAX_IS_WEIGHT)
            advantages[action_i] = terminal_reward * is_weight
            packed = self.pack(obs_i, advantages)
            self.regret_buffer.add(packed, 0, weight)

        return terminal_reward
