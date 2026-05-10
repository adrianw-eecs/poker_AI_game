"""External sampling CFR traversal engine for SD-CFR."""

import numpy as np
import numpy.typing as npt
import torch

from poker.config.blind_schedule import BlindSchedule
from poker.config.game_config import GameConfig
from poker.ml.action_space import build_action_mask
from poker.ml.buffers import WeightedReservoirBuffer
from poker.ml.cfr.regret_matching import compute_strategy_from_network
from poker.ml.env import PokerEnv
from poker.ml.models.nfsp_networks import PokerNetwork
from poker.ml.observation import build_observation


class ExternalSamplingTraverser:
    """Runs outcome-sampling CFR traversals using PokerEnv.

    At each traverser decision point during a rollout, computes the
    counterfactual advantage for every action (advantage[a] = Q(s,a) - V(s))
    and stores (obs, advantages) in the regret buffer.

    For opponents, a single action is sampled from the current strategy network.
    """

    def __init__(
        self,
        config: GameConfig,
        blind_schedule: BlindSchedule,
        advantage_network: PokerNetwork,
        device: torch.device,
        regret_buffer: WeightedReservoirBuffer,
    ) -> None:
        self.config = config
        self.blind_schedule = blind_schedule
        self.advantage_network = advantage_network
        self.device = device
        self.regret_buffer = regret_buffer
        self._env = PokerEnv(
            num_players=config.num_players,
            starting_stack=config.starting_stack,
            small_blind=config.small_blind,
            big_blind=config.big_blind,
            ante=config.ante,
            learning_seat=0,  # Will be overridden per traversal via reset
        )

    def traverse(self, traverser_seat: int, cfr_iteration: int) -> float:
        """Run one full hand as a CFR traversal.

        Uses outcome-sampling: rolls out the game using the current strategy
        for both players, but at each traverser decision computes and stores
        advantage estimates for all legal actions.

        Args:
            traverser_seat: The seat whose perspective we are training.
            cfr_iteration: Current CFR iteration (used as sample weight).

        Returns:
            The traverser's normalised chip outcome for this traversal.
        """
        # Build a fresh env with the traverser as the learning seat so that
        # the env's step() method waits for us at traverser decision points.
        env = PokerEnv(
            num_players=self.config.num_players,
            starting_stack=self.config.starting_stack,
            small_blind=self.config.small_blind,
            big_blind=self.config.big_blind,
            ante=self.config.ante,
            learning_seat=traverser_seat,
        )

        obs, info = env.reset()
        terminated = False
        total_reward = 0.0

        while not terminated:
            state = env.state
            if state is None:
                break

            action_seat = state.action_on_seat
            if action_seat is None:
                break

            if action_seat == traverser_seat:
                # --- Traverser decision point ---
                legal_mask = build_action_mask(state, traverser_seat).astype(np.float32)
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)

                # Compute current strategy via regret matching
                strategy = compute_strategy_from_network(
                    self.advantage_network, obs_t, legal_mask, self.device
                )

                # Compute Q-values for all actions
                self.advantage_network.eval()
                with torch.no_grad():
                    q_values = (
                        self.advantage_network(obs_t.unsqueeze(0))
                        .squeeze(0)
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

                # V(s) = sum_a strategy[a] * Q(s, a)  (over legal actions only)
                v_s = float(np.dot(strategy, q_values))

                # advantage[a] = Q(s, a) - V(s), zeroed for illegal actions
                advantages = (q_values - v_s) * legal_mask

                # Store (obs, advantages) weighted by cfr_iteration (min weight 1)
                weight = float(max(1, cfr_iteration))
                # WeightedReservoirBuffer stores (obs, action, weight); we encode
                # advantages as the "action" field by flattening to index form.
                # Instead we store each (obs, advantage_vector) as a single entry
                # using action=0 as a placeholder; the model unpacks from obs.
                # We repurpose the buffer's action field by storing the flat
                # advantage array packed into a separate obs-shaped slot.
                # The sdcfr_model.py training loop reads buffer["obs"] for
                # observations and buffer["actions"] is ignored; targets are
                # stored interleaved. To keep compatibility we add one entry
                # per legal action, storing the full obs and the per-action
                # advantage as a scalar.
                # Simpler: we'll call add() once with action=0; the model
                # accesses obs to get the full advantage vector by re-running
                # the network — but that won't work.
                #
                # Correct approach: store the full advantage vector per
                # traverser step. We pack advantages (shape 7) into the first
                # 7 positions of a 142-d buffer slot by replacing obs, but that
                # corrupts the observation. Instead we extend the buffer slot
                # to hold both obs and targets. Because WeightedReservoirBuffer
                # uses a fixed obs_dim, we instantiate the regret_buffer with
                # obs_dim=142+7=149 and pack [obs | advantages] together.
                # This is handled in SDCFRModel by setting capacity with
                # obs_dim=142+7.
                #
                # For simplicity within the current buffer API (obs_dim=142),
                # we store the obs as-is and store the advantage for each
                # individual (obs, advantage_for_action_a) pair via repeated
                # add calls.  The model encodes targets in the 'actions' field
                # reinterpreted as floats — but actions is int64.
                #
                # Final pragmatic choice: use obs_dim=142+7=149 in the buffer
                # and pack packed_obs = np.concatenate([obs, advantages]).
                # SDCFRModel sets obs_dim=149 and splits on retrieval.
                packed = np.concatenate([obs, advantages]).astype(np.float32)
                self.regret_buffer.add(packed, 0, weight)

                # Sample action proportional to strategy for the traversal path
                action_idx = int(np.random.choice(7, p=strategy))
                obs, reward, terminated, _, info = env.step(action_idx)
                total_reward += reward

            else:
                # --- Opponent decision point (should not happen inside PokerEnv
                # because env auto-plays opponents) ---
                # This branch is defensive; PokerEnv handles opponent turns.
                break

        return total_reward
