"""Neural Fictitious Self-Play (NFSP) model."""

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim

from poker.ml.action_space import BetScheme
from poker.ml.buffers import CircularBuffer, ReservoirBuffer, WeightedCircularBuffer
from poker.ml.cfr.regret_matching import regret_matching
from poker.ml.observation import adapt_observation
from poker.ml.observation_v2 import OBS_V2_DIM
from poker.ml.models.nfsp_networks import (
    build_hybrid_network_v2,
    build_policy_network,
    build_policy_network_v2,
    build_q_network,
    build_q_network_v2,
    make_target_network,
    network_kwargs_from_state_dict,
    sync_target_network,
)

# Maximum entries persisted per buffer when saving a checkpoint.
# Avoids multi-GB checkpoints for large buffers while still warm-starting the
# next training run with meaningful experience.
_MAX_CHECKPOINT_BUFFER = 100_000


# Paper / Steinberger NFSP: clip both the best-response and average-policy grads.
_GRAD_CLIP = 10.0
# Illegal-action logit fill. -inf breaks log_softmax when a row is fully masked.
_ILLEGAL_LOGIT = -1.0e9


def _checkpoint_arch(state: dict) -> str:
    if "rank_emb.weight" in state:
        return "v2"
    if "input_proj.0.weight" in state:
        return "v1"
    raise ValueError("Checkpoint network is neither PokerNetwork nor PokerNetV2")


def _checkpoint_actions(state: dict) -> int:
    for key in ("adv_head.weight", "policy_head.weight", "head.weight"):
        if key in state:
            return int(state[key].shape[0])
    raise ValueError("Checkpoint has no action head")


def _require_matching_checkpoint(q_state: dict, policy_state: dict, meta: dict) -> str:
    """Reject a checkpoint whose metadata and weights describe different models."""
    q_arch = _checkpoint_arch(q_state)
    policy_arch = _checkpoint_arch(policy_state)
    if q_arch != policy_arch:
        raise ValueError(
            f"Q network is {q_arch} but the policy network is {policy_arch}"
        )
    q_actions = _checkpoint_actions(q_state)
    policy_actions = _checkpoint_actions(policy_state)
    if q_actions != policy_actions:
        raise ValueError(
            f"Q head has {q_actions} actions but the policy head has {policy_actions}"
        )
    if meta:
        version = int(meta["obs_version"])
        if (version == 2 and q_arch != "v2") or (version == 1 and q_arch != "v1"):
            raise ValueError(
                f"Checkpoint metadata obs_version={version} does not match {q_arch} weights"
            )
        label = meta.get("arch")
        if label == "pokernet_v2" and q_arch != "v2":
            raise ValueError("Checkpoint metadata arch=pokernet_v2 does not match the weights")
        if label == "pokernet_v1" and q_arch != "v1":
            raise ValueError("Checkpoint metadata arch=pokernet_v1 does not match the weights")
        if int(meta.get("num_actions", q_actions)) != q_actions:
            raise ValueError(
                f"Checkpoint metadata num_actions={meta.get('num_actions')} "
                f"does not match the action head ({q_actions})"
            )
    if q_arch == "v2":
        width = int(network_kwargs_from_state_dict(q_state)["input_dim"])
        if width != OBS_V2_DIM:
            raise ValueError(
                f"PokerNetV2 checkpoint input width {width} does not match obs v2 ({OBS_V2_DIM})"
            )
    return q_arch


class NFSPModel:
    """Neural Fictitious Self-Play agent.

    One agent is two networks and two memories (Heinrich & Silver 2016):
    - Q-network: Double DQN on M_RL, a circular buffer of every transition.
      Best response is epsilon-greedy over legal actions.
    - Average-policy network: cross-entropy on M_SL, a reservoir of
      best-response actions only, with a legal-action mask.

    Anticipatory dynamics pick one mode per episode: best response with
    probability eta (0.1), otherwise the average policy. Evaluation and
    average-policy actions are sampled from the policy network.
    """

    def __init__(
        self,
        eta: float = 0.1,
        epsilon_start: float = 0.10,
        epsilon_end: float = 0.005,
        epsilon_decay_steps: int = 300_000,
        gamma: float | None = None,
        batch_size: int = 4096,
        lr: float = 0.0005,
        q_lr: float | None = None,
        policy_lr: float | None = None,
        q_capacity: int = 1_000_000,
        policy_capacity: int = 500_000,
        target_sync_every: int = 300,
        train_every: int = 32,
        updates_per_train: int = 1,
        device: str | None = None,
        obs_version: int = 1,
        bet_scheme: BetScheme | None = None,
        epsilon_schedule: str = "linear",
        q_loss: str = "auto",
        shared_trunk: bool = False,
        distill_kappa: float = 0.0,
        distill_capacity: int = 200_000,
    ) -> None:
        if obs_version not in (1, 2):
            raise ValueError(f"obs_version must be 1 or 2, got {obs_version}")
        if shared_trunk and obs_version != 2:
            raise ValueError("shared_trunk requires obs_version=2")
        if not 0.0 <= distill_kappa <= 1.0:
            raise ValueError(f"distill_kappa must be in [0, 1], got {distill_kappa}")
        if epsilon_schedule not in ("linear", "inv_sqrt"):
            raise ValueError(f"epsilon_schedule must be linear or inv_sqrt, got {epsilon_schedule}")
        if updates_per_train < 1:
            raise ValueError(f"updates_per_train must be >= 1, got {updates_per_train}")
        if gamma is None:
            gamma = 1.0 if obs_version == 2 else 0.99
        if q_lr is None:
            q_lr = lr
        if policy_lr is None:
            policy_lr = lr
        if q_loss == "auto":
            q_loss = "huber" if obs_version == 2 else "mse"
        if q_loss not in ("mse", "huber"):
            raise ValueError(f"q_loss must be mse or huber, got {q_loss}")
        if bet_scheme is None:
            bet_scheme = BetScheme.POT_FRACTION if obs_version == 2 else BetScheme.LEGACY
        elif not isinstance(bet_scheme, BetScheme):
            bet_scheme = BetScheme(bet_scheme)

        self.eta = eta
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_schedule = epsilon_schedule
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync_every = target_sync_every
        self.train_every = train_every
        self.updates_per_train = updates_per_train
        self.obs_version = obs_version
        self.bet_scheme = bet_scheme
        self.num_actions = 7
        self.q_loss_name = q_loss
        self.q_lr = q_lr
        self.policy_lr = policy_lr
        self.force_best_response = False
        # Hybrid (SD-CFR-as-teacher) settings. kappa weighs the regret-matching
        # distillation term against the NFSP reservoir term in the policy loss;
        # kappa=0 reproduces plain NFSP exactly.
        self.distill_kappa = float(distill_kappa)
        self._distill_buffer: WeightedCircularBuffer | None = None

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        obs_dim = 98 if obs_version == 1 else OBS_V2_DIM
        self._obs_dtype = np.dtype(np.float16 if obs_version == 2 else np.float32)
        if obs_version == 1:
            self.q_network = build_q_network().to(self.device)
            self.policy_network = build_policy_network().to(self.device)
        elif shared_trunk:
            # One network, two heads: the policy head and the dueling Q head
            # share card embeddings and the residual trunk.
            hybrid = build_hybrid_network_v2().to(self.device)
            self.q_network = hybrid
            self.policy_network = hybrid
        else:
            self.q_network = build_q_network_v2().to(self.device)
            self.policy_network = build_policy_network_v2().to(self.device)
        self.shared_trunk = bool(shared_trunk and obs_version == 2)
        # A shared module cannot be deep-copied independently of its optimizer
        # wiring, but a frozen copy is exactly what the target net needs.
        self.target_network = make_target_network(self.q_network).to(self.device)

        self.q_buffer = CircularBuffer(capacity=q_capacity, obs_dim=obs_dim, dtype=self._obs_dtype)
        self.policy_buffer = ReservoirBuffer(
            capacity=policy_capacity, obs_dim=obs_dim, dtype=self._obs_dtype
        )

        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=policy_lr)

        # Distillation memory: one row per (obs, 7 advantages, 7 mask) from an
        # SD-CFR traversal, packed as [obs | advantages | mask] so a single
        # WeightedCircularBuffer carries it.
        if self.distill_kappa > 0.0 or self.shared_trunk:
            self._ensure_distill_buffer(distill_capacity)

        self._step = 0
        self._q_updates = 0
        self._p_updates = 0
        # None until begin_episode(). Held for the whole hand.
        self._episode_br: bool | None = None
        self._episode_br_by_seat: dict[int, bool] = {}
        # Warm-up threshold: require at least one full batch worth of distinct
        # episodes before training begins. Minimum 1% of buffer capacity so
        # early samples don't monopolize training.
        self._q_min = max(batch_size, q_capacity // 100)
        self._base_q_lr = q_lr

        self._config = {
            "eta": eta,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay_steps": epsilon_decay_steps,
            "epsilon_schedule": epsilon_schedule,
            "gamma": gamma,
            "batch_size": batch_size,
            "lr": lr,
            "q_lr": q_lr,
            "policy_lr": policy_lr,
            "q_capacity": q_capacity,
            "policy_capacity": policy_capacity,
            "target_sync_every": target_sync_every,
            "train_every": train_every,
            "updates_per_train": updates_per_train,
            "obs_version": obs_version,
            "bet_scheme": bet_scheme.value,
            "q_loss": q_loss,
        }

    def begin_episode(self, seat: int | None = None) -> None:
        """Sample anticipatory mode once for the upcoming episode.

        Steinberger's SeatActor._pick_training_policy runs on episode start,
        not on every decision. Best-response mode also writes (s, a) into the
        reservoir, including epsilon-random exploration actions.

        Pass ``seat`` when several seats share this model. Each seat keeps
        the mode sampled at the start of its hand.
        """
        if self.force_best_response:
            br = True
        else:
            br = bool(np.random.random() < self.eta)
        if seat is None:
            self._episode_br = br
        else:
            self._episode_br_by_seat[int(seat)] = br

    def _episode_is_br(self, seat: int | None) -> bool:
        if seat is None:
            if self._episode_br is None:
                self.begin_episode()
            return bool(self._episode_br)
        mode = self._episode_br_by_seat.get(int(seat))
        if mode is None:
            self.begin_episode(seat)
            mode = bool(self._episode_br_by_seat[int(seat)])
        return mode

    def _current_epsilon(self) -> float:
        span = self.epsilon_start - self.epsilon_end
        if self.epsilon_schedule == "inv_sqrt":
            scale = 1.0 / np.sqrt(1.0 + self._step / max(self.epsilon_decay_steps, 1))
            return float(self.epsilon_end + span * scale)
        decay = span * min(1.0, self._step / self.epsilon_decay_steps)
        return max(self.epsilon_end, self.epsilon_start - decay)

    def _apply_mask(self, logits: torch.Tensor, legal_mask: npt.NDArray[np.int32]) -> torch.Tensor:
        mask_t = torch.as_tensor(legal_mask, dtype=torch.bool, device=self.device)
        return logits.masked_fill(~mask_t, float("-inf"))

    def _prepare_obs(self, obs: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        arr = np.asarray(obs, dtype=np.float32)
        expected = int(self.q_network.input_dim)
        if arr.shape[-1] != expected:
            # obs v1: adapt 98-dim legacy → 155-dim, or raise if mismatch
            if self.obs_version == 1:
                arr = adapt_observation(arr, expected)
            # obs v2: observations should already match (318-dim), no adaptation needed
            else:
                raise ValueError(
                    f"Observation width {arr.shape[-1]} does not match "
                    f"obs v{self.obs_version} network input {expected}. "
                    f"v2 expects {expected} but got {arr.shape[-1]}. "
                    f"This usually means loading a v1 checkpoint for v2 training."
                )
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _masked_q(
        self,
        network: torch.nn.Module,
        obs_t: torch.Tensor,
        mask_t: torch.Tensor,
    ) -> torch.Tensor:
        """Q-values with illegal actions removed. V2 averages advantages over legal actions."""
        if self.obs_version == 2:
            q_vals = network(obs_t, legal_mask=mask_t)
        else:
            q_vals = network(obs_t)
        return q_vals.masked_fill(~mask_t, _ILLEGAL_LOGIT)

    def select_action(
        self,
        obs: npt.NDArray[np.float32],
        legal_mask: npt.NDArray[np.int32],
        training: bool = True,
        seat: int | None = None,
    ) -> int:
        obs = self._prepare_obs(obs)
        if training and (
            self.q_buffer.obs_dim != obs.shape[0] or self.policy_buffer.obs_dim != obs.shape[0]
        ):
            self._ensure_buffer_width(int(obs.shape[0]))
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        if not training:
            self.policy_network.eval()
            with torch.no_grad():
                logits = self._apply_mask(self.policy_network(obs_t).squeeze(0), legal_mask)
                probs = torch.softmax(logits, dim=0)
            return int(torch.multinomial(probs, 1).item())

        use_best_response = self._episode_is_br(seat)

        if use_best_response:
            epsilon = self._current_epsilon()
            if np.random.random() < epsilon:
                legal_indices = np.where(legal_mask)[0]
                action = int(np.random.choice(legal_indices))
            else:
                self.q_network.eval()
                with torch.no_grad():
                    mask_t = torch.as_tensor(
                        legal_mask, dtype=torch.bool, device=self.device
                    ).unsqueeze(0)
                    q_vals = self._masked_q(self.q_network, obs_t, mask_t).squeeze(0)
                action = int(q_vals.argmax().item())
            self.policy_buffer.add(obs, action, legal_mask=np.asarray(legal_mask, dtype=np.float32))
            return action
        else:
            self.policy_network.eval()
            with torch.no_grad():
                logits = self._apply_mask(self.policy_network(obs_t).squeeze(0), legal_mask)
                probs = torch.softmax(logits, dim=0)
            return int(torch.multinomial(probs, 1).item())

    def store_transition(
        self,
        obs: npt.NDArray[np.float32],
        action: int,
        reward: float,
        next_obs: npt.NDArray[np.float32],
        done: bool,
        next_mask: npt.NDArray[np.float32] | None = None,
        current_mask: npt.NDArray[np.float32] | None = None,
        seat: int | None = None,
    ) -> None:
        """Store a transition into the Q memory (M_RL).

        NFSP feeds M_SL only from best-response behaviour, and
        ``select_action`` already writes those ``(s, a)`` pairs. This method
        used to add every transition to the reservoir as well, which doubled
        each best-response sample and, worse, copied pure average-policy
        actions in, so the supervised loss trained the average policy to
        imitate itself instead of the best response.
        """
        obs = self._prepare_obs(obs)
        next_obs = self._prepare_obs(next_obs)
        if self.q_buffer.obs_dim != obs.shape[0]:
            self._ensure_buffer_width(int(obs.shape[0]))
        self.q_buffer.add(
            obs, action, reward, next_obs, done, next_mask=next_mask, current_mask=current_mask
        )

    def train_step(self) -> dict[str, float | None]:
        self._step += 1

        if self._step % self.train_every != 0:
            return {"q_loss": None, "policy_loss": None}

        # The two memories train independently. Gating Q on the reservoir
        # delays the best response until enough BR actions have been stored.
        q_loss = None
        policy_loss = None
        for _ in range(self.updates_per_train):
            if len(self.q_buffer) >= self._q_min:
                q_loss = self._update_q()
            if len(self.policy_buffer) >= self.batch_size:
                policy_loss = self._update_policy()
                # SD-CFR-as-teacher: an additional cross-entropy step toward
                # regret-matching targets, weighted by kappa. Kept as a separate
                # optimiser step so the reservoir loss is never diluted.
                if self.distill_kappa > 0.0:
                    self._distillation_step()
        return {"q_loss": q_loss, "policy_loss": policy_loss}

    def _update_q(self) -> float:
        """One Double DQN step. Targets are computed in eval mode (no dropout)."""
        batch = self.q_buffer.sample(self.batch_size)
        obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        next_mask_t = torch.as_tensor(batch["next_masks"], dtype=torch.bool, device=self.device)
        current_masks_t = torch.as_tensor(batch["current_masks"], dtype=torch.bool, device=self.device)
        dones_t = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        self.q_network.eval()
        self.target_network.eval()
        with torch.no_grad():
            # Double DQN: online net selects the greedy next action (masked),
            # target net evaluates it.
            next_q_online = self._masked_q(self.q_network, next_obs_t, next_mask_t)
            next_actions = next_q_online.argmax(dim=1)
            if self.obs_version == 2:
                next_q_target = self.target_network(next_obs_t, legal_mask=next_mask_t)
            else:
                next_q_target = self.target_network(next_obs_t)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            next_q = torch.where(next_mask_t.any(dim=1), next_q, torch.zeros_like(next_q))
            td_targets = rewards_t + self.gamma * next_q * (1.0 - dones_t)
            td_targets = torch.nan_to_num(td_targets, nan=0.0, posinf=0.0, neginf=0.0)

            # Fold-return target for action 0 (fold)
            fold_legal = current_masks_t[:, 0]
            if self.obs_version == 2:
                committed_frac = obs_t[:, 48]
                fold_return = -0.5 * committed_frac
            else:
                fold_return = torch.zeros_like(fold_legal, dtype=torch.float32, device=self.device)

        self.q_network.train()
        # PokerNetwork (v1) takes no legal_mask argument; PokerNetV2 does.
        # _masked_q routes per architecture so both shapes stay legal.
        q_vals = self._masked_q(self.q_network, obs_t, current_masks_t)
        predicted_q = q_vals.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        if self.q_loss_name == "huber":
            q_loss = nn.functional.smooth_l1_loss(predicted_q, td_targets)
        else:
            q_loss = nn.functional.mse_loss(predicted_q, td_targets)

        total_loss = q_loss
        # The fold-return target reads the v2 committed-fraction feature, so it
        # only carries information under obs v2. Under v1 fold_return is a
        # constant zero, which would add a pure shrinkage term on the fold Q
        # value and compete with the TD signal.
        if self.obs_version == 2:
            fold_q = q_vals[:, 0]
            fold_loss = nn.functional.mse_loss(fold_q, fold_return, reduction="none")
            fold_loss = torch.where(fold_legal, fold_loss, torch.zeros_like(fold_loss))
            total_loss = total_loss + fold_loss.mean()

        self.q_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=_GRAD_CLIP)
        self.q_optimizer.step()

        self._q_updates += 1
        if self._q_updates % self.target_sync_every == 0:
            sync_target_network(self.q_network, self.target_network)
        return float(total_loss.item())

    def _ensure_distill_buffer(self, capacity: int) -> None:
        """(Re)create the distillation memory at the current obs width."""
        want = int(self.q_network.input_dim) + 2 * self.num_actions
        if self._distill_buffer is not None and self._distill_buffer.obs_dim == want:
            return
        self._distill_buffer = WeightedCircularBuffer(capacity=capacity, obs_dim=want)

    def add_regret_sample(
        self,
        obs: npt.NDArray[np.float32],
        advantages: npt.NDArray[np.float32],
        legal_mask: npt.NDArray[np.float32],
        weight: float = 1.0,
    ) -> None:
        """Store one SD-CFR traversal target for policy distillation.

        ``advantages`` are the counterfactual advantages for the 7 actions and
        ``legal_mask`` marks which were legal. Regret matching is applied at
        sample time, so the stored row stays interpretable.
        """
        if self._distill_buffer is None:
            self._ensure_distill_buffer(200_000)
        row = np.concatenate(
            [
                np.asarray(obs, dtype=np.float32),
                np.asarray(advantages, dtype=np.float32),
                np.asarray(legal_mask, dtype=np.float32),
            ]
        )
        assert self._distill_buffer is not None
        if row.shape[0] != self._distill_buffer.obs_dim:
            raise ValueError(
                f"regret row is {row.shape[0]} wide, buffer expects "
                f"{self._distill_buffer.obs_dim}"
            )
        self._distill_buffer.add(row, 0, float(weight))

    def _distillation_step(self) -> float | None:
        """Cross-entropy from the policy head to regret-matching targets."""
        buf = self._distill_buffer
        if buf is None or len(buf) < min(self.batch_size, 64):
            return None
        obs_dim = int(self.q_network.input_dim)
        batch = buf.sample(self.batch_size)
        packed = batch["obs"]
        obs_batch = packed[:, :obs_dim]
        adv_batch = packed[:, obs_dim : obs_dim + self.num_actions]
        mask_batch = packed[:, obs_dim + self.num_actions :]

        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(mask_batch, dtype=torch.bool, device=self.device)
        # Regret matching must run on CPU: it is a small normalisation and the
        # target is not differentiable.
        targets = np.stack(
            [
                regret_matching(adv_batch[i], mask_batch[i])
                for i in range(adv_batch.shape[0])
            ]
        ).astype(np.float32)
        target_t = torch.as_tensor(targets, dtype=torch.float32, device=self.device)

        self.policy_network.train()
        logits = self.policy_network(obs_t)
        logits = logits.masked_fill(~mask_t, _ILLEGAL_LOGIT)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(target_t * log_probs).sum(dim=-1).mean()
        self.policy_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=_GRAD_CLIP)
        self.policy_optimizer.step()
        self._p_updates += 1
        return float(loss.item())

    def _update_policy(self) -> float:
        """Cross-entropy on reservoir best-response actions, masked to legal actions."""
        self.policy_network.train()
        pbatch = self.policy_buffer.sample(self.batch_size)
        pobs_t = torch.as_tensor(pbatch["obs"], dtype=torch.float32, device=self.device)
        plabels_t = torch.as_tensor(pbatch["actions"], dtype=torch.int64, device=self.device)
        masks_t = torch.as_tensor(pbatch["masks"], dtype=torch.bool, device=self.device)

        logits = self.policy_network(pobs_t).masked_fill(~masks_t, _ILLEGAL_LOGIT)
        policy_loss = nn.functional.cross_entropy(logits, plabels_t)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=_GRAD_CLIP)
        self.policy_optimizer.step()
        self._p_updates += 1
        return float(policy_loss.item())

    def trim_q_buffer(self, keep_fraction: float = 0.75) -> int:
        """Drop oldest Q-buffer entries after a distribution shift (e.g. SP ramp)."""
        return self.q_buffer.keep_recent_fraction(keep_fraction, min_size=self.batch_size)

    def set_q_learning_rate(self, lr: float) -> None:
        """Adjust Q-network learning rate (e.g. lower at high self-play)."""
        for group in self.q_optimizer.param_groups:
            group["lr"] = lr

    def save(self, filepath: str, training_meta: dict | None = None) -> None:
        """Save model weights, optimizer states, and a capped buffer snapshot."""
        q_state = self.q_buffer.state_dict()
        p_state = self.policy_buffer.state_dict()

        # Cap buffer snapshots to avoid multi-GB checkpoint files.
        # Q: keep the newest rows and reset the write cursor (see CircularBuffer).
        # Policy: subsample, but keep total_seen so the reservoir stays a
        # uniform sample of the whole stream instead of a sliding window.
        if q_state["size"] > _MAX_CHECKPOINT_BUFFER:
            q_state = self.q_buffer.recent_state_dict(_MAX_CHECKPOINT_BUFFER)
        if p_state["size"] > _MAX_CHECKPOINT_BUFFER:
            n = _MAX_CHECKPOINT_BUFFER
            idx = np.random.choice(p_state["size"], n, replace=False)
            for k in ("obs", "actions", "masks"):
                p_state[k] = p_state[k][idx]
            p_state["size"] = n

        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "policy_network": self.policy_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
                "step": self._step,
                "config": self._config,
                "model_meta": {
                    "obs_version": self.obs_version,
                    "bet_scheme": self.bet_scheme.value,
                    "num_actions": self.num_actions,
                    "arch": "pokernet_v2" if self.obs_version == 2 else "pokernet_v1",
                    "shared_trunk": self.shared_trunk,
                    "distill_kappa": self.distill_kappa,
                },
                "q_buffer": q_state,
                "policy_buffer": p_state,
                "training_meta": training_meta or {},
            },
            filepath,
        )

    def _rebuild_networks(self, q_state: dict, policy_state: dict) -> None:
        """Replace trunks so checkpoint shapes (width, depth, input) match."""
        q_kwargs = network_kwargs_from_state_dict(q_state)
        policy_kwargs = network_kwargs_from_state_dict(policy_state)
        q_arch = q_kwargs.pop("arch")
        policy_kwargs.pop("arch", None)
        q_lr = float(self._config.get("q_lr", self._config.get("lr", self.q_lr)))
        policy_lr = float(self._config.get("policy_lr", self._config.get("lr", self.policy_lr)))

        # A shared-trunk checkpoint stores ONE module's state under both keys and
        # carries a policy_head, which a plain PokerNetV2 would reject.
        hybrid_state = "policy_head.weight" in q_state
        if hybrid_state:
            self.shared_trunk = True
            hybrid = build_hybrid_network_v2(**q_kwargs).to(self.device)
            self.q_network = hybrid
            self.policy_network = hybrid
            self.target_network = make_target_network(hybrid).to(self.device)
            self.q_optimizer = optim.Adam(hybrid.parameters(), lr=q_lr)
            self.policy_optimizer = optim.Adam(hybrid.parameters(), lr=policy_lr)
        elif q_arch == "v2":
            self.q_network = build_q_network_v2(**q_kwargs).to(self.device)
            self.policy_network = build_policy_network_v2(**policy_kwargs).to(self.device)
            self.target_network = make_target_network(self.q_network).to(self.device)
            self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=q_lr)
            self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=policy_lr)
        else:
            self.q_network = build_q_network(**q_kwargs).to(self.device)
            self.policy_network = build_policy_network(**policy_kwargs).to(self.device)
            self.target_network = make_target_network(self.q_network).to(self.device)
            self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=q_lr)
            self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=policy_lr)
        self.num_actions = int(q_kwargs["num_actions"])
        self.q_lr = q_lr
        self.policy_lr = policy_lr

    def _ensure_buffer_width(self, obs_dim: int) -> None:
        dtype = self._obs_dtype
        if self.q_buffer.obs_dim != obs_dim or self.q_buffer.dtype != dtype:
            self.q_buffer = CircularBuffer(
                capacity=self.q_buffer.capacity, obs_dim=obs_dim, dtype=dtype
            )
        if self.policy_buffer.obs_dim != obs_dim or self.policy_buffer.dtype != dtype:
            self.policy_buffer = ReservoirBuffer(
                capacity=self.policy_buffer.capacity, obs_dim=obs_dim, dtype=dtype
            )

    def _ensure_buffer_capacity(self, q_capacity: int | None, policy_capacity: int | None) -> None:
        """Grow buffers so a checkpoint's stored rows fit.

        ``_ensure_buffer_width`` keeps the *current* capacity, so loading a
        100k-row snapshot into a model built with smaller buffers raised a
        broadcast error. A saved snapshot is never larger than the capacity it
        was written from, so the checkpoint is authoritative here.
        """
        dtype = self._obs_dtype
        if q_capacity is not None and q_capacity > self.q_buffer.capacity:
            self.q_buffer = CircularBuffer(
                capacity=q_capacity, obs_dim=self.q_buffer.obs_dim, dtype=dtype
            )
        if policy_capacity is not None and policy_capacity > self.policy_buffer.capacity:
            self.policy_buffer = ReservoirBuffer(
                capacity=policy_capacity, obs_dim=self.policy_buffer.obs_dim, dtype=dtype
            )

    def load(self, filepath: str) -> dict:
        # weights_only=False required because checkpoints include numpy arrays
        # (buffer state_dict). Only load checkpoints from trusted sources.
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        q_state = checkpoint["q_network"]
        policy_state = checkpoint["policy_network"]
        meta = dict(checkpoint.get("model_meta") or {})
        arch = _require_matching_checkpoint(q_state, policy_state, meta)
        self._config = dict(checkpoint.get("config") or {})
        self._config.setdefault("q_lr", self._config.get("lr", self.q_lr))
        self._config.setdefault("policy_lr", self._config.get("lr", self.policy_lr))
        self.obs_version = 2 if arch == "v2" else 1
        if meta.get("bet_scheme"):
            self.bet_scheme = BetScheme(meta["bet_scheme"])
        elif self._config.get("bet_scheme"):
            self.bet_scheme = BetScheme(self._config["bet_scheme"])
        else:
            self.bet_scheme = BetScheme.POT_FRACTION if arch == "v2" else BetScheme.LEGACY
        self._config["obs_version"] = self.obs_version
        self._config["bet_scheme"] = self.bet_scheme.value
        self._obs_dtype = np.dtype(np.float16 if self.obs_version == 2 else np.float32)
        # Restore hybrid settings from metadata. shared_trunk decides whether one
        # module backs both heads; kappa is a training-time weight only.
        self.distill_kappa = float(meta.get("distill_kappa", self.distill_kappa))
        if meta.get("shared_trunk"):
            self.shared_trunk = True
        self._rebuild_networks(q_state, policy_state)
        if self.shared_trunk:
            # One module, one state dict: loading it twice would be redundant
            # and the two saved copies are identical.
            self.q_network.load_state_dict(q_state)
            self.target_network.load_state_dict(checkpoint["target_network"])
        else:
            self.q_network.load_state_dict(q_state)
            self.policy_network.load_state_dict(policy_state)
            self.target_network.load_state_dict(checkpoint["target_network"])
        if "q_optimizer" in checkpoint:
            self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        if "policy_optimizer" in checkpoint:
            self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self._step = int(checkpoint["step"])
        self.gamma = float(self._config.get("gamma", self.gamma))
        self.eta = float(self._config.get("eta", self.eta))
        self.epsilon_start = float(self._config.get("epsilon_start", self.epsilon_start))
        self.epsilon_end = float(self._config.get("epsilon_end", self.epsilon_end))
        self.epsilon_decay_steps = int(self._config.get("epsilon_decay_steps", self.epsilon_decay_steps))
        self.epsilon_schedule = str(self._config.get("epsilon_schedule", "linear"))
        self.batch_size = int(self._config.get("batch_size", self.batch_size))
        self.train_every = int(self._config.get("train_every", self.train_every))
        self.updates_per_train = int(self._config.get("updates_per_train", 1))
        self.target_sync_every = int(self._config.get("target_sync_every", self.target_sync_every))
        self.q_loss_name = str(
            self._config.get("q_loss", "huber" if self.obs_version == 2 else "mse")
        )
        q_capacity = int(self._config.get("q_capacity", self.q_buffer.capacity))
        self._q_min = max(self.batch_size, q_capacity // 100)
        obs_dim = int(self.q_network.input_dim)
        if "q_buffer" in checkpoint or "policy_buffer" in checkpoint:
            self._ensure_buffer_width(obs_dim)
            # A stored snapshot can hold more rows than the buffers this model
            # was constructed with, so size them to what the checkpoint needs.
            policy_capacity = int(
                self._config.get("policy_capacity", self.policy_buffer.capacity)
            )
            q_rows = int((checkpoint.get("q_buffer") or {}).get("size", 0))
            p_rows = int((checkpoint.get("policy_buffer") or {}).get("size", 0))
            self._ensure_buffer_capacity(
                max(q_capacity, q_rows) if q_rows else q_capacity,
                max(policy_capacity, p_rows) if p_rows else policy_capacity,
            )
        if "q_buffer" in checkpoint:
            self.q_buffer.load_state_dict(checkpoint["q_buffer"])
        if "policy_buffer" in checkpoint:
            self.policy_buffer.load_state_dict(checkpoint["policy_buffer"])
        return dict(checkpoint.get("training_meta") or {})
