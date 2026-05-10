#!/usr/bin/env python
"""NFSP debug training with comprehensive logging to identify learning failures.

This script instruments every critical component:
- Action selection distribution
- Q-network value outputs and masking
- TD target calculations
- Policy network learning
- Gradient flow and weight statistics
- Reward statistics
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poker.bots.call_bot import CallBot
from poker.bots.flop_bot import FlopBot
from poker.bots.random_bot import RandomBot
from poker.ml.env import PokerEnv
from poker.ml.models.nfsp_model import NFSPModel


class DebugNFSPModel(NFSPModel):
    """NFSP with instrumented training for debugging."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_stats = {
            "q_network_outputs": [],
            "masked_outputs": [],
            "action_counts": defaultdict(int),
            "td_targets": [],
            "predicted_q": [],
            "td_errors": [],
            "policy_logits": [],
            "gradient_norms_q": [],
            "gradient_norms_policy": [],
            "q_weight_stats": [],
            "policy_weight_stats": [],
            "reward_stats": [],
        }

    def select_action(self, obs, legal_mask, training=True):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        if not training:
            self.policy_network.eval()
            with torch.no_grad():
                logits = self._apply_mask(self.policy_network(obs_t).squeeze(0), legal_mask)
                probs = torch.softmax(logits, dim=0)
            return int(torch.multinomial(probs, 1).item())

        use_best_response = np.random.random() < self.eta

        if use_best_response:
            epsilon = self._current_epsilon()
            if np.random.random() < epsilon:
                legal_indices = np.where(legal_mask)[0]
                action = int(np.random.choice(legal_indices))
            else:
                self.q_network.eval()
                with torch.no_grad():
                    q_vals_raw = self.q_network(obs_t).squeeze(0)
                    # LOG: Raw Q-network output
                    self.debug_stats["q_network_outputs"].append(q_vals_raw.cpu().numpy())

                    q_vals = self._apply_mask(q_vals_raw, legal_mask)
                    # LOG: Masked output
                    self.debug_stats["masked_outputs"].append(q_vals.cpu().numpy())

                action = int(q_vals.argmax().item())
            self.policy_buffer.add(obs, action)
            self.debug_stats["action_counts"][action] += 1
            return action
        else:
            self.policy_network.eval()
            with torch.no_grad():
                logits = self._apply_mask(self.policy_network(obs_t).squeeze(0), legal_mask)
                self.debug_stats["policy_logits"].append(logits.cpu().numpy())
                probs = torch.softmax(logits, dim=0)
            action = int(torch.multinomial(probs, 1).item())
            self.debug_stats["action_counts"][action] += 1
            return action

    def train_step(self):
        self._step += 1

        if self._step % self.train_every != 0:
            return {"q_loss": None, "policy_loss": None}

        q_ready = len(self.q_buffer) >= 512
        policy_ready = len(self.policy_buffer) >= 64
        if not (q_ready and policy_ready):
            return {"q_loss": None, "policy_loss": None}

        # Q-network update with detailed logging
        self.q_network.train()
        batch = self.q_buffer.sample(self.batch_size)
        obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        # LOG reward statistics
        self.debug_stats["reward_stats"].append({
            "mean": rewards_t.mean().item(),
            "std": rewards_t.std().item(),
            "min": rewards_t.min().item(),
            "max": rewards_t.max().item(),
        })

        with torch.no_grad():
            next_q = self.target_network(next_obs_t).max(dim=1).values
            td_targets = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        q_vals = self.q_network(obs_t)
        predicted_q = q_vals.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        q_loss = torch.nn.functional.mse_loss(predicted_q, td_targets)

        # LOG TD targets and predictions
        self.debug_stats["td_targets"].append(td_targets.mean().item())
        self.debug_stats["predicted_q"].append(predicted_q.mean().item())
        td_error = (td_targets - predicted_q).abs().mean().item()
        self.debug_stats["td_errors"].append(td_error)

        self.q_optimizer.zero_grad()
        q_loss.backward()

        # LOG gradient norms
        q_grad_norm = 0.0
        for param in self.q_network.parameters():
            if param.grad is not None:
                q_grad_norm += param.grad.norm().item() ** 2
        q_grad_norm = np.sqrt(q_grad_norm)
        self.debug_stats["gradient_norms_q"].append(q_grad_norm)

        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.q_optimizer.step()

        # LOG Q-network weight statistics
        q_weight_norm = 0.0
        for param in self.q_network.parameters():
            q_weight_norm += param.data.norm().item() ** 2
        self.debug_stats["q_weight_stats"].append({
            "norm": np.sqrt(q_weight_norm),
            "loss": q_loss.item(),
        })

        # Policy-network update with detailed logging
        self.policy_network.train()
        pbatch = self.policy_buffer.sample(self.batch_size)
        pobs_t = torch.as_tensor(pbatch["obs"], dtype=torch.float32, device=self.device)
        plabels_t = torch.as_tensor(pbatch["actions"], dtype=torch.int64, device=self.device)

        logits = self.policy_network(pobs_t)
        policy_loss = torch.nn.functional.cross_entropy(logits, plabels_t)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()

        # LOG gradient norms
        policy_grad_norm = 0.0
        for param in self.policy_network.parameters():
            if param.grad is not None:
                policy_grad_norm += param.grad.norm().item() ** 2
        policy_grad_norm = np.sqrt(policy_grad_norm)
        self.debug_stats["gradient_norms_policy"].append(policy_grad_norm)

        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        # LOG policy weight statistics
        policy_weight_norm = 0.0
        for param in self.policy_network.parameters():
            policy_weight_norm += param.data.norm().item() ** 2
        self.debug_stats["policy_weight_stats"].append({
            "norm": np.sqrt(policy_weight_norm),
            "loss": policy_loss.item(),
        })

        if self._step % self.target_sync_every == 0:
            from poker.ml.models.nfsp_networks import sync_target_network
            sync_target_network(self.q_network, self.target_network)

        return {"q_loss": q_loss.item(), "policy_loss": policy_loss.item()}


def get_opponent_for_episode(episode: int, seed: int):
    """Rotate through opponent types."""
    cycle_length = 10000
    cycle_type = (episode // cycle_length) % 3
    if cycle_type == 0:
        return [RandomBot(seed=seed + episode)]
    elif cycle_type == 1:
        return [FlopBot(seed=seed + episode)]
    else:
        return [CallBot(seed=seed + episode)]


def main():
    parser = argparse.ArgumentParser(description="NFSP Debug Training")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    print(f"\n{'='*80}")
    print(f"NFSP DEBUG TRAINING - Comprehensive Logging")
    print(f"{'='*80}\n")

    model = DebugNFSPModel()
    start_time = time.time()
    episode_times = []

    for episode in range(args.episodes):
        ep_start = time.time()
        opponents = get_opponent_for_episode(episode, args.seed or 42)
        env = PokerEnv(
            num_players=2,
            learning_seat=0,
            opponent_bots=opponents,
            seed=args.seed,
        )

        obs, _ = env.reset()
        done = False
        while not done:
            mask = env.get_action_mask()
            action = model.select_action(obs, mask, training=True)
            next_obs, reward, done, _, _ = env.step(action)

            # Anti-folding bonus
            if action == 0:
                reward -= 0.01
            else:
                reward += 0.005

            model.store_transition(obs, action, reward, next_obs, done)
            model.train_step()
            obs = next_obs

        episode_times.append(time.time() - ep_start)

        # Every 500 episodes, print detailed diagnostics
        if episode % 500 == 0 and episode > 0:
            elapsed = time.time() - start_time
            avg_ep_time = np.mean(episode_times[-100:])

            # Action distribution
            total_actions = sum(model.debug_stats["action_counts"].values())
            action_names = ["FOLD", "CHECK", "CALL", "RAISE_2x", "RAISE_3x", "RAISE_4x", "ALL_IN"]

            print(f"\n[Ep {episode:5d}] Time: {elapsed:7.1f}s")
            print(f"  Action Distribution:")
            for i in range(7):
                count = model.debug_stats["action_counts"].get(i, 0)
                pct = 100 * count / total_actions if total_actions > 0 else 0
                print(f"    {action_names[i]:10s}: {count:6d} ({pct:5.1f}%)")

            # Q-network diagnostics
            if model.debug_stats["q_weight_stats"]:
                recent_q_loss = [s["loss"] for s in model.debug_stats["q_weight_stats"][-10:]]
                print(f"  Q-network:")
                print(f"    Recent loss: {np.mean(recent_q_loss):.6f}")
                print(f"    Gradient norm (mean): {np.mean(model.debug_stats['gradient_norms_q'][-10:]):.6f}")
                print(f"    Weight norm (latest): {model.debug_stats['q_weight_stats'][-1]['norm']:.2f}")

            # Policy-network diagnostics
            if model.debug_stats["policy_weight_stats"]:
                recent_policy_loss = [s["loss"] for s in model.debug_stats["policy_weight_stats"][-10:]]
                print(f"  Policy-network:")
                print(f"    Recent loss: {np.mean(recent_policy_loss):.6f}")
                print(f"    Gradient norm (mean): {np.mean(model.debug_stats['gradient_norms_policy'][-10:]):.6f}")
                print(f"    Weight norm (latest): {model.debug_stats['policy_weight_stats'][-1]['norm']:.2f}")

            # TD diagnostics
            if model.debug_stats["td_targets"]:
                print(f"  TD Learning:")
                print(f"    Target (mean, last 10): {np.mean(model.debug_stats['td_targets'][-10:]):.6f}")
                print(f"    Predicted (mean, last 10): {np.mean(model.debug_stats['predicted_q'][-10:]):.6f}")
                print(f"    TD Error (mean, last 10): {np.mean(model.debug_stats['td_errors'][-10:]):.6f}")

            # Reward diagnostics
            if model.debug_stats["reward_stats"]:
                last_reward_stat = model.debug_stats["reward_stats"][-1]
                print(f"  Rewards (latest batch):")
                print(f"    Mean: {last_reward_stat['mean']:+.4f}, Std: {last_reward_stat['std']:.4f}")
                print(f"    Range: [{last_reward_stat['min']:+.4f}, {last_reward_stat['max']:+.4f}]")

            # Reset stats for next period
            model.debug_stats["action_counts"].clear()

    print(f"\n{'='*80}")
    print(f"Training Complete")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
