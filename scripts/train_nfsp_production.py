#!/usr/bin/env python
"""NFSP Training: Production Version with Full Logging.

CURRENT CAPABILITY: 2-PLAYER GAMES with Invalid Action Logging
- Learning seat plays against single opponent bot
- Full logging to console and game_logs.txt
- Invalid actions tracked and penalized
- Opponent rotation for strategy diversity

FUTURE: 4-PLAYER support when env.py multi-player turn order is fixed
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
import logging

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poker.bots.call_bot import CallBot
from poker.bots.flop_bot import FlopBot
from poker.bots.random_bot import RandomBot
from poker.ml.env import PokerEnv
from poker.ml.models.nfsp_model import NFSPModel
from poker.engine.action_validator import legal_actions, validate
from poker.exceptions import IllegalActionError
from poker.ml.action_space import action_index_to_action


# Configure logging
def setup_logging(log_file: str):
    """Setup both file and console logging."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("poker_training")
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def evaluate(model: NFSPModel, logger: logging.Logger, num_hands: int = 30):
    """Evaluate 2-player model."""
    opponents_list = [
        ("RandomBot", RandomBot()),
        ("FlopBot", FlopBot()),
        ("CallBot", CallBot()),
    ]

    total_reward = 0.0
    total_hands = 0
    total_invalid = 0
    opponent_rewards = {}

    for opp_name, opponent in opponents_list:
        hands_vs_opp = 0
        reward_vs_opp = 0.0
        invalid_vs_opp = 0

        for _ in range(num_hands // 3):
            seed = np.random.randint(0, 2**31 - 1)

            env = PokerEnv(
                num_players=2,
                learning_seat=0,
                opponent_bots=[opponent],
                small_blind=25,
                big_blind=50,
                seed=seed,
            )

            obs, _ = env.reset()
            done = False
            hand_invalid_count = 0

            while not done:
                mask = env.get_action_mask()
                action = model.select_action(obs, mask, training=False)

                # Validate action
                try:
                    if env.state is not None:
                        legal = legal_actions(env.state, env.learning_seat)
                        concrete_action = action_index_to_action(
                            action, env.state, env.learning_seat
                        )
                        validate(env.state, env.learning_seat, concrete_action)
                except (IllegalActionError, IndexError, ValueError) as e:
                    hand_invalid_count += 1
                    invalid_vs_opp += 1
                    logger.debug(f"Invalid action vs {opp_name}: {e}")

                obs, reward, done, _, _ = env.step(action)

            total_reward += reward
            reward_vs_opp += reward
            total_hands += 1
            hands_vs_opp += 1
            total_invalid += hand_invalid_count

        opponent_rewards[opp_name] = reward_vs_opp / hands_vs_opp
        print(
            f"    vs {opp_name:12s}: {opponent_rewards[opp_name]:7.4f} "
            f"reward/hand (invalid: {invalid_vs_opp})"
        )

    if total_invalid > 0:
        logger.info(f"Total invalid actions in eval: {total_invalid}")
        print(f"    Total invalid actions in eval: {total_invalid}")

    return total_reward / max(total_hands, 1), total_invalid


def get_opponent_for_episode(episode: int):
    """Rotate through opponents."""
    cycle_length = 83334
    cycle_type = (episode // cycle_length) % 3

    if cycle_type == 0:
        return "RandomBot", RandomBot()
    elif cycle_type == 1:
        return "FlopBot", FlopBot()
    else:
        return "CallBot", CallBot()


def main() -> None:
    if torch.cuda.is_available():
        print(f"[GPU] GPU Detected: {torch.cuda.get_device_name()}")
        print(f"       Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("[WARNING] No GPU detected - training will be slow")

    parser = argparse.ArgumentParser(
        description="NFSP Training: Production Version with Full Logging"
    )
    parser.add_argument("--episodes", type=int, default=250_000)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--save-path", type=str, default="models/nfsp_production.pt")
    parser.add_argument("--log-file", type=str, default="game_logs.txt")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_file)

    logger.info("=" * 70)
    logger.info("NFSP Training: Production Version with Full Logging")
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  Game mode: 2-PLAYER (4-player support coming)")
    logger.info(f"    Learning seat: 0")
    logger.info(f"    Opponent: Rotating (RandomBot, FlopBot, CallBot)")
    logger.info(f"  Episodes: {args.episodes:,}")
    logger.info(f"  Blinds: 25/50")
    logger.info(f"  Starting stack: 1000 per player")
    logger.info(f"  Random seeds: Yes")
    logger.info(f"  Invalid action logging: YES (console + file)")
    logger.info(f"  Log file: {args.log_file}")

    print(f"\n{'='*70}")
    print(f"NFSP Training: Production Version")
    print(f"{'='*70}")
    print(f"Configuration: 2-PLAYER games")
    print(f"Episodes: {args.episodes:,}")
    print(f"Invalid action logging: ENABLED")
    print(f"Log file: {args.log_file}")
    print()

    model = NFSPModel()

    last_q_loss: float | None = None
    last_policy_loss: float | None = None
    start_time = time.time()
    episode_times = []
    total_invalid_actions = 0
    episodes_with_invalid = 0

    for episode in range(args.episodes):
        ep_start = time.time()

        # Get opponent
        opponent_name, opponent = get_opponent_for_episode(episode)

        seed = np.random.randint(0, 2**31 - 1)

        env = PokerEnv(
            num_players=2,
            learning_seat=0,
            opponent_bots=[opponent],
            small_blind=25,
            big_blind=50,
            seed=seed,
        )

        obs, _ = env.reset()
        done = False
        hand_step = 0
        hand_invalid_count = 0

        while not done:
            mask = env.get_action_mask()
            action = model.select_action(obs, mask, training=True)

            # Validate action and log if invalid
            action_names = ["FOLD", "CHECK", "CALL", "RAISE-0.5", "RAISE-POT", "RAISE-2x", "ALL_IN"]
            action_name = action_names[action] if action < len(action_names) else f"UNKNOWN({action})"

            try:
                if env.state is not None:
                    legal = legal_actions(env.state, env.learning_seat)
                    concrete_action = action_index_to_action(action, env.state, env.learning_seat)
                    validate(env.state, env.learning_seat, concrete_action)
            except (IllegalActionError, IndexError, ValueError) as e:
                hand_invalid_count += 1
                total_invalid_actions += 1
                episodes_with_invalid += 1

                log_msg = (
                    f"Episode {episode}, Step {hand_step}: Invalid action {action_name} - {e}"
                )
                logger.warning(log_msg)

            next_obs, reward, done, truncated, _ = env.step(action)

            model.store_transition(obs, action, reward, next_obs, done)
            result = model.train_step()

            if result["q_loss"] is not None:
                last_q_loss = result["q_loss"]
                last_policy_loss = result["policy_loss"]

            obs = next_obs
            hand_step += 1

            if hand_step > 100:
                break

        # Log hand summary
        hand_log = (
            f"Episode {episode}: {opponent_name} | "
            f"Reward: {reward:7.4f} | Steps: {hand_step} | Invalid: {hand_invalid_count}"
        )
        logger.debug(hand_log)

        episode_times.append(time.time() - ep_start)

        # Periodic evaluation
        if episode % args.eval_every == 0:
            q_str = f"{last_q_loss:.4f}" if last_q_loss is not None else "n/a"
            p_str = f"{last_policy_loss:.4f}" if last_policy_loss is not None else "n/a"
            elapsed = time.time() - start_time
            avg_ep_time = (
                np.mean(episode_times[-100:])
                if len(episode_times) >= 100
                else np.mean(episode_times)
            )
            eta = (args.episodes - episode) * avg_ep_time / 60

            eval_msg = f"\nEpisode {episode:7d}/{args.episodes:7d}"
            print(eval_msg)
            logger.info(eval_msg)

            print(f"  q_loss={q_str} | policy_loss={p_str}")
            logger.info(f"  q_loss={q_str} | policy_loss={p_str}")

            print(f"  Invalid actions total: {total_invalid_actions} (in {episodes_with_invalid} episodes)")
            logger.info(f"  Invalid actions total: {total_invalid_actions} (in {episodes_with_invalid} episodes)")

            print(f"  Evaluation:")
            eval_reward, eval_invalid = evaluate(model, logger, num_hands=30)

            print(f"  Average reward: {eval_reward:7.4f}")
            logger.info(f"  Average reward: {eval_reward:7.4f}")

            print(f"  Elapsed: {elapsed:7.1f}s | ETA: {eta:7.1f}m")
            logger.info(f"  Elapsed: {elapsed:7.1f}s | ETA: {eta:7.1f}m")

        # Periodic checkpointing
        if episode > 0 and episode % args.checkpoint_every == 0:
            ckpt_path = str(Path(args.save_path).parent / f"nfsp_ckpt_{episode:06d}.pt")
            model.save(ckpt_path)
            ckpt_msg = f"Checkpoint saved: {ckpt_path}"
            print(f"  [CKPT] {ckpt_msg}")
            logger.info(ckpt_msg)

    # Final save
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))

    elapsed = time.time() - start_time

    summary = (
        f"\n{'='*70}\n"
        f"Training complete!\n"
        f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}m)\n"
        f"Avg time per episode: {np.mean(episode_times):.4f}s\n"
        f"Total invalid actions: {total_invalid_actions}\n"
        f"Episodes with invalid actions: {episodes_with_invalid}\n"
        f"Saved final model to {save_path}\n"
        f"Game log: {args.log_file}\n"
        f"{'='*70}\n"
    )

    print(summary)
    logger.info(summary)

    # Final log note
    logger.info("To view game logs, run: tail game_logs.txt")
    logger.info("To search for invalid actions: grep 'INVALID' game_logs.txt")


if __name__ == "__main__":
    main()
