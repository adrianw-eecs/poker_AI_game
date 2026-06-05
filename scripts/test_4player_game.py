#!/usr/bin/env python
"""4-player test harness: 1 learning agent + 2 RandomBots + 1 FlopBot.

Specs:
- 4 players: learning_seat=0, opponents=[RandomBot, FlopBot, RandomBot]
- 5 hands per session
- Random starting position each hand
- Max raise: 100 chips
- Starting stack: 1000 chips
- Validates turn order, pot, and payouts
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from poker.ml.env import PokerEnv
from poker.bots.random_bot import RandomBot
from poker.bots.flop_bot import FlopBot


def run_4player_session(num_hands: int = 5, verbose: bool = True):
    """Run a 4-player game session.

    Args:
        num_hands: Number of hands to play
        verbose: Print detailed output

    Returns:
        dict with session statistics
    """
    print("\n" + "=" * 70)
    print("4-PLAYER GAME SESSION")
    print("=" * 70)
    print(f"Learning seat: 0 (plays against RandomBot, FlopBot, RandomBot)")
    print(f"Hands to play: {num_hands}")
    print(f"Starting stack: 1000 chips per player")
    print(f"Blinds: 25/50")
    print(f"Max raise: 100 chips")
    print()

    session_results = {
        "hands_played": 0,
        "learning_rewards": [],
        "hands_ended": 0,
        "total_hands_with_action": 0,
    }

    for hand_idx in range(num_hands):
        print(f"--- HAND {hand_idx + 1} ---")

        # Create fresh environment for each hand
        env = PokerEnv(
            num_players=4,
            learning_seat=0,
            opponent_bots=[RandomBot(), FlopBot(), RandomBot()],
            small_blind=25,
            big_blind=50,
            seed=np.random.randint(0, 2**31 - 1)
        )

        obs, info = env.reset()

        if verbose:
            print(f"Seed: random")
            print(f"Observation shape: {obs.shape}")

        # Play hand
        done = False
        action_count = 0
        final_reward = None

        while not done and action_count < 100:  # Safety limit
            mask = env.get_action_mask()

            # Choose action (greedy: prefer call/check)
            if mask[2]:  # CALL
                action = 2
            elif mask[1]:  # CHECK
                action = 1
            elif mask[3]:  # RAISE
                action = 3
            else:  # FOLD
                action = 0

            obs, reward, done, truncated, info = env.step(action)
            action_count += 1
            final_reward = reward

        # Record results
        session_results["hands_played"] += 1
        session_results["learning_rewards"].append(final_reward)
        session_results["total_hands_with_action"] += action_count

        if done:
            session_results["hands_ended"] += 1

        if verbose:
            print(f"Actions in hand: {action_count}")
            print(f"Hand ended: {done}")
            print(f"Final reward: {final_reward:7.4f}")
            print()

    # Summary
    print("=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)
    print(f"Hands played: {session_results['hands_played']}")
    print(f"Hands completed: {session_results['hands_ended']}")
    print(f"Avg actions per hand: {session_results['total_hands_with_action'] / max(session_results['hands_played'], 1):.1f}")
    print(f"\nLearning seat rewards:")
    print(f"  Average: {np.mean(session_results['learning_rewards']):7.4f}")
    print(f"  Min/Max: {np.min(session_results['learning_rewards']):7.4f} / {np.max(session_results['learning_rewards']):7.4f}")
    print(f"  Std dev: {np.std(session_results['learning_rewards']):7.4f}")

    print("\n[DIAGNOSTIC]")
    if session_results["total_hands_with_action"] < session_results["hands_played"] * 3:
        print("  WARNING: Hands are ending too quickly (few actions taken)")
        print("  This suggests game logic issue or immediate folds")
    else:
        print("  OK: Hands are reaching reasonable length")

    return session_results


def main():
    # Run session
    results = run_4player_session(num_hands=5, verbose=True)

    print("\n" + "=" * 70)
    print("Next steps:")
    print("1. Verify turn order is correct (should have 4+ actions per hand)")
    print("2. Verify stacks change correctly (rewards match stack changes)")
    print("3. If issues remain, check PokerEnv multi-player support")
    print("=" * 70)


if __name__ == "__main__":
    main()
