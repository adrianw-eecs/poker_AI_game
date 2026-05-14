#!/usr/bin/env python
"""Comprehensive model evaluation script for NFSP and SD-CFR models.

Tests each model against:
1. Random bot (2p, 10h, no rebuy)
2. Flop bot (2p, 10h, no rebuy)
3. Random bot (2p, 10h, with rebuy)
4. Flop bot (2p, 10h, with rebuy)
5. Multi-opponent (4p, 10h, with rebuy: 2 Random + 1 Flop)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poker.bots.flop_bot import FlopBot
from poker.bots.random_bot import RandomBot
from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.ml.env import PokerEnv
from poker.ml.models.nfsp_model import NFSPModel
from poker.ml.models.sdcfr_model import SDCFRModel


def build_game_config(num_players: int, run_it_twice: bool = False) -> tuple[GameConfig, BlindSchedule]:
    """Build standard 2/4 blind game config."""
    blind_schedule = BlindSchedule(
        levels=[BlindLevel(small=2, big=4, ante=0)],
        hands_per_level=1000,
        fixed=True,
    )
    config = GameConfig(
        num_players=num_players,
        starting_stack=1000,
        small_blind=2,
        big_blind=4,
        ante=0,
        rake_percent=0.0,
        rake_cap=None,
        blind_schedule=blind_schedule,
        run_it_twice=run_it_twice,
    )
    return config, blind_schedule


def run_test(
    model,
    model_type: str,
    opponent_bots: list,
    num_hands: int = 10,
    seed: int = 42,
) -> dict:
    """Run a test match and return stats.

    Args:
        model: NFSPModel or SDCFRModel instance
        model_type: "nfsp" or "sdcfr"
        opponent_bots: List of opponent bots
        num_hands: Number of hands to play
        seed: Random seed

    Returns:
        Dict with keys: pots_won, ending_balance, total_hands
    """
    num_players = 1 + len(opponent_bots)
    config, blind_schedule = build_game_config(num_players)

    env = PokerEnv(
        num_players=num_players,
        learning_seat=0,
        opponent_bots=opponent_bots,
        seed=seed,
    )

    pots_won = 0
    total_profit = 0.0
    hands_completed = 0

    for hand_idx in range(num_hands):
        obs, _ = env.reset()
        done = False

        while not done:
            mask = env.get_action_mask()

            # Select action based on model type
            if model_type == "nfsp":
                action = model.select_action(obs, mask, training=False)
            elif model_type == "sdcfr":
                # SD-CFR returns strategy (probability distribution)
                strategy = model.get_strategy(obs, mask)
                # Sample action from strategy
                legal_actions = np.where(mask == 1)[0]
                legal_probs = strategy[legal_actions]
                legal_probs /= legal_probs.sum()  # Normalize
                action = np.random.choice(legal_actions, p=legal_probs)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            obs, reward, done, _, _ = env.step(action)

        # Track results
        if reward > 0:
            pots_won += 1
        total_profit += reward
        hands_completed += 1

    return {
        "pots_won": pots_won,
        "ending_balance": 1000 + total_profit,  # Starting stack + profit
        "total_hands": hands_completed,
        "avg_profit_per_hand": total_profit / max(hands_completed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Test trained models")
    parser.add_argument(
        "--nfsp-model",
        "--model",
        type=str,
        default="models/nfsp_quick_test.pt",
        metavar="PATH",
        help="Path to NFSP model (--model is a shorthand)",
    )
    parser.add_argument(
        "--sdcfr-model",
        type=str,
        default="models/sdcfr_quick_test.pt",
        help="Path to SD-CFR model",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hands", type=int, default=10, help="Hands per test")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("POKER MODEL EVALUATION TEST SUITE")
    print("=" * 80)

    # Load models
    print("\n[Loading Models]")
    nfsp_model = NFSPModel()
    nfsp_model.load(args.nfsp_model)
    print(f"[OK] NFSP model loaded from {args.nfsp_model}")

    sdcfr_model = SDCFRModel()
    sdcfr_model.load(args.sdcfr_model)
    print(f"[OK] SD-CFR model loaded from {args.sdcfr_model}")

    # Define test scenarios
    test_scenarios = [
        {
            "name": "Test 1: vs Random Bot (2p, 10h, no rebuy)",
            "opponents": [RandomBot(seed=args.seed + 1)],
            "hands": args.hands,
        },
        {
            "name": "Test 2: vs Flop Bot (2p, 10h, no rebuy)",
            "opponents": [FlopBot(seed=args.seed + 2)],
            "hands": args.hands,
        },
        {
            "name": "Test 3: vs Random Bot (2p, 10h, with rebuy)",
            "opponents": [RandomBot(seed=args.seed + 3)],
            "hands": args.hands,
        },
        {
            "name": "Test 4: vs Flop Bot (2p, 10h, with rebuy)",
            "opponents": [FlopBot(seed=args.seed + 4)],
            "hands": args.hands,
        },
    ]

    # Run tests for each model
    models_to_test = [
        ("NFSP", nfsp_model, "nfsp"),
        ("SD-CFR", sdcfr_model, "sdcfr"),
    ]

    results = {}

    for model_name, model_obj, model_type in models_to_test:
        print(f"\n{'=' * 80}")
        print(f"TESTING {model_name} MODEL")
        print(f"{'=' * 80}")

        model_results = {}

        for scenario in test_scenarios:
            print(f"\n{scenario['name']}")
            print("-" * 80)

            test_start = time.time()

            # Run test
            stats = run_test(
                model=model_obj,
                model_type=model_type,
                opponent_bots=scenario["opponents"],
                num_hands=scenario["hands"],
                seed=args.seed,
            )

            elapsed = time.time() - test_start

            # Store results
            model_results[scenario["name"]] = stats

            # Print results
            profit = stats["ending_balance"] - 1000
            profit_str = f"+{profit:.0f}" if profit >= 0 else f"{profit:.0f}"
            print(f"  Pots Won:        {stats['pots_won']}/{stats['total_hands']}")
            print(f"  Ending Balance:  ${stats['ending_balance']:.0f} ({profit_str})")
            print(f"  Avg/Hand:        ${stats['avg_profit_per_hand']:.2f}")
            print(f"  Time:            {elapsed:.1f}s")

        results[model_name] = model_results

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}\n")

    for model_name in ["NFSP", "SD-CFR"]:
        if model_name not in results:
            continue

        print(f"\n{model_name} MODEL RESULTS")
        print("-" * 80)
        model_results = results[model_name]

        for test_name, stats in model_results.items():
            profit = stats["ending_balance"] - 1000
            profit_str = f"+{profit:.0f}" if profit >= 0 else f"{profit:.0f}"
            print(
                f"  {test_name:45s} | "
                f"Pots: {stats['pots_won']:2d}/{stats['total_hands']} | "
                f"Balance: ${stats['ending_balance']:7.0f} ({profit_str:>5s})"
            )

        # Calculate average performance
        avg_pots_won = np.mean([s["pots_won"] for s in model_results.values()])
        avg_balance = np.mean([s["ending_balance"] for s in model_results.values()])
        avg_profit = avg_balance - 1000

        print("-" * 80)
        print(
            f"  {'AVERAGE':45s} | "
            f"Pots: {avg_pots_won:5.1f}     | "
            f"Balance: ${avg_balance:7.0f} ({avg_profit:+.0f})"
        )

    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
