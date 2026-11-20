#!/usr/bin/env python
"""Train an SD-CFR teacher on observation v2 or v5.

v2 is the 318-wide buy-in encoder. v5 is the 398-wide tournament encoder:
money in big blinds, the same stack profiles the hybrid learner sits, and
per-traversal blinds, antes, and stacks. A v2 teacher must not label v5
states. The two widths are different games.

  python scripts/train_sdcfr_v2.py --iterations 400 --traversals 200
  python scripts/train_sdcfr_v2.py --obs-version 5 --iterations 400 --traversals 200

Note: SD-CFR's trainer re-initialises the advantage net each iteration by
design, so iterations do not accumulate weights. Each iteration fits the
regret buffer collected so far.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker.bots.flop_bot import FlopBot
from poker.bots.random_bot import RandomBot
from poker.bots.sdcfr_bot import SDCFRBot
from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.ml.action_space import BetScheme
from poker.ml.cfr.traversal import ExternalSamplingTraverser
from poker.ml.models.sdcfr_model import SDCFRModel
from poker.ml.sitting import buyin_chips, draw_sitting


def make_config(
    num_players: int,
    starting_stack: int = 1000,
    schedule: BlindSchedule | None = None,
) -> GameConfig:
    if schedule is None:
        schedule = BlindSchedule(
            levels=[BlindLevel(25, 50, 0)], hands_per_level=10**9, fixed=True
        )
    opening = schedule.levels[0]
    return GameConfig(
        num_players=num_players,
        starting_stack=starting_stack,
        small_blind=opening.small,
        big_blind=opening.big,
        ante=opening.ante,
        rake_percent=0.0,
        rake_cap=None,
        blind_schedule=schedule,
        run_it_twice=False,
    )


def evaluate_vs_bots(
    model: SDCFRModel, hands: int, players: int, seed: int, scheme: BetScheme
) -> float:
    """Mean reward per hand against Random/Flop, using the regret-matched policy."""
    from poker.ml.env import PokerEnv

    bot = SDCFRBot(name="eval", model=model)
    rewards: list[float] = []
    rng = np.random.RandomState(seed)
    for _ in range(hands):
        n = int(rng.randint(2, players + 1))
        opponents = [
            RandomBot(seed=int(rng.randint(0, 2**31 - 1))) if i % 2 == 0 else FlopBot()
            for i in range(n - 1)
        ]
        env = PokerEnv(
            num_players=n,
            learning_seat=0,
            opponent_bots=opponents,
            small_blind=25,
            big_blind=50,
            seed=int(rng.randint(0, 2**31 - 1)),
            obs_version=int(model.obs_version),
            bet_scheme=scheme,
        )
        obs, _ = env.reset()
        done = False
        reward = 0.0
        while not done:
            if env.state is not None and env.state.action_on_seat == 0:
                mask = env.get_action_mask()
                strategy = model.get_strategy(obs, mask.astype(np.float32))
                action = int(np.random.choice(7, p=strategy))
            else:
                action = 1
            try:
                obs, reward, done, _, _ = env.step(action)
            except ValueError:
                obs, reward, done, _, _ = env.step(1)
        rewards.append(float(reward))
    return float(np.mean(rewards)) if rewards else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description="SD-CFR training on obs v2 or v5")
    p.add_argument("--obs-version", type=int, default=2, choices=(2, 5),
                   help="2 keeps the buy-in encoder. 5 trains on sitting profiles.")
    p.add_argument("--iterations", type=int, default=400)
    p.add_argument("--traversals", type=int, default=200,
                   help="Traversals per iteration")
    p.add_argument("--min-players", type=int, default=2)
    p.add_argument("--max-players", type=int, default=9)
    p.add_argument("--capacity", type=int, default=500_000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--sgd-steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-hands", type=int, default=30)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"[SEED] {args.seed}")

    if args.model_name is None:
        args.model_name = f"sdcfr_v{args.obs_version}"
    models_dir = Path("models") / args.model_name
    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / f"sdcfr_v{args.obs_version}.pt"

    model = SDCFRModel(
        lr=args.lr,
        batch_size=args.batch_size,
        sgd_steps_per_iteration=args.sgd_steps,
        capacity=args.capacity,
        obs_version=args.obs_version,
        bet_scheme=BetScheme.POT_FRACTION,
    )
    if args.checkpoint:
        print(f"[LOAD] {args.checkpoint}")
        model.load(args.checkpoint)
        if model.obs_version != args.obs_version:
            raise SystemExit(
                f"{args.checkpoint} is obs v{model.obs_version}; "
                f"this run trains v{args.obs_version}."
            )

    print(
        f"[INIT] obs_v={model.obs_version} dim={model.obs_dim} "
        f"scheme={model.bet_scheme.value} capacity={args.capacity:,}"
    )

    rng = np.random.RandomState(args.seed if args.seed is not None else 0)
    start = time.time()
    best = -1e18

    profile_rng = np.random.default_rng(args.seed if args.seed is not None else 0)
    for iteration in range(1, args.iterations + 1):
        spec = None
        if args.obs_version == 5:
            spec = draw_sitting(
                profile_rng,
                min_players=args.min_players,
                max_players=args.max_players,
                hand_cap=8,
            )
            n = spec.field_size
            config = make_config(n, buyin_chips(), spec.schedule)
        else:
            n = int(rng.randint(args.min_players, args.max_players + 1))
            config = make_config(n)
        # Rebuild the traverser each iteration: train_iteration() replaces the
        # advantage network, so a stale reference would read an untrained net.
        model.advantage_network.eval()

        traverser = ExternalSamplingTraverser(
            config=config,
            advantage_network=model.advantage_network,
            device=model.device,
            regret_buffer=model.regret_buffer,
            obs_version=args.obs_version,
            bet_scheme=BetScheme.POT_FRACTION,
        )

        for t in range(args.traversals):
            seat = int(rng.randint(0, n))
            # Mix baseline bots with the model's own policy so the regret
            # targets are not specialised to random play.
            opponents = []
            for k in range(n - 1):
                if t % 3 == 0:
                    opponents.append(SDCFRBot(name=f"self{k}", model=model))
                elif (t + k) % 2 == 0:
                    opponents.append(RandomBot(seed=int(rng.randint(0, 2**31 - 1))))
                else:
                    opponents.append(FlopBot())
            traverser.traverse(
                traverser_seat=seat,
                cfr_iteration=iteration,
                opponent_bots=opponents,
                num_players=n,
                starting_stacks=None if spec is None else spec.starting_stacks,
                blind_schedule=None if spec is None else spec.schedule,
                hand_number=0,
                deck_seed=int(rng.randint(0, 2**31 - 1)),
            )

        result = model.train_iteration()
        if iteration % 10 == 0 or iteration == 1:
            loss = result["loss"]
            print(
                f"[{time.strftime('%H:%M:%S')}] iter {iteration:5d}: "
                f"loss={loss:.4f} buffer={result['buffer_size']:,} "
                f"elapsed={time.time() - start:.0f}s",
                flush=True,
            )

        if iteration % args.eval_every == 0:
            score = evaluate_vs_bots(
                model, args.eval_hands, args.max_players, 17_001, BetScheme.POT_FRACTION
            )
            print(f"  [EVAL] {score:+.4f} reward/hand vs Random+Flop")
            if score > best:
                best = score
                best_path = models_dir / f"sdcfr_v{args.obs_version}_best.pt"
                model.save(str(best_path))
                print(f"  [BEST] -> {best_path}")

    model.save(str(save_path))
    print(f"\n[DONE] {time.time() - start:.0f}s | {save_path}")
    print(f"       use as teacher: --teacher {save_path}")


if __name__ == "__main__":
    main()
