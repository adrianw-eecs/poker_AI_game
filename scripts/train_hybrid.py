#!/usr/bin/env python
"""Train the NFSP + SD-CFR hybrid: SD-CFR acts as teacher, NFSP as learner.

Implements Phase C/D of docs/HYBRID_PLAN.md. SD-CFR traversals generate
regret targets; the NFSP policy head is trained on them (weight kappa) in
addition to its own reservoir of best-response actions.

  # v2 SD-CFR teacher is required (v1 weights cannot feed a 318-wide policy)
  python scripts/train_sdcfr_v2.py --iterations 400 --traversals 200
  python scripts/train_hybrid.py --teacher models/sdcfr_v2/sdcfr_v2.pt

  # quick smoke
  python scripts/train_hybrid.py --episodes 200 --kappa 0.25 --no-validation
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker.bots.call_bot import CallBot
from poker.bots.flop_bot import FlopBot
from poker.bots.random_bot import RandomBot
from poker.ml.action_space import BetScheme, build_action_mask
from poker.ml.cfr.traversal import ExternalSamplingTraverser
from poker.ml.env import PokerEnv
from poker.ml.metrics import bb100_interval
from poker.ml.models.nfsp_model import NFSPModel
from poker.ml.models.sdcfr_model import SDCFRModel
from poker.ml.observation_v2 import OBS_V2_DIM, build_observation_v2

_MAX_STEPS = 200
_EVAL_SEED = 17_001


def _load_teacher(path: Path) -> SDCFRModel | None:
    if not path.is_file():
        return None
    teacher = SDCFRModel(obs_version=2)
    teacher.load(str(path))
    if teacher.obs_version != 2:
        raise SystemExit(
            f"{path} is an obs v{teacher.obs_version} checkpoint. The hybrid needs a "
            "v2 teacher (318-wide); retrain it with scripts/train_sdcfr_v2.py."
        )
    teacher.advantage_network.eval()
    return teacher


def collect_regret_targets(
    model: NFSPModel,
    teacher: SDCFRModel,
    traversals: int,
    num_players: int,
    seed: int,
) -> tuple[int, int]:
    """Run SD-CFR traversals and feed their targets into the distillation memory.

    Opponents are the current NFSP average policy, which is what makes this a
    self-play distillation rather than a distillation of random play.

    Returns (rows added, traversals that reached a decision).
    """
    from poker.config.blind_schedule import BlindLevel, BlindSchedule
    from poker.config.game_config import GameConfig

    config = GameConfig(
        num_players=num_players,
        starting_stack=1000,
        small_blind=25,
        big_blind=50,
        ante=0,
        rake_percent=0.0,
        rake_cap=None,
        blind_schedule=BlindSchedule(
            levels=[BlindLevel(25, 50, 0)], hands_per_level=10**9, fixed=True
        ),
        run_it_twice=False,
    )
    traverser = ExternalSamplingTraverser(
        config=config,
        advantage_network=teacher.advantage_network,
        device=teacher.device,
        regret_buffer=teacher.regret_buffer,
        obs_version=2,
        bet_scheme=BetScheme.POT_FRACTION,
    )

    before = len(teacher.regret_buffer)
    used = 0
    rng = np.random.RandomState(seed)
    for i in range(traversals):
        seat = int(rng.randint(0, num_players))
        bots = [
            RandomBot(seed=int(rng.randint(0, 2**31 - 1)))
            if (i + k) % 2 == 0
            else FlopBot()
            for k in range(num_players - 1)
        ]
        traverser.traverse(
            traverser_seat=seat,
            cfr_iteration=max(1, teacher.cfr_iteration),
            opponent_bots=bots,
            num_players=num_players,
        )
        used += 1

    added = len(teacher.regret_buffer) - before
    # Move the freshly collected rows into the learner's distillation memory.
    rows = _drain_new_targets(teacher, added, model)
    return rows, used


def _drain_new_targets(teacher: SDCFRModel, count: int, model: NFSPModel) -> int:
    """Copy the newest buffer rows into the learner as (obs, adv, mask) triples.

    The traverser stores [obs | advantages] and does not persist the legal mask,
    so legality is recovered from the advantages' support: an advantage row is
    zero off-support by construction in outcome sampling.
    """
    if count <= 0 or len(teacher.regret_buffer) == 0:
        return 0
    buf = teacher.regret_buffer
    obs_dim = teacher.obs_dim
    taken = min(count, len(buf))
    batch = buf.sample(taken)
    packed = batch["obs"]
    added = 0
    for row in packed:
        obs = row[:obs_dim].astype(np.float32)
        adv = row[obs_dim: obs_dim + 7].astype(np.float32)
        # Recover a usable mask: treat "fold/call" as always legal and any
        # action carrying an advantage as legal, so regret matching has support.
        mask = np.zeros(7, dtype=np.float32)
        mask[0] = 1.0
        mask[1] = 1.0
        mask[np.abs(adv) > 0] = 1.0
        if mask.sum() < 2:
            continue
        model.add_regret_sample(obs, adv, mask, weight=1.0)
        added += 1
    return added


def play_episode(model: NFSPModel, num_players: int, seed: int) -> tuple[int, float]:
    """One single-seat hand, mirroring train_nfsp_v2's loop."""
    learning_seat = int(np.random.randint(0, num_players))
    opponents = [
        RandomBot(seed=seed + i) if i % 2 == 0 else FlopBot()
        for i in range(num_players - 1)
    ]
    env = PokerEnv(
        num_players=num_players,
        learning_seat=learning_seat,
        opponent_bots=opponents,
        small_blind=25,
        big_blind=50,
        seed=seed,
        obs_version=2,
        bet_scheme=BetScheme.POT_FRACTION,
    )
    obs, _ = env.reset()
    model.begin_episode(learning_seat)

    decisions = 0
    reward = 0.0
    while True:
        mask = env.get_action_mask()
        current = np.asarray(obs)
        action = model.select_action(current, mask, training=True, seat=learning_seat)
        obs, reward, done, _, _ = env.step(action)
        if env.state is not None and env.state.action_on_seat == learning_seat:
            model.store_transition(
                current,
                action,
                reward if done else 0.0,
                np.asarray(obs),
                done,
                next_mask=mask,
            )
        model.train_step()
        decisions += 1
        if done or env.state is None:
            break
        if env.state.action_on_seat != learning_seat:
            break
        if decisions >= _MAX_STEPS:
            break
    return decisions, float(reward)


def evaluate(model: NFSPModel, hands: int, seed: int) -> tuple[float, float, float, int]:
    """Score the average policy against Random + Flop, one hand per deck."""
    model.q_network.eval()
    model.policy_network.eval()
    rewards: list[float] = []
    rng = np.random.RandomState(seed)
    for _ in range(hands):
        num_players = int(rng.randint(2, 7))
        seat = 0
        opponents = [
            RandomBot(seed=int(rng.randint(0, 2**31 - 1))) if i % 2 == 0 else FlopBot()
            for i in range(num_players - 1)
        ]
        env = PokerEnv(
            num_players=num_players,
            learning_seat=seat,
            opponent_bots=opponents,
            small_blind=25,
            big_blind=50,
            seed=int(rng.randint(0, 2**31 - 1)),
            obs_version=2,
            bet_scheme=BetScheme.POT_FRACTION,
        )
        obs, _ = env.reset()
        done = False
        while not done:
            action = model.select_action(obs, env.get_action_mask(), training=False)
            obs, reward, done, _, _ = env.step(action)
        rewards.append(float(reward))
    model.q_network.train()
    model.policy_network.train()
    return bb100_interval(rewards)


def main() -> None:
    p = argparse.ArgumentParser(description="Hybrid NFSP + SD-CFR training")
    p.add_argument("--teacher", type=str, default="models/sdcfr_v2/sdcfr_v2.pt",
                   help="SD-CFR v2 checkpoint used as the teacher")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Resume the NFSP learner from a checkpoint")
    p.add_argument("--model-name", type=str, default="hybrid")
    p.add_argument("--episodes", type=int, default=50_000)
    p.add_argument("--kappa", type=float, default=0.25,
                   help="Weight of the CFR distillation term (0 = plain NFSP)")
    p.add_argument("--shared-trunk", action="store_true",
                   help="Share one trunk between Q and policy heads")
    p.add_argument("--traversals-per-episode", type=int, default=2,
                   help="SD-CFR traversals run per training episode")
    p.add_argument("--players", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--train-every", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=5_000)
    p.add_argument("--eval-hands", type=int, default=60)
    p.add_argument("--checkpoint-every", type=int, default=5_000)
    p.add_argument("--no-validation", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if not 0.0 <= args.kappa <= 1.0:
        p.error("--kappa must be in [0, 1]")
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"[SEED] {args.seed}")

    models_dir = Path("models") / args.model_name
    models_dir.mkdir(parents=True, exist_ok=True)

    teacher = _load_teacher(Path(args.teacher))
    if teacher is None:
        raise SystemExit(
            f"Teacher not found: {args.teacher}\n"
            "Train one first:  python scripts/train_sdcfr_v2.py --iterations 400"
        )
    print(f"[TEACHER] {args.teacher} obs_v={teacher.obs_version} "
          f"iter={teacher.cfr_iteration}")

    model = NFSPModel(
        obs_version=2,
        bet_scheme=BetScheme.POT_FRACTION,
        eta=0.1,
        epsilon_start=0.06,
        epsilon_end=0.01,
        epsilon_decay_steps=300_000,
        epsilon_schedule="inv_sqrt",
        gamma=1.0,
        batch_size=args.batch_size,
        train_every=args.train_every,
        updates_per_train=2,
        q_lr=2.5e-4,
        policy_lr=1e-4,
        q_capacity=1_000_000,
        policy_capacity=1_000_000,
        target_sync_every=300,
        q_loss="huber",
        shared_trunk=args.shared_trunk,
        distill_kappa=args.kappa,
    )
    if args.checkpoint:
        print(f"[LOAD] {args.checkpoint}")
        model.load(args.checkpoint)
        model.distill_kappa = args.kappa

    print(
        f"[INIT] kappa={model.distill_kappa} shared_trunk={model.shared_trunk} "
        f"params={sum(x.numel() for x in model.q_network.parameters()):,} "
        f"obs_dim={int(model.q_network.input_dim)}"
    )

    actions = Counter()
    start = time.time()
    last_q: float | None = None
    last_p: float | None = None
    best_bb100 = -1e18

    for episode in range(args.episodes):
        decisions, _reward = play_episode(model, args.players, seed=episode)

        if args.kappa > 0.0 and episode % 25 == 0:
            rows, _used = collect_regret_targets(
                model, teacher, args.traversals_per_episode, args.players, seed=episode
            )
            actions["targets"] += rows

        result = model.train_step()
        if result["q_loss"] is not None:
            last_q = result["q_loss"]
        if result["policy_loss"] is not None:
            last_p = result["policy_loss"]

        if episode % 100 == 0:
            print(
                f"[{time.strftime('%H:%M:%S')}] Ep {episode:6d}: "
                f"Q_buf={len(model.q_buffer):7d} P_buf={len(model.policy_buffer):7d} "
                f"D_buf={len(model._distill_buffer) if model._distill_buffer else 0:6d} "
                f"Q_upd={model._q_updates:5d} P_upd={model._p_updates:5d} "
                f"Q_loss={last_q if last_q is None else round(last_q, 4)} "
                f"P_loss={last_p if last_p is None else round(last_p, 4)}",
                flush=True,
            )

        if (not args.no_validation) and episode and episode % args.eval_every == 0:
            mean, lower, upper, n = evaluate(model, args.eval_hands, _EVAL_SEED)
            print(f"  [EVAL] {mean:+.1f} BB/100 (95% [{lower:+.1f}, {upper:+.1f}]) n={n}")
            if mean > best_bb100:
                best_bb100 = mean
                model.save(str(models_dir / "hybrid_best.pt"),
                           training_meta={"episode": episode, "bb100": mean,
                                          "kappa": args.kappa})
                print(f"  [BEST] {mean:+.1f} BB/100 -> {models_dir / 'hybrid_best.pt'}")

        if episode and episode % args.checkpoint_every == 0:
            path = models_dir / f"hybrid_gen_{episode}.pt"
            model.save(str(path), training_meta={"episode": episode, "kappa": args.kappa})
            print(f"  [GEN] {path}")

    final = models_dir / "hybrid.pt"
    if (models_dir / "hybrid_best.pt").is_file():
        import shutil
        shutil.copy(models_dir / "hybrid_best.pt", final)
    else:
        model.save(str(final), training_meta={"kappa": args.kappa})
    elapsed = time.time() - start
    print(f"\n[DONE] {elapsed:.0f}s | kappa={args.kappa} | {final}")
    print(f"       distillation rows added: {actions['targets']}")


if __name__ == "__main__":
    main()
