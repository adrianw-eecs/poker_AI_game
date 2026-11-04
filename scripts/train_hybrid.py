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
import numpy.typing as npt
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker.bots.flop_bot import FlopBot
from poker.bots.random_bot import RandomBot
from poker.ml.action_space import BetScheme
from poker.ml.env import PokerEnv
from poker.ml.metrics import bb100_interval
from poker.ml.models.nfsp_model import NFSPModel
from poker.ml.models.sdcfr_model import SDCFRModel

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


def distill_from_teacher(
    model: NFSPModel,
    teacher: SDCFRModel,
    states: list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]],
    batch: int | None = None,
    rng: np.random.RandomState | None = None,
) -> int:
    """Store SD-CFR's strategy on states the learner actually reaches.

    The teacher is queried through its *network*, not through individual
    traversal buffer rows. A traversal row carries one nonzero advantage out of
    seven (outcome sampling), and regret-matching that collapses to a one-hot or
    uniform target that frequently disagrees with what SD-CFR actually plays.
    The network output is the teacher's averaged, low-variance strategy.

    The legal mask comes from the env that produced the state, so no mask is
    inferred. States are the learner's own, which keeps the distillation target
    on the distribution the policy will be evaluated on.

    Args:
        states: ``(obs, legal_mask)`` pairs collected during learner play.
        batch: How many states to query; ``None`` uses all of them.

    Returns:
        Number of (obs, advantages, mask) rows added to the distillation memory.
    """
    if not states:
        return 0
    subset = states
    if batch is not None and batch < len(states):
        pick = (rng or np.random).choice(len(states), batch, replace=False)
        subset = [states[i] for i in pick]

    obs_batch = np.stack([obs for obs, _ in subset]).astype(np.float32)
    mask_batch = np.stack([mk for _, mk in subset]).astype(np.float32)

    # The teacher may use an older encoder. v3 and v4 keep the v2 prefix
    # byte-identical (each version only appends), so slice rather than refusing
    # to distil.
    teacher_width = int(teacher.advantage_network.input_dim)
    if obs_batch.shape[1] != teacher_width:
        if obs_batch.shape[1] > teacher_width:
            obs_batch = obs_batch[:, :teacher_width]
        else:
            raise ValueError(
                f"learner obs is {obs_batch.shape[1]} wide but the teacher expects "
                f"{teacher_width}; cannot align the encoders"
            )

    teacher.advantage_network.eval()
    with torch.no_grad():
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=teacher.device)
        advantages = teacher.advantage_network(obs_t).cpu().numpy().astype(np.float32)

    added = 0
    for i in range(obs_batch.shape[0]):
        mask = mask_batch[i]
        if mask.sum() < 2:
            continue
        # Store the LEARNER's observation (full width) with the teacher's
        # advantages: the distillation loss runs through the learner's policy
        # network, so the row must match that network's input width.
        learner_obs = subset[i][0].astype(np.float32)
        model.add_regret_sample(learner_obs, advantages[i], mask, weight=1.0)
        added += 1
    return added


def play_episode(
    model: NFSPModel, num_players: int, seed: int
) -> tuple[int, float, list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]]:
    """One single-seat hand, mirroring train_nfsp_v2's loop.

    Returns ``(decisions, reward, visited)`` where ``visited`` holds the
    ``(obs, legal_mask)`` pair for each learner decision, so distillation can be
    queried on states the learner actually reaches.
    """
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
        obs_version=int(getattr(model, "obs_version", 2)),
        bet_scheme=BetScheme.POT_FRACTION,
    )
    obs, _ = env.reset()
    model.begin_episode(learning_seat)

    decisions = 0
    reward = 0.0
    # Teacher queries are made on the learner's own decision states, so the
    # distillation target stays on the policy's operating distribution.
    visited: list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]] = []
    while True:
        mask = env.get_action_mask()
        current = np.asarray(obs)
        visited.append((current.copy(), np.asarray(mask, dtype=np.float32).copy()))
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
    # One hand finished. The Q warm-up gate counts hands, not decisions.
    model.end_hand()
    return decisions, float(reward), visited


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
            obs_version=int(getattr(model, "obs_version", 2)),
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
                   help="(unused) kept for compatibility; distillation now queries "
                        "the teacher on the learner's own visited states")
    p.add_argument("--distill-every", type=int, default=25,
                   help="Distil the teacher's strategy every N episodes (default: 25)")
    p.add_argument("--stop-min-episode", type=int, default=40_000,
                   help="Earliest episode at which stall detection may stop training")
    p.add_argument("--stop-patience", type=int, default=10,
                   help="Evaluations without a better lower bound before stopping")
    p.add_argument("--updates-per-train", type=int, default=6,
                   help="Gradient updates per training step. More, smaller steps "
                        "on fresher data beats fewer large batches at this model size.")
    p.add_argument("--obs-version", type=int, default=3, choices=(2, 3, 4),
                   help="Observation encoder: 2 (318-wide), 3 (354-wide, adds "
                        "opponent tendencies + attention), or 4 (372-wide, adds "
                        "equity and pot structure). v2 checkpoints cannot load "
                        "into a later version; the v2 teacher still distils from "
                        "the shared prefix.")
    p.add_argument("--value-distribution", type=str, default="c51",
                   choices=("none", "c51"),
                   help="Q head: scalar ('none') or categorical distribution ('c51')")
    p.add_argument("--num-atoms", type=int, default=51)
    p.add_argument("--vmin", type=float, default=-10.0)
    p.add_argument("--vmax", type=float, default=10.0)
    p.add_argument("--n-step", type=int, default=3,
                   help="Bootstrap every N transitions instead of every 1")
    p.add_argument("--reservoir-min-prob", type=float, default=0.25,
                   help="Exponential averaging floor for the average-policy "
                        "reservoir; 0 = uniform (drags the policy backward)")
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
        obs_version=args.obs_version,
        bet_scheme=BetScheme.POT_FRACTION,
        eta=0.1,
        epsilon_start=0.06,
        epsilon_end=0.01,
        epsilon_decay_steps=300_000,
        epsilon_schedule="inv_sqrt",
        gamma=1.0,
        batch_size=args.batch_size,
        train_every=args.train_every,
        updates_per_train=args.updates_per_train,
        q_lr=2.5e-4,
        policy_lr=1e-4,
        q_capacity=1_000_000,
        policy_capacity=1_000_000,
        target_sync_every=300,
        q_loss="huber",
        shared_trunk=args.shared_trunk,
        distill_kappa=args.kappa,
        reservoir_min_prob=args.reservoir_min_prob,
        n_step=args.n_step,
        value_distribution=args.value_distribution,
        num_atoms=args.num_atoms,
        vmin=args.vmin,
        vmax=args.vmax,
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
    stalls = 0
    stop_reason = "episode cap reached"

    for episode in range(args.episodes):
        decisions, _reward, visited = play_episode(model, args.players, seed=episode)

        # Distil the teacher's strategy on the states just visited. Queried in
        # one batch per episode rather than per decision: one forward pass over
        # the teacher is far cheaper than one call per state, and the target is
        # the teacher's averaged policy rather than a single sampled row.
        if args.kappa > 0.0 and visited and episode % args.distill_every == 0:
            rows = distill_from_teacher(model, teacher, visited)
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
            # Select on the interval's lower bound, not the mean: at these
            # sample sizes the mean is noisy and picking on it rewards luck.
            if lower > best_bb100:
                best_bb100 = lower
                stalls = 0
                model.save(str(models_dir / "hybrid_best.pt"),
                           training_meta={"episode": episode, "bb100_lower": lower,
                                          "bb100_mean": mean, "kappa": args.kappa})
                print(f"  [BEST] lower bound {lower:+.1f} BB/100 -> {models_dir / 'hybrid_best.pt'}")
            else:
                stalls += 1
                print(f"  [STALL] no improvement for {stalls} eval(s)")
                if episode >= args.stop_min_episode and stalls >= args.stop_patience:
                    stop_reason = (
                        f"stall detection after {stalls} evals "
                        f"(patience {args.stop_patience})"
                    )
                    print(f"  [STOP] {stop_reason}")
                    break

        if episode and episode % args.checkpoint_every == 0:
            path = models_dir / f"hybrid_gen_{episode}.pt"
            model.save(str(path), training_meta={"episode": episode, "kappa": args.kappa})
            print(f"  [GEN] {path}")

    import shutil

    final = models_dir / "hybrid.pt"
    if (models_dir / "hybrid_best.pt").is_file():
        shutil.copy(models_dir / "hybrid_best.pt", final)
    else:
        model.save(str(final), training_meta={"kappa": args.kappa})
    elapsed = time.time() - start
    print(f"\n[DONE] {elapsed:.0f}s | kappa={args.kappa} | {final}")
    print(f"       stopped: {stop_reason}")
    print(f"       distillation rows added: {actions['targets']}")
    print(f"       Q updates={model._q_updates}  P updates={model._p_updates}")


if __name__ == "__main__":
    main()
