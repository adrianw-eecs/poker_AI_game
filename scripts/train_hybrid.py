#!/usr/bin/env python
"""Train the NFSP + SD-CFR hybrid on tournament sittings.

One episode is one sitting: 2-9 seats, stacks carried forward, blinds that
escalate, busted seats staying out. SD-CFR labels states the learner visits.
Anticipatory mode is drawn once per hand. The n-step return is flushed at
the end of each hand and is not bootstrapped into the next one.

  python scripts/train_sdcfr_v2.py --iterations 400 --traversals 200
  python scripts/train_hybrid.py --teacher models/sdcfr_v2/sdcfr_v2.pt

  # mechanics smoke (checkpoints from this are not a population)
  python scripts/train_hybrid.py --episodes 2 --sitting-hand-cap 3 --kappa 0 --no-validation
"""

from __future__ import annotations

import argparse
import json
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
from poker.ml.env import REWARD_SCALE, PokerEnv
from poker.ml.gate import GateReport, evaluate_gate
from poker.ml.metrics import equity_share_delta
from poker.ml.models.nfsp_model import NFSPModel
from poker.ml.models.sdcfr_model import SDCFRModel
from poker.ml.roster import CheckpointRoster
from poker.ml.sitting import (
    SittingResult,
    SittingSpec,
    buyin_chips,
    draw_sitting,
    make_result,
)
from poker.state.game_state import Street

_MAX_STEPS = 200
_EVAL_SEED = 17_001


def _load_teacher(path: Path) -> SDCFRModel | None:
    if not path.is_file():
        return None
    teacher = SDCFRModel(obs_version=2)
    teacher.load(str(path))
    if teacher.obs_version not in (2, 5):
        raise SystemExit(
            f"{path} is an obs v{teacher.obs_version} checkpoint. The hybrid teacher "
            "must be obs v2 or v5. Retrain with scripts/train_sdcfr_v2.py."
        )
    teacher.advantage_network.eval()
    return teacher


def _load_previous_best(path: Path, obs_version: int) -> NFSPModel | None:
    """The frozen checkpoint the paired gate sits across from the learner."""
    if not path.is_file():
        return None
    previous = NFSPModel(obs_version=obs_version, bet_scheme=BetScheme.POT_FRACTION)
    try:
        previous.load(str(path))
    except (OSError, ValueError, RuntimeError, KeyError):
        return None
    if int(previous.obs_version) != int(obs_version):
        return None
    previous.q_network.eval()
    previous.policy_network.eval()
    return previous


def _print_gate(report: GateReport) -> None:
    print("  [GATE] baselines (selection is the minimum share lower bound)")
    for cell in report.baselines:
        print(
            f"    {cell.name}: win {cell.win_rate:.1%} "
            f"Wilson [{cell.win_low:.1%}, {cell.win_high:.1%}] "
            f"share {cell.share_mean:+.4f} [{cell.share_low:+.4f}, {cell.share_high:+.4f}] "
            f"bb/hand {cell.bb_per_hand:+.2f} n={cell.hands}"
        )
    if report.head_to_head:
        print("  [GATE] paired head-to-head")
        for cell in report.head_to_head:
            print(
                f"    {cell.name}: win {cell.win_rate:.1%} "
                f"Wilson [{cell.win_low:.1%}, {cell.win_high:.1%}] "
                f"share {cell.share_mean:+.4f} [{cell.share_low:+.4f}, {cell.share_high:+.4f}] "
                f"bb/hand {cell.bb_per_hand:+.2f} n={cell.hands}"
            )
    print(f"  [GATE] min baseline lower bound {report.selection_lower:+.4f}")


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

    # v3 and v4 still share the v2 prefix, money included, so a wider learner
    # can be sliced down to a v2 teacher. v5 changes what those money slots
    # mean, and slicing them would teach the policy the wrong game.
    teacher_width = int(teacher.advantage_network.input_dim)
    learner_version = int(getattr(model, "obs_version", 0))
    teacher_version = int(getattr(teacher, "obs_version", 0))
    if learner_version >= 5 or teacher_version >= 5:
        if obs_batch.shape[1] != teacher_width or learner_version != teacher_version:
            raise ValueError(
                f"Refusing to distill obs v{learner_version} ({obs_batch.shape[1]}-wide) "
                f"from an obs v{teacher_version} teacher ({teacher_width}-wide). "
                "Big-blind money is not the v2 prefix. Retrain with "
                "scripts/train_sdcfr_v2.py --obs-version 5."
            )
    elif obs_batch.shape[1] != teacher_width:
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


def baseline_opponents(num_opponents: int, rng: np.random.Generator) -> list:
    """Random and Flop, shuffled so a seat is not tied to one bot type."""
    bots: list = []
    for i in range(num_opponents):
        if i % 2 == 0:
            bots.append(RandomBot(name=f"Random_{i}", seed=int(rng.integers(0, 2**31 - 1))))
        else:
            bots.append(FlopBot(name=f"Flop_{i}"))
    rng.shuffle(bots)
    return bots


def _bot_label(bot: object) -> str:
    name = getattr(bot, "name", None)
    return str(name) if name else type(bot).__name__


def _live_count(env: PokerEnv) -> int:
    """Seats that still have chips, including chips committed in the current hand."""
    if env.state is None:
        return 0
    return sum(
        1
        for player in env.state.players
        if not player.is_eliminated and (player.stack + player.committed_this_hand) > 0
    )


def _stacks_now(env: PokerEnv) -> tuple[int, ...]:
    """Spendable stacks after a hand, or stack plus commitment if it is still live."""
    assert env.state is not None
    if env.state.street == Street.SHOWDOWN:
        return tuple(int(player.stack) for player in env.state.players)
    return tuple(int(player.stack + player.committed_this_hand) for player in env.state.players)


def play_sitting(
    model: NFSPModel,
    spec: SittingSpec,
    *,
    hero_seat: int,
    rng: np.random.Generator,
    teacher: SDCFRModel | None = None,
    opponents: list | None = None,
    roster: CheckpointRoster | None = None,
    distill_every_hands: int = 10,
    distill_flush_cap: int = 512,
    reward_exponent: float = 1.0,
    tendency_half_life: float | None = 40.0,
) -> tuple[SittingResult, dict[str, object]]:
    """Play one sitting. Anticipatory mode and the n-step window are per hand.

    Visited states are flushed to the teacher every ``distill_every_hands``
    rather than held until the sitting ends. Returns the sitting result and
    a log record (seeds, roster, decision count).
    """
    if opponents is None:
        if roster is not None:
            opponents = roster.seat(spec.field_size - 1, rng)
        else:
            opponents = baseline_opponents(spec.field_size - 1, rng)
    env = PokerEnv(
        num_players=spec.field_size,
        starting_stack=buyin_chips(),
        learning_seat=hero_seat,
        opponent_bots=opponents,
        seed=int(rng.integers(0, 2**31 - 1)),
        obs_version=int(getattr(model, "obs_version", 2)),
        bet_scheme=BetScheme.POT_FRACTION,
        blind_schedule=spec.schedule,
        hand_cap=spec.hand_cap,
        reward_exponent=reward_exponent,
        tendency_half_life=tendency_half_life,
    )
    roster: list[str] = []
    opponent_at = iter(opponents)
    for seat in range(spec.field_size):
        if seat == hero_seat:
            roster.append("learner")
        else:
            roster.append(_bot_label(next(opponent_at)))

    visited: list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]] = []
    seeds: list[int] = []
    decisions = 0
    distill_rows = 0
    hands_played = 0
    incomplete = False
    short_decisions = 0
    allin_decisions = 0
    last_q: float | None = None
    last_p: float | None = None

    def flush_distill() -> None:
        nonlocal distill_rows
        if teacher is None or model.distill_kappa <= 0.0 or not visited:
            visited.clear()
            return
        batch = visited
        if len(batch) > distill_flush_cap:
            pick = rng.choice(len(batch), size=distill_flush_cap, replace=False)
            batch = [visited[int(index)] for index in pick]
        distill_rows += distill_from_teacher(model, teacher, batch)
        visited.clear()

    env.hand_number = 0
    first_hand = True
    while hands_played < spec.hand_cap:
        deck_seed = int(rng.integers(0, 2**31 - 1))
        seeds.append(deck_seed)
        if first_hand:
            obs, _info = env.reset(
                options={
                    "stacks": spec.starting_stacks,
                    "deck_seed": deck_seed,
                    "dealer_seat": spec.button_seat,
                }
            )
            # A dealer override would pin the button for the rest of the sitting.
            env._dealer_override = None
            first_hand = False
        else:
            opened = env.start_next_hand(seed=deck_seed)
            if opened is None:
                break
            obs, _info = opened
        hands_played += 1
        # Per hand, not per sitting. NFSP's anticipatory draw is a hand-level choice.
        model.begin_episode(hero_seat)
        hand_over = (
            env.state is None
            or env.state.action_on_seat is None
            or env.state.street == Street.SHOWDOWN
        )
        if not hand_over:
            done = False
            steps = 0
            while not done and steps < _MAX_STEPS:
                if env.state is None or env.state.action_on_seat != hero_seat:
                    if env.state is not None and env.state.street != Street.SHOWDOWN:
                        done = bool(env._run_until_decision())
                    finished = (
                        done
                        or env.state is None
                        or env.state.street == Street.SHOWDOWN
                    )
                    if finished:
                        break
                    if (
                        env.state.action_on_seat == hero_seat
                        and env._seat_can_decide(hero_seat)
                    ):
                        continue
                    incomplete = True
                    break
                mask = env.get_action_mask()
                big_blind = max(env.state.config.big_blind, 1)
                hero_player = env.state.players[hero_seat]
                if hero_player.stack / big_blind <= 12:
                    short_decisions += 1
                facing_allin = any(
                    other.is_all_in and other.seat != hero_seat
                    for other in env.state.players
                ) or env.state.current_bet_to_call >= hero_player.stack + hero_player.committed_this_street
                if facing_allin and env.state.current_bet_to_call > hero_player.committed_this_street:
                    allin_decisions += 1
                current = np.asarray(obs)
                visited.append((current.copy(), np.asarray(mask, dtype=np.float32).copy()))
                action = model.select_action(current, mask, training=True, seat=hero_seat)
                obs, reward, done, _, _ = env.step(action)
                next_mask = (
                    np.ones(7, dtype=np.float32) if done else np.asarray(env.get_action_mask())
                )
                model.store_transition(
                    current,
                    action,
                    float(reward) if done else 0.0,
                    np.asarray(obs),
                    done,
                    next_mask=next_mask,
                    current_mask=np.asarray(mask, dtype=np.float32),
                    seat=hero_seat,
                )
                trained = model.train_step()
                if trained["q_loss"] is not None:
                    last_q = float(trained["q_loss"])
                if trained["policy_loss"] is not None:
                    last_p = float(trained["policy_loss"])
                decisions += 1
                steps += 1
            if steps >= _MAX_STEPS and not done:
                incomplete = True
        # Flush the n-step window before the next hand. The hand's chip
        # change is the whole return, so the next deal is a new episode.
        model.end_hand()
        if hands_played % max(distill_every_hands, 1) == 0:
            flush_distill()
        if incomplete or _live_count(env) < 2:
            break

    flush_distill()
    final_stacks = _stacks_now(env) if env.state is not None else spec.starting_stacks
    live_players = _live_count(env)
    result = make_result(
        final_stacks,
        hero_seat,
        hands_played,
        spec.hand_cap,
        live_players=live_players,
        incomplete=incomplete,
    )
    record: dict[str, object] = {
        "profile": spec.profile,
        "field": spec.field_size,
        "stacks": list(spec.starting_stacks),
        "button": spec.button_seat,
        "hero": hero_seat,
        "level_index": spec.level_index,
        "target_depth_bb": spec.target_depth_bb,
        "hands_per_level": spec.schedule.hands_per_level,
        "opening_blinds": [
            spec.schedule.levels[0].small,
            spec.schedule.levels[0].big,
            spec.schedule.levels[0].ante,
        ],
        "hand_cap": spec.hand_cap,
        "deck_seeds": seeds,
        "roster": roster,
        "hands_played": result.hands_played,
        "finish": result.finish_position,
        "final_stacks": list(result.final_stacks),
        "censored": result.censored,
        "decisions": decisions,
        "short_decisions": short_decisions,
        "allin_decisions": allin_decisions,
        "distill_rows": distill_rows,
        "q_loss": last_q,
        "p_loss": last_p,
    }
    return result, record


def evaluate(model: NFSPModel, hands: int, seed: int) -> tuple[float, float, float, int]:
    """Mean prize-share change against Random + Flop, one hand per deck.

    Kept as a diagnostic. Checkpoint selection uses ``evaluate_gate``.
    """
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
    return equity_share_delta([reward / REWARD_SCALE for reward in rewards])


def main() -> None:
    p = argparse.ArgumentParser(description="Hybrid NFSP + SD-CFR training")
    p.add_argument("--teacher", type=str, default=None,
                   help="SD-CFR checkpoint. v5 training requires an obs v5 teacher.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Resume the NFSP learner from a checkpoint")
    p.add_argument("--model-name", type=str, default="hybrid")
    p.add_argument("--episodes", type=int, default=50_000,
                   help="Number of sittings. Each sitting carries stacks until "
                        "one player remains or the hand cap.")
    p.add_argument("--kappa", type=float, default=0.25,
                   help="Weight of the CFR distillation term (0 = plain NFSP)")
    p.add_argument("--shared-trunk", action="store_true",
                   help="Share one trunk between Q and policy heads")
    p.add_argument("--traversals-per-episode", type=int, default=2,
                   help="(unused) kept for compatibility; distillation now queries "
                        "the teacher on the learner's own visited states")
    p.add_argument("--distill-every", type=int, default=None,
                   help="Deprecated alias of --distill-every-hands")
    p.add_argument("--distill-every-hands", type=int, default=10,
                   help="Flush visited states through the teacher every N hands")
    p.add_argument("--distill-flush-cap", type=int, default=512,
                   help="Maximum rows in one teacher flush")
    p.add_argument("--min-players", type=int, default=2)
    p.add_argument("--max-players", type=int, default=9)
    p.add_argument("--profiles", type=str, default="level_start,middle,end",
                   help="Comma-separated sitting profiles")
    p.add_argument("--profile-weights", type=str, default="0.4,0.4,0.2",
                   help="Comma-separated weights, one per --profiles entry")
    p.add_argument("--hands-per-level", type=str, default="6:10",
                   help="Inclusive range drawn once per sitting, e.g. 6:10")
    p.add_argument("--sitting-hand-cap", type=int, default=160,
                   help="Stop a sitting after this many hands if it has no winner")
    p.add_argument("--stop-min-episode", type=int, default=40_000,
                   help="Earliest sitting at which stall detection may stop training")
    p.add_argument("--stop-patience", type=int, default=10,
                   help="Evaluations without a better lower bound before stopping")
    p.add_argument("--updates-per-train", type=int, default=6,
                   help="Gradient updates per training step. More, smaller steps "
                        "on fresher data beats fewer large batches at this model size.")
    p.add_argument("--obs-version", type=int, default=3, choices=(2, 3, 4, 5),
                   help="Observation encoder: 2 (318), 3 (354), 4 (372), or 5 "
                        "(v4 width plus a tournament tail, money in big blinds). "
                        "A checkpoint from an earlier version is refused.")
    p.add_argument("--value-distribution", type=str, default="c51",
                   choices=("none", "c51"),
                   help="Q head: scalar ('none') or categorical distribution ('c51')")
    p.add_argument("--num-atoms", type=int, default=51)
    p.add_argument("--vmin", type=float, default=-1.5,
                   help="C51 support minimum. Prize-share rewards live near ±0.5.")
    p.add_argument("--vmax", type=float, default=1.5,
                   help="C51 support maximum. Prize-share rewards live near ±0.5.")
    p.add_argument("--reward-exponent", type=float, default=1.0,
                   help="beta on stack/total. 1 is winner-take-all. Leave it there "
                        "unless a run is punting; beta<1 is a different game.")
    p.add_argument("--tendency-half-life", type=float, default=40.0,
                   help="Hands for opponent-tendency counters to decay by half. "
                        "0 disables forgetting.")
    p.add_argument("--n-step", type=int, default=3,
                   help="Bootstrap every N transitions instead of every 1")
    p.add_argument("--reservoir-min-prob", type=float, default=0.25,
                   help="Exponential averaging floor for the average-policy "
                        "reservoir; 0 = uniform (drags the policy backward)")
    p.add_argument("--players", type=int, default=None,
                   help="If set, every sitting uses this field size")
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
    if args.min_players < 2 or args.max_players > 9 or args.min_players > args.max_players:
        p.error("--min-players and --max-players must satisfy 2 <= min <= max <= 9")
    if args.players is not None:
        if not 2 <= args.players <= 9:
            p.error("--players must be in 2..9")
        args.min_players = args.players
        args.max_players = args.players
    profiles = tuple(part.strip() for part in args.profiles.split(",") if part.strip())
    try:
        weights = tuple(float(part) for part in args.profile_weights.split(",") if part.strip())
    except ValueError:
        p.error("--profile-weights must be comma-separated numbers")
    if len(weights) != len(profiles):
        p.error("--profile-weights must have one number per profile")
    try:
        if ":" in args.hands_per_level:
            low_text, high_text = args.hands_per_level.split(":", 1)
            hands_per_level = (int(low_text), int(high_text))
        else:
            only = int(args.hands_per_level)
            hands_per_level = (only, only)
    except ValueError:
        p.error("--hands-per-level must look like 6:10")
    distill_every = (
        args.distill_every if args.distill_every is not None else args.distill_every_hands
    )
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"[SEED] {args.seed}")
    rng = np.random.default_rng(args.seed)

    models_dir = Path("models") / args.model_name
    models_dir.mkdir(parents=True, exist_ok=True)
    sitting_log = models_dir / "sittings.jsonl"

    if args.teacher is None:
        args.teacher = (
            "models/sdcfr_v5/sdcfr_v5.pt" if args.obs_version >= 5
            else "models/sdcfr_v2/sdcfr_v2.pt"
        )
    teacher = None
    if args.kappa > 0.0:
        teacher = _load_teacher(Path(args.teacher))
        if teacher is None:
            raise SystemExit(
                f"Teacher not found: {args.teacher}\n"
                "Train one first:  python scripts/train_sdcfr_v2.py --obs-version "
                f"{args.obs_version} --iterations 400"
            )
        if int(teacher.obs_version) != int(args.obs_version) and (
            int(args.obs_version) >= 5 or int(teacher.obs_version) >= 5
        ):
            raise SystemExit(
                f"Teacher {args.teacher} is obs v{teacher.obs_version} but this run "
                f"is obs v{args.obs_version}. v5 money is in big blinds, so a v2 "
                "teacher would label the wrong features. Retrain with "
                "scripts/train_sdcfr_v2.py --obs-version 5."
            )
        print(f"[TEACHER] {args.teacher} obs_v={teacher.obs_version} "
              f"iter={teacher.cfr_iteration}")
    else:
        print("[TEACHER] skipped because kappa is 0")

    model = NFSPModel(
        obs_version=args.obs_version,
        bet_scheme=BetScheme.POT_FRACTION,
        eta=0.1,
        epsilon_start=0.06,
        epsilon_end=0.01,
        epsilon_decay_steps=300_000,
        epsilon_schedule="inv_sqrt",
        gamma=1.0,  # hand is the episode; do not discount across a sitting
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
        if int(model.obs_version) != int(args.obs_version):
            raise SystemExit(
                f"Refusing to resume an obs v{model.obs_version} checkpoint "
                f"into an obs v{args.obs_version} run. Start a new directory "
                f"(models/hybrid_v{args.obs_version}/) instead of mixing widths."
            )
        model.distill_kappa = args.kappa

    print(
        f"[INIT] kappa={model.distill_kappa} shared_trunk={model.shared_trunk} "
        f"gamma={model.gamma} params={sum(x.numel() for x in model.q_network.parameters()):,} "
        f"obs_dim={int(model.q_network.input_dim)} "
        f"field={args.min_players}-{args.max_players} cap={args.sitting_hand_cap} "
        f"reward=prize_share beta={args.reward_exponent} "
        f"c51=[{args.vmin}, {args.vmax}] half_life={args.tendency_half_life}"
    )
    roster = CheckpointRoster(models_dir, args.obs_version, teacher)
    print(f"[ROSTER] {roster.refresh()} frozen checkpoint(s) available")

    actions = Counter()
    finishes: Counter[int] = Counter()
    short_decisions = 0
    allin_decisions = 0
    hero_decisions = 0
    start = time.time()
    last_q: float | None = None
    last_p: float | None = None
    best_lower = -1e18
    cell_best: dict[str, float] = {}
    seen_regression = False
    stalls = 0
    stop_reason = "sitting cap reached"
    half_life = None if args.tendency_half_life <= 0 else float(args.tendency_half_life)

    for episode in range(args.episodes):
        spec = draw_sitting(
            rng,
            min_players=args.min_players,
            max_players=args.max_players,
            profiles=profiles,
            weights=weights,
            hands_per_level=hands_per_level,
            hand_cap=args.sitting_hand_cap,
        )
        hero_seat = int(rng.integers(0, spec.field_size))
        _sitting, record = play_sitting(
            model,
            spec,
            hero_seat=hero_seat,
            rng=rng,
            teacher=teacher,
            roster=roster,
            distill_every_hands=distill_every,
            distill_flush_cap=args.distill_flush_cap,
            reward_exponent=args.reward_exponent,
            tendency_half_life=half_life,
        )
        record["sitting"] = episode
        actions["targets"] += int(record["distill_rows"])
        actions["hands"] += int(record["hands_played"])
        actions[str(record["profile"])] += 1
        finishes[int(record["finish"])] += 1
        short_decisions += int(record["short_decisions"])
        allin_decisions += int(record["allin_decisions"])
        hero_decisions += int(record["decisions"])
        if record["q_loss"] is not None:
            last_q = float(record["q_loss"])
        if record["p_loss"] is not None:
            last_p = float(record["p_loss"])
        with sitting_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        if episode < 5 or episode % 100 == 0:
            print(
                f"[{time.strftime('%H:%M:%S')}] Sit {episode:6d}: "
                f"{record['profile']} {record['field']}-max "
                f"bb={record['opening_blinds'][1]} ante={record['opening_blinds'][2]} "
                f"hands={record['hands_played']} finish={record['finish']} "
                f"censored={record['censored']} decisions={record['decisions']} "
                f"Q_buf={len(model.q_buffer):7d} P_buf={len(model.policy_buffer):7d} "
                f"D_buf={len(model._distill_buffer) if model._distill_buffer else 0:6d} "
                f"Q_upd={model._q_updates:5d} P_upd={model._p_updates:5d} "
                f"Q_loss={last_q if last_q is None else round(last_q, 4)} "
                f"P_loss={last_p if last_p is None else round(last_p, 4)}",
                flush=True,
            )

        if episode and episode % 1000 == 0 and hero_decisions:
            short_rate = short_decisions / hero_decisions
            allin_rate = allin_decisions / hero_decisions
            print(
                f"  [DIST] sittings={episode} short<=12bb={short_rate:.1%} "
                f"facing_allin={allin_rate:.1%} finishes={dict(sorted(finishes.items()))}",
                flush=True,
            )

        if (not args.no_validation) and episode and episode % args.eval_every == 0:
            previous = _load_previous_best(models_dir / "hybrid_best.pt", args.obs_version)
            report = evaluate_gate(model, previous, hands=args.eval_hands, seed=_EVAL_SEED)
            _print_gate(report)
            improved = False
            regressed = False
            for cell in report.baselines:
                prior = cell_best.get(cell.name)
                if prior is None or cell.share_low > prior:
                    cell_best[cell.name] = cell.share_low
                    improved = True
                elif cell.share_low < prior:
                    regressed = True
            selection = report.selection_lower
            if selection > best_lower:
                best_lower = selection
                stalls = 0
                model.save(
                    str(models_dir / "hybrid_best.pt"),
                    training_meta={
                        "episode": episode,
                        "selection_lower": selection,
                        "kappa": args.kappa,
                    },
                )
                roster.refresh()
                print(
                    f"  [BEST] min baseline lower bound {selection:+.4f} "
                    f"-> {models_dir / 'hybrid_best.pt'}"
                )
            else:
                if regressed:
                    seen_regression = True
                if seen_regression and not improved:
                    stalls += 1
                    print(f"  [STALL] no cell improvement for {stalls} gate run(s)")
                    if episode >= args.stop_min_episode and stalls >= args.stop_patience:
                        stop_reason = (
                            f"stall detection after {stalls} gate runs "
                            f"(patience {args.stop_patience})"
                        )
                        print(f"  [STOP] {stop_reason}")
                        break
                elif not improved:
                    print("  [GATE] no new cell best yet; patience has not started")

        if episode and episode % args.checkpoint_every == 0:
            path = models_dir / f"hybrid_gen_{episode}.pt"
            model.save(str(path), training_meta={"episode": episode, "kappa": args.kappa})
            roster.refresh()
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
    print(f"       hands={actions['hands']} profiles={dict(actions)}")
    print(f"       Q updates={model._q_updates}  P updates={model._p_updates}")
    print(f"       sitting log: {sitting_log}")


if __name__ == "__main__":
    main()
