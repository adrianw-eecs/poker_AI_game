#!/usr/bin/env python
"""Training loop profiler - find the slowest component before optimising.

Runs a real training loop (including actual weight updates) and records the
wall-clock time spent in every distinct sub-component:

  build_observation()  - feature vector construction (called by env and bots)
  model.select_action  - agent neural-net forward pass
  {BotType}.act()      - each distinct opponent bot type
  env.step() [game]    - pure engine logic (step total minus bot + obs time)
  env.reset()          - game setup per hand (includes early bot/obs time)
  get_action_mask()    - legal-action mask construction
  store_transition()   - circular-buffer write
  train_step [Q]       - Q-network forward + backward + Adam step
  train_step [Policy]  - policy-network forward + backward + Adam step
  train_step [sync]    - target-network hard sync (every N steps)

Results are printed as a table sorted by % of total wall-clock, with an ASCII
bar chart so the bottleneck is immediately obvious.

USAGE:
  python scripts/profile_training.py                   # 300 eps, 2-9 players
  python scripts/profile_training.py --episodes 500 --players 4
  python scripts/profile_training.py --no-train        # skip weight updates
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------

class TimingStat:
    """Accumulates timing samples; uses reservoir sampling for percentiles."""

    _RESERVOIR = 4_000

    def __init__(self, name: str) -> None:
        self.name = name
        self.total_s: float = 0.0
        self.count: int = 0
        self.min_s: float = float("inf")
        self.max_s: float = 0.0
        self._reservoir: list[float] = []
        self._seen: int = 0

    def record(self, elapsed: float) -> None:
        self.total_s += elapsed
        self.count += 1
        if elapsed < self.min_s:
            self.min_s = elapsed
        if elapsed > self.max_s:
            self.max_s = elapsed
        # Reservoir sampling - O(1) memory.
        self._seen += 1
        if len(self._reservoir) < self._RESERVOIR:
            self._reservoir.append(elapsed)
        else:
            j = random.randint(0, self._seen - 1)
            if j < self._RESERVOIR:
                self._reservoir[j] = elapsed

    def _sorted_sample(self) -> list[float]:
        return sorted(self._reservoir)

    @property
    def mean_ms(self) -> float:
        return 1_000 * self.total_s / self.count if self.count else 0.0

    @property
    def median_ms(self) -> float:
        s = self._sorted_sample()
        if not s:
            return 0.0
        mid = len(s) // 2
        return 1_000 * (s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0)

    @property
    def p95_ms(self) -> float:
        s = self._sorted_sample()
        if not s:
            return 0.0
        return 1_000 * s[min(int(0.95 * len(s)), len(s) - 1)]

    @property
    def total_ms(self) -> float:
        return 1_000 * self.total_s

    def __repr__(self) -> str:
        return (f"TimingStat({self.name!r}, n={self.count}, "
                f"mean={self.mean_ms:.3f}ms, total={self.total_s:.3f}s)")


class Profiler:
    """Central registry of timing stats; used as a context manager."""

    def __init__(self) -> None:
        self._stats: dict[str, TimingStat] = {}
        # Per-step accumulators (zeroed at each env.reset / env.step call).
        self._step_bot_s: float = 0.0
        self._step_obs_s: float = 0.0
        self._in_reset: bool = False

    def stat(self, name: str) -> TimingStat:
        if name not in self._stats:
            self._stats[name] = TimingStat(name)
        return self._stats[name]

    def record(self, name: str, elapsed: float) -> None:
        self.stat(name).record(elapsed)

    def all_stats(self) -> dict[str, TimingStat]:
        return dict(self._stats)


# Global profiler (monkey-patches write into it).
_P = Profiler()


# ---------------------------------------------------------------------------
# Timed bot wrapper
# ---------------------------------------------------------------------------

class _TimedBot:
    """Wraps any Bot, recording act() time under "bot/{ClassName}"."""

    def __init__(self, bot: Any) -> None:
        self._bot = bot
        self._key = f"bot/{type(bot).__name__}"

    @property
    def name(self) -> str:
        return self._bot.name

    def act(self, state: Any, legal: Any) -> Any:
        t0 = time.perf_counter()
        result = self._bot.act(state, legal)
        elapsed = time.perf_counter() - t0
        _P.record(self._key, elapsed)
        _P._step_bot_s += elapsed
        return result

    def observe_result(self, *args: Any, **kwargs: Any) -> None:
        if hasattr(self._bot, "observe_result"):
            self._bot.observe_result(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bot, name)


# ---------------------------------------------------------------------------
# Monkey-patch build_observation
# ---------------------------------------------------------------------------

import poker.ml.observation as _obs_mod
_original_build_obs = _obs_mod.build_observation


def _timed_build_obs(state: Any, seat: int) -> Any:
    t0 = time.perf_counter()
    result = _original_build_obs(state, seat)
    elapsed = time.perf_counter() - t0
    _P.record("build_observation()", elapsed)  # key matches display_order entry
    _P._step_obs_s += elapsed
    return result


_obs_mod.build_observation = _timed_build_obs

# Patch all modules that may have already imported the symbol.
import poker.ml.env as _env_mod
import poker.ml.observation_v2 as _obs2_mod

_env_mod.build_observation = _timed_build_obs
_original_build_obs_v2 = _obs2_mod.build_observation_v2


def _timed_build_obs_v2(state: Any, seat: int) -> Any:
    t0 = time.perf_counter()
    result = _original_build_obs_v2(state, seat)
    elapsed = time.perf_counter() - t0
    _P.record("build_observation()", elapsed)
    _P._step_obs_s += elapsed
    return result


_obs2_mod.build_observation_v2 = _timed_build_obs_v2
_env_mod.build_observation_v2 = _timed_build_obs_v2


# ---------------------------------------------------------------------------
# Monkey-patch NFSPModel.train_step to get sub-component breakdown
# ---------------------------------------------------------------------------

from poker.ml.models.nfsp_model import NFSPModel


def _profiled_train_step(self: NFSPModel) -> dict[str, float | None]:
    """Time the real NFSP updates. Do not reimplement the losses here."""
    self._step += 1

    if self._step % self.train_every != 0:
        return {"q_loss": None, "policy_loss": None}

    cuda = self.device.type == "cuda"

    def _sync() -> None:
        if cuda:
            torch.cuda.synchronize()

    total_t0 = time.perf_counter()
    q_loss: float | None = None
    policy_loss: float | None = None
    repeats = max(1, int(getattr(self, "updates_per_train", 1)))

    for _ in range(repeats):
        if len(self.q_buffer) >= self._q_min:
            _sync()
            t0 = time.perf_counter()
            q_loss = self._update_q()
            _sync()
            _P.record("train_step/Q_update", time.perf_counter() - t0)

        if len(self.policy_buffer) >= self.batch_size:
            _sync()
            t0 = time.perf_counter()
            policy_loss = self._update_policy()
            _sync()
            _P.record("train_step/policy_update", time.perf_counter() - t0)

    _P.record("train_step() [total]", time.perf_counter() - total_t0)
    return {"q_loss": q_loss, "policy_loss": policy_loss}


NFSPModel.train_step = _profiled_train_step


# ---------------------------------------------------------------------------
# Main profiling loop
# ---------------------------------------------------------------------------

def _make_opponents(num_opponents: int, timed: bool) -> list:
    from poker.bots.flop_bot import FlopBot
    from poker.bots.random_bot import RandomBot
    bots = []
    for i in range(num_opponents):
        b = RandomBot(name=f"Random_{i}") if i % 2 == 0 else FlopBot(name=f"Flop_{i}")
        bots.append(_TimedBot(b) if timed else b)
    return bots


def run_profile(
    num_episodes: int = 300,
    min_players: int = 2,
    max_players: int = 9,
    do_train: bool = True,
    warmup: int = 30,
    batch_size: int = 64,
    v2: bool = False,
) -> None:
    from poker.ml.action_space import BetScheme
    from poker.ml.env import PokerEnv

    cuda = torch.cuda.is_available()
    device_label = torch.cuda.get_device_name() if cuda else "CPU"
    print(f"\n[Device] {device_label}")

    # Small capacity + batch_size so training fires well within the episode
    # budget. The policy buffer only receives best-response episodes (eta=0.1), so
    # we need policy_capacity modest enough that the threshold is reachable.
    q_cap = max(batch_size * 20, 2_000)
    p_cap = max(batch_size * 20, 2_000)
    model_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "q_capacity": q_cap,
        "policy_capacity": p_cap,
        "train_every": 4,  # train more often so we measure update time
    }
    if v2:
        model_kwargs.update(
            obs_version=2,
            bet_scheme=BetScheme.POT_FRACTION,
            gamma=1.0,
            q_lr=2.5e-4,
            policy_lr=1e-4,
            updates_per_train=2,
            epsilon_schedule="inv_sqrt",
            epsilon_start=0.06,
            epsilon_end=0.01,
        )
    model = NFSPModel(**model_kwargs)

    # --- Buffer pre-warm: fill both buffers past their minimum thresholds using
    # synthetic random transitions so train_step() fires during the timed run.
    # The policy buffer fills only at rate eta (~15%) so without this it would
    # need ~batch_size/0.15 ≈ 13 000 env steps to start training with batch=2048.
    q_threshold = model._q_min
    p_threshold = model.batch_size
    rng_pre = np.random.default_rng(0)
    print(f"[Pre-warm] filling Q-buffer to {q_threshold} and policy-buffer to {p_threshold} "
          f"with synthetic transitions ...", end=" ", flush=True)
    mask_all = np.ones(7, dtype=np.float32)
    obs_dim = int(model.q_network.input_dim)
    while len(model.q_buffer) < q_threshold or len(model.policy_buffer) < p_threshold:
        obs = rng_pre.standard_normal(obs_dim).astype(np.float32)
        next_obs = rng_pre.standard_normal(obs_dim).astype(np.float32)
        if v2:
            obs[:7] = rng_pre.integers(0, 52, size=7)
            next_obs[:7] = rng_pre.integers(0, 52, size=7)
        action = int(rng_pre.integers(0, 7))
        model.q_buffer.add(obs, action, 0.0, next_obs, False, next_mask=mask_all)
        model.policy_buffer.add(obs, action)
    print("done\n")

    total_episodes = warmup + num_episodes
    print(f"[Warmup] {warmup} episodes (discarded)")
    label = "v2 multi-seat" if v2 else "v1 single-seat"
    print(f"[Profile] {num_episodes} episodes across {min_players}-{max_players} players ({label})\n")
    profiled_hands = 0
    profiled_decisions = 0
    profiled_t0: float | None = None

    for episode in range(total_episodes):
        is_warmup = episode < warmup
        if not is_warmup and profiled_t0 is None:
            profiled_t0 = time.perf_counter()
        num_players = int(np.random.randint(min_players, max_players + 1))
        if v2:
            learning_seats = set(range(num_players))
            env = PokerEnv(
                num_players=num_players,
                learning_seats=learning_seats,
                opponent_bots=[],
                small_blind=25,
                big_blind=50,
                obs_version=2,
                bet_scheme=BetScheme.POT_FRACTION,
            )
            _P._step_bot_s = 0.0
            _P._step_obs_s = 0.0
            t0 = time.perf_counter()
            actor, obs, mask = env.reset_multi()
            reset_elapsed = time.perf_counter() - t0
            if not is_warmup:
                _P.record("env.reset() [total]", reset_elapsed)
                _P.record(
                    "env.reset() [game_logic]",
                    reset_elapsed - _P._step_bot_s - _P._step_obs_s,
                )
            for seat in learning_seats:
                model.begin_episode(seat)
            pending: dict[int, tuple[np.ndarray, int]] = {}
            done = False
            step_count = 0
            while not done and actor >= 0 and step_count < 200:
                if cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                action = model.select_action(obs, mask, training=True, seat=actor)
                if cuda:
                    torch.cuda.synchronize()
                if not is_warmup:
                    _P.record("model.select_action()", time.perf_counter() - t0)
                if actor in pending:
                    prev_obs, prev_action = pending[actor]
                    t0 = time.perf_counter()
                    model.store_transition(
                        prev_obs, prev_action, 0.0, obs, False,
                        next_mask=np.asarray(mask, dtype=np.float32),
                    )
                    if not is_warmup:
                        _P.record("store_transition()", time.perf_counter() - t0)
                pending[actor] = (np.array(obs, copy=True), int(action))
                _P._step_bot_s = 0.0
                _P._step_obs_s = 0.0
                t0 = time.perf_counter()
                actor, obs, mask, rewards, done = env.step_multi(action)
                step_elapsed = time.perf_counter() - t0
                if not is_warmup:
                    _P.record("env.step() [total]", step_elapsed)
                    game_logic = step_elapsed - _P._step_bot_s - _P._step_obs_s
                    if game_logic > 0:
                        _P.record("env.step() [game_logic]", game_logic)
                if do_train:
                    if cuda:
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    result = model.train_step()
                    if cuda:
                        torch.cuda.synchronize()
                    if not is_warmup and result["q_loss"] is not None:
                        _P.record("train_step() [total]", time.perf_counter() - t0)
                step_count += 1
                if done:
                    terminal = np.zeros(obs_dim, dtype=np.float32)
                    ones = np.ones(7, dtype=np.float32)
                    for seat, (prev_obs, prev_action) in pending.items():
                        t0 = time.perf_counter()
                        model.store_transition(
                            prev_obs, prev_action, float(rewards.get(seat, 0.0)),
                            terminal, True, next_mask=ones,
                        )
                        if not is_warmup:
                            _P.record("store_transition()", time.perf_counter() - t0)
            if not is_warmup and done:
                profiled_hands += 1
                profiled_decisions += step_count
            if not is_warmup and episode % 50 == warmup:
                pct_done = 100 * (episode - warmup + 1) / num_episodes
                print(
                    f"  profiled {episode - warmup + 1}/{num_episodes} episodes ({pct_done:.0f}%)...",
                    flush=True,
                )
            continue

        num_opponents = num_players - 1
        learning_seat = int(np.random.randint(0, num_players))

        opponents = _make_opponents(num_opponents, timed=not is_warmup)

        env = PokerEnv(
            num_players=num_players,
            learning_seat=learning_seat,
            opponent_bots=opponents,
            small_blind=25,
            big_blind=50,
        )

        # --- env.reset() ---
        _P._step_bot_s = 0.0
        _P._step_obs_s = 0.0
        t0 = time.perf_counter()
        obs, _ = env.reset()
        reset_elapsed = time.perf_counter() - t0
        if not is_warmup:
            _P.record("env.reset() [total]", reset_elapsed)
            # Obs built at end of reset() + bot actions within reset().
            _P.record("env.reset() [game_logic]",
                      reset_elapsed - _P._step_bot_s - _P._step_obs_s)

        # Skip episode if the hand ended before the learning agent got to act
        # (e.g. all other players folded pre-flop, no action required).
        if env.state is None or env.state.action_on_seat != learning_seat:
            continue

        model.begin_episode()
        done = False
        step_count = 0

        while not done and step_count < 200:
            # --- get_action_mask ---
            t0 = time.perf_counter()
            mask = env.get_action_mask()
            if not is_warmup:
                _P.record("get_action_mask()", time.perf_counter() - t0)

            # --- select_action ---
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            action = model.select_action(obs, mask, training=True)
            if cuda:
                torch.cuda.synchronize()
            if not is_warmup:
                _P.record("model.select_action()", time.perf_counter() - t0)

            # --- env.step() ---
            _P._step_bot_s = 0.0
            _P._step_obs_s = 0.0
            t0 = time.perf_counter()
            next_obs, reward, done, _, _ = env.step(action)
            step_elapsed = time.perf_counter() - t0
            if not is_warmup:
                step_bot = _P._step_bot_s
                step_obs = _P._step_obs_s
                _P.record("env.step() [total]", step_elapsed)
                game_logic = step_elapsed - step_bot - step_obs
                if game_logic > 0:
                    _P.record("env.step() [game_logic]", game_logic)

            next_mask = env.get_action_mask() if not done else np.ones(7, dtype=np.float32)
            q_reward = reward if done else 0.0

            # --- store_transition ---
            t0 = time.perf_counter()
            model.store_transition(obs, action, q_reward, next_obs, done, next_mask=next_mask)
            if not is_warmup:
                _P.record("store_transition()", time.perf_counter() - t0)

            # --- train_step ---
            if do_train:
                if cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                result = model.train_step()
                if cuda:
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                if not is_warmup and result["q_loss"] is not None:
                    _P.record("train_step() [total]", elapsed)

            obs = next_obs
            step_count += 1

        if not is_warmup and episode % 50 == warmup:
            pct_done = 100 * (episode - warmup + 1) / num_episodes
            print(f"  profiled {episode - warmup + 1}/{num_episodes} episodes ({pct_done:.0f}%)...",
                  flush=True)
        if not is_warmup and done:
            profiled_hands += 1
            profiled_decisions += step_count

    elapsed = 0.0 if profiled_t0 is None else time.perf_counter() - profiled_t0
    if elapsed > 0 and profiled_hands:
        print(
            f"[Throughput] {profiled_hands / elapsed:.2f} hands/s | "
            f"{profiled_decisions / elapsed:.1f} decisions/s "
            f"({profiled_hands} hands, {profiled_decisions} decisions)"
        )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

_BAR_WIDTH = 30


def _bar(pct: float) -> str:
    filled = int(round(_BAR_WIDTH * pct / 100.0))
    return "#" * filled + "." * (_BAR_WIDTH - filled)


def print_report(no_train: bool = False) -> None:
    stats = _P.all_stats()
    if not stats:
        print("No data recorded.")
        return

    # Total wall-clock = sum of env.step() + env.reset() totals
    # (using top-level totals to avoid double-counting sub-components).
    toplevel_keys = {
        "env.reset() [total]",
        "env.step() [total]",
        "model.select_action()",
        "get_action_mask()",
        "store_transition()",
        "train_step() [total]",
    }
    wall_clock = sum(
        stats[k].total_s for k in toplevel_keys if k in stats
    )
    if wall_clock == 0:
        print("No timing data - did the profiling loop run?")
        return

    # Build display rows (include sub-components too, indented).
    display_order = [
        # Top-level components
        ("env.reset() [total]",         "env.reset() - game setup + initial bots"),
        ("env.reset() [game_logic]",     "  +-- game logic only (setup, deal, blinds)"),
        ("env.step() [total]",           "env.step() - total per action"),
        ("env.step() [game_logic]",      "  +-- game logic only (state machine)"),
        ("build_observation()",          "build_observation() - feature vector"),
        ("model.select_action()",        "model.select_action() - agent inference"),
        ("get_action_mask()",            "get_action_mask() - legal-action mask"),
        ("store_transition()",           "store_transition() - buffer write"),
        ("train_step() [total]",         "train_step() - full weight update"),
        ("train_step/Q_update",          "  +-- Q-network forward + backward"),
        ("train_step/policy_update",     "  +-- policy-net forward + backward"),
        ("train_step/target_sync",       "  +-- target-network hard sync"),
    ]

    # Append bot types discovered during profiling.
    bot_keys = sorted(k for k in stats if k.startswith("bot/"))
    for k in bot_keys:
        display_order.append((k, f"  {k} - opponent decision"))

    # Header
    print(f"\n{'='*100}")
    print(" TRAINING LOOP PROFILER - RESULTS")
    print(f"{'='*100}")
    print(f"{'Component':<45} {'Count':>7} {'Mean':>9} {'Median':>9} {'P95':>9} "
          f"{'Total(s)':>9} {'% Wall':>7}  Bar")
    print(f"{'-'*45} {'-'*7} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*7}  {'-'*_BAR_WIDTH}")

    rows_for_sort: list[tuple[float, str, str]] = []

    for key, label in display_order:
        if key not in stats:
            continue
        s = stats[key]
        pct = 100.0 * s.total_s / wall_clock
        rows_for_sort.append((s.total_s, key, label))
        print(
            f"{label:<45} {s.count:>7,d} {s.mean_ms:>8.3f}ms {s.median_ms:>8.3f}ms "
            f"{s.p95_ms:>8.3f}ms {s.total_s:>9.3f}s {pct:>6.1f}%  {_bar(pct)}"
        )

    print(f"{'='*100}")

    # --- Ranked bottleneck summary ---
    print("\nBOTTLENECK RANKING (top-level components only, sorted by total time)")
    print("-" * 70)
    top = [
        (stats[k].total_s, label)
        for k, label in display_order
        if k in stats and not label.startswith("  ")  # skip sub-components
    ]
    top.sort(reverse=True)
    for rank, (total_s, label) in enumerate(top, 1):
        pct = 100.0 * total_s / wall_clock
        bar = _bar(pct)
        print(f"  #{rank:d}  {pct:5.1f}%  {bar}  {label.strip()}")

    # --- Derived observations ---
    print("\nDERIVED OBSERVATIONS")
    print("-" * 70)

    def _pct(k: str) -> float:
        return 100.0 * stats[k].total_s / wall_clock if k in stats else 0.0

    obs_pct = _pct("build_observation()")
    game_pct = _pct("env.step() [game_logic]")
    inference_pct = _pct("model.select_action()")
    q_pct = _pct("train_step/Q_update")
    p_pct = _pct("train_step/policy_update")

    notes: list[str] = []

    if obs_pct > 15:
        notes.append(
            f"build_observation() consumes {obs_pct:.0f}% of wall-clock. "
            "Optimise by caching static features (cards, position flags) and "
            "computing only the dynamic fields (pot, action counts, SPR) each call."
        )
    if game_pct > 25:
        notes.append(
            f"Pure game logic (env.step [game_logic]) is {game_pct:.0f}%. "
            "Profile poker.engine.* with cProfile for hot paths. "
            "Common culprits: hand evaluator calls, action history list appends."
        )
    if inference_pct > 20:
        notes.append(
            f"Agent inference is {inference_pct:.0f}%. "
            "Consider: smaller network, torch.compile(), or batch-collecting "
            "observations before the forward pass (environment vectorisation)."
        )
    if q_pct + p_pct > 15:
        notes.append(
            f"Weight updates (Q={q_pct:.0f}%, policy={p_pct:.0f}%) are significant. "
            "Already using GPU? If not, this is the highest-leverage target for a GPU. "
            "If on GPU: increase batch_size to better utilise GPU parallelism."
        )
    bot_total = sum(stats[k].total_s for k in bot_keys)
    bot_pct = 100.0 * bot_total / wall_clock
    if bot_pct > 10:
        notes.append(
            f"Opponent bots total {bot_pct:.0f}%. "
            "Replace slow bots with faster stubs during early training, or "
            "vectorise the environment so bots are batched."
        )
    if not notes:
        notes.append("No single dominant bottleneck - performance is well-balanced.")

    for i, note in enumerate(notes, 1):
        # Wrap at 95 chars
        words = note.split()
        lines: list[str] = []
        line = f"  {i}. "
        for w in words:
            if len(line) + len(w) + 1 > 97:
                lines.append(line)
                line = "     "
            line += w + " "
        lines.append(line)
        print("\n".join(lines))

    print(f"\n{'='*100}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the NFSP training loop")
    parser.add_argument("--episodes",    type=int, default=300,
                        help="Episodes to profile after warmup (default 300)")
    parser.add_argument("--warmup",      type=int, default=30,
                        help="Warmup episodes discarded before timing begins (default 30)")
    parser.add_argument("--min-players", type=int, default=2)
    parser.add_argument("--max-players", type=int, default=9)
    parser.add_argument("--no-train",    action="store_true",
                        help="Skip weight updates (profile inference + game only)")
    parser.add_argument("--batch-size",  type=int, default=64,
                        help="Batch size used for training (default 64, smaller fires sooner)")
    parser.add_argument("--v2", action="store_true",
                        help="Profile the v2 multi-seat loop (obs v2, pot-fraction bets)")
    args = parser.parse_args()

    if not (2 <= args.min_players <= args.max_players <= 9):
        parser.error("Require 2 <= min-players <= max-players <= 9")

    print(f"\nPROFILER CONFIG")
    print(f"  Episodes:    {args.episodes} (+ {args.warmup} warmup)")
    print(f"  Players:     {args.min_players}-{args.max_players}")
    print(f"  Weight upd:  {'disabled' if args.no_train else 'enabled (batch=' + str(args.batch_size) + ')'}")
    print(f"  Loop:        {'v2 multi-seat' if args.v2 else 'v1 single-seat'}")
    if torch.cuda.is_available():
        print(f"  GPU:         {torch.cuda.get_device_name()} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        print("  GPU:         none (CPU only)")

    t_wall = time.perf_counter()
    run_profile(
        num_episodes=args.episodes,
        min_players=args.min_players,
        max_players=args.max_players,
        do_train=not args.no_train,
        warmup=args.warmup,
        batch_size=args.batch_size,
        v2=args.v2,
    )
    wall = time.perf_counter() - t_wall
    print(f"\nTotal profiling time: {wall:.1f}s")

    print_report(no_train=args.no_train)


if __name__ == "__main__":
    main()
