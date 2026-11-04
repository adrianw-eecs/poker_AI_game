#!/usr/bin/env python
"""Validate NFSP/SDCFR training progress.

Reports performance in big blinds per 100 hands (BB/100) with 95% confidence
intervals. Retries hands where the learning agent never acts (e.g. blind walk).

Usage:
  python scripts/validate_training.py models/nfsp_selfplay.pt
  python scripts/validate_training.py models/sdcfr_selfplay.pt --type sdcfr
  python scripts/validate_training.py models/nfsp_gen_6.pt --hands 200
  python scripts/validate_training.py --compare models/nfsp_gen_5.pt models/nfsp_gen_10.pt models/nfsp_selfplay.pt
  python scripts/validate_training.py --hands 200 --seed 1 --bots nfsp:models/nfsp_production.pt sdcfr:models/sdcfr_prod/sdcfr_ckpt_500.pt random flop
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from poker.bots.call_bot import CallBot
from poker.bots.flop_bot import FlopBot
from poker.bots.nfsp_bot import NFSPBot
from poker.bots.random_bot import RandomBot
from poker.bots.sdcfr_bot import SDCFRBot
from poker.domain.action import Action
from poker.domain.card import Card
from poker.domain.hand import HandType
from poker.engine.action_validator import legal_actions
from poker.evaluation.evaluator import evaluate
from poker.ml.action_space import (
    BetScheme,
    action_index_to_action,
    action_to_action_index,
)
from poker.ml.env import REWARD_SCALE, PokerEnv
from poker.ml.models.nfsp_model import NFSPModel
from poker.ml.models.sdcfr_model import SDCFRModel
from poker.ml.model_analysis import ModelStatistics, extract_card_strength_from_obs
from poker.state.game_state import GameState

_ACTION_NAMES = ["FOLD", "CHECK/CALL", "RAISE_1", "RAISE_2", "RAISE_3", "RAISE_4", "RAISE_5"]
_LARGE_RAISE = {5, 6}


@dataclass
class _ChoiceLog:
    """Counts decisions that match the cards in front of the player."""

    decisions: int = 0
    sensible: int = 0
    examples: list[str] = field(default_factory=list)

    def snapshot(self) -> tuple[int, int, list[str]]:
        return self.decisions, self.sensible, list(self.examples)

    def restore(self, snap: tuple[int, int, list[str]]) -> None:
        self.decisions, self.sensible, self.examples = snap[0], snap[1], list(snap[2])

    def note(self, hole: tuple[Card, ...], board: tuple[Card, ...], action: int) -> None:
        if len(hole) != 2:
            return
        self.decisions += 1
        tier = _hand_tier(hole, board)
        reason = _choice_mismatch(tier, action)
        if reason is None:
            self.sensible += 1
            return
        if len(self.examples) < 3:
            cards = "".join(_card_str(card) for card in hole)
            street = "preflop" if not board else "".join(_card_str(card) for card in board)
            self.examples.append(f"{reason}: {cards} on {street} ({_ACTION_NAMES[action]})")


def _preflop_tier(hole: tuple[Card, ...]) -> str:
    high, low = sorted((int(hole[0].rank), int(hole[1].rank)), reverse=True)
    suited = hole[0].suit == hole[1].suit
    if high == low and high >= 10:
        return "premium"
    if high == 14 and low >= 12:
        return "premium"
    if high == low or (high == 14 and suited):
        return "playable"
    if suited and high - low == 1 and low >= 5:
        return "playable"
    return "trash"


def _hole_draw(hole: tuple[Card, ...], board: tuple[Card, ...]) -> str | None:
    """A draw that uses at least one hole card. Ace-high flush draw counts as nut."""
    cards = list(hole) + list(board)
    suit_counts: dict[object, int] = {}
    for card in cards:
        suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1
    for suit, count in suit_counts.items():
        if count < 4:
            continue
        hole_suited = [card for card in hole if card.suit == suit]
        if not hole_suited:
            continue
        if any(int(card.rank) == 14 for card in hole_suited):
            return "nut-draw"
        return "draw"
    ranks = {int(card.rank) for card in cards}
    hole_ranks = {int(card.rank) for card in hole}
    if 14 in ranks:
        ranks.add(1)
    if 14 in hole_ranks:
        hole_ranks.add(1)
    ordered = sorted(ranks)
    for start in ordered:
        window = [rank for rank in ordered if start <= rank <= start + 4]
        if len(set(window)) >= 4 and any(rank in hole_ranks for rank in window):
            return "draw"
    return None


def _postflop_tier(hole: tuple[Card, ...], board: tuple[Card, ...]) -> str:
    rank = evaluate([*hole, *board])
    board_top = max(int(card.rank) for card in board)
    if rank.type >= HandType.TWO_PAIR:
        return "strong"
    if rank.type == HandType.PAIR:
        pair_rank = rank.kickers[0]
        hole_ranks = {int(card.rank) for card in hole}
        if pair_rank > board_top or (pair_rank == board_top and pair_rank in hole_ranks):
            return "strong"
        return "medium"
    draw = _hole_draw(hole, board)
    if draw == "nut-draw":
        return "strong"
    if draw == "draw":
        return "medium"
    return "weak"


def _hand_tier(hole: tuple[Card, ...], board: tuple[Card, ...]) -> str:
    if not board:
        return _preflop_tier(hole)
    return _postflop_tier(hole, board)


def _choice_mismatch(tier: str, action: int) -> str | None:
    if tier in ("premium", "strong") and action == 0:
        return "folded a strong hand"
    if tier in ("trash", "weak") and action in _LARGE_RAISE:
        return "large bet with a weak hand"
    return None

_DEFAULT_STACK = 1000
_DEFAULT_SB = 25
_DEFAULT_BB = 50
_MAX_HAND_ATTEMPTS = 10
_TABLE_KINDS = {
    "nfsp": "nfsp",
    "nfsp_bot": "nfsp",
    "sdcfr": "sdcfr",
    "sdcfr_bot": "sdcfr",
    "random": "random",
    "random_bot": "random",
    "flop": "flop",
    "flop_bot": "flop",
}


@dataclass
class CellStats:
    rewards: list[float] = field(default_factory=list)
    skipped: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.rewards)

    def bb_per_hand(self, reward: float) -> float:
        stack_change = (reward / REWARD_SCALE) * _DEFAULT_STACK
        return stack_change / _DEFAULT_BB

    def mean_bb100(self) -> float:
        if not self.rewards:
            return 0.0
        return float(np.mean([self.bb_per_hand(r) for r in self.rewards])) * 100.0

    def stderr_bb100(self) -> float:
        if self.n < 2:
            return 0.0
        bb100 = np.array([self.bb_per_hand(r) for r in self.rewards]) * 100.0
        return float(np.std(bb100, ddof=1) / np.sqrt(self.n))

    def ci_bb100(self) -> tuple[float, float]:
        mean = self.mean_bb100()
        margin = 1.96 * self.stderr_bb100()
        return mean - margin, mean + margin


@dataclass
class SkipSummary:
    total_skipped: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def record(self, reason: str) -> None:
        self.total_skipped += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


def _paired_seed(base_seed: int, hand_offset: int, hand_idx: int, attempt: int) -> int:
    """Deterministic seed shared by every model in a comparison."""
    return (base_seed + hand_offset * 17 + hand_idx * 10007 + attempt * 104729) % (2**31 - 1)


def _build_opponent_list(num_players: int, config_name: str, opp_seed: int = 0) -> list:
    n = num_players - 1
    if config_name == "RandomBot":
        return [RandomBot(seed=(opp_seed + i * 97) % (2**31 - 1)) for i in range(n)]
    if config_name == "CallBot":
        return [CallBot() for _ in range(n)]
    if config_name == "FlopBot":
        return [FlopBot() for _ in range(n)]
    bots = []
    types = [RandomBot, FlopBot, CallBot]
    for i in range(n):
        bot_cls = types[i % len(types)]
        if bot_cls is RandomBot:
            bots.append(RandomBot(seed=(opp_seed + i * 97) % (2**31 - 1)))
        elif bot_cls is FlopBot:
            bots.append(FlopBot())
        else:
            bots.append(CallBot())
    return bots


def _eval_player_counts(min_n: int, max_n: int) -> list[int]:
    candidates = [2, 4, 6, 9]
    counts = sorted(set(n for n in candidates if min_n <= n <= max_n))
    if not counts:
        counts = [min_n]
    for n in (min_n, max_n):
        if n not in counts:
            counts = sorted(set(counts + [n]))
    return counts


def _skip_reason(env: PokerEnv) -> str | None:
    if env.state is None:
        return "null_state"
    if env.state.action_on_seat is None:
        return "won_without_acting"
    if env.state.action_on_seat != env.learning_seat:
        return "wrong_turn"
    return None


def _card_str(card: object) -> str:
    s = re.sub(r"\x1b\[[0-9;]*m", "", str(card))
    for sym, rep in [("♣", "c"), ("♦", "d"), ("♥", "h"), ("♠", "s")]:
        s = s.replace(sym, rep)
    return s


def _get_position_name(dealer_seat: int, learning_seat: int, num_players: int) -> str:
    """Get position name based on number of players."""
    relative_pos = (learning_seat - dealer_seat) % num_players

    # Position names depend on player count
    if num_players == 2:
        pos_names = {0: "Button", 1: "BB"}
    elif num_players == 4:
        pos_names = {0: "Button", 1: "SB", 2: "BB", 3: "UTG"}
    elif num_players == 5:
        pos_names = {0: "Button", 1: "SB", 2: "BB", 3: "UTG", 4: "HJ"}
    elif num_players == 6:
        pos_names = {0: "Button", 1: "SB", 2: "BB", 3: "UTG", 4: "HJ", 5: "CO"}
    elif num_players == 9:
        pos_names = {
            0: "Button", 1: "SB", 2: "BB", 3: "UTG", 4: "UTG+1",
            5: "UTG+2", 6: "HJ", 7: "CO", 8: "MP"
        }
    else:
        # Generic names for unsupported counts
        pos_names = {i: f"Pos{i}" for i in range(num_players)}

    return pos_names.get(relative_pos, f"Pos{relative_pos}")


def _play_hand(
    env: PokerEnv,
    select_action: Callable,
    stats: ModelStatistics | None = None,
) -> tuple[float, str, list[int], str | None]:
    """Play one hand. Returns (reward, hole_cards, actions, skip_reason).

    Optionally collects statistics if stats object provided.
    """
    obs, _ = env.reset()
    reason = _skip_reason(env)
    if reason is not None:
        return 0.0, "??", [0] * 7, reason

    hole_cards = "??"
    try:
        if env.state and env.state.players:
            player = env.state.players[env.learning_seat]
            if player.hole_cards and len(player.hole_cards) == 2:
                hole_cards = "".join(_card_str(c) for c in player.hole_cards)
    except Exception:
        pass

    actions = [0] * 7
    done = False
    reward = 0.0

    # Track position for stats (compute once at start)
    position_name = None
    if stats and env.state:
        position_name = _get_position_name(env.state.dealer_seat, env.learning_seat, env.num_players)

    while not done:
        mask = env.get_action_mask()
        action = select_action(obs, mask)
        actions[action] += 1

        # Collect stats if enabled
        if stats and position_name and env.state:
            street_name = env.state.street.value
            card_strength_idx = extract_card_strength_from_obs(obs)
            stats.record_action(action, street_name, position_name, card_strength_idx)

        obs, reward, done, _, _ = env.step(action)

    # Record final reward
    if stats and position_name:
        stats.record_reward(position_name, reward)

    return reward, hole_cards, actions, None


def _nfsp_env_kwargs(model: object) -> dict:
    """Match the table encoding to a loaded NFSP checkpoint."""
    version = int(getattr(model, "obs_version", 1))
    kwargs: dict = {"obs_version": version}
    scheme = getattr(model, "bet_scheme", None)
    if scheme is not None:
        kwargs["bet_scheme"] = scheme
    return kwargs


def _table_encoding(seated: list[_Seat]) -> dict:
    """Encoding the shared env must use, given every model at the table.

    One ``PokerEnv`` serves all seats, so a single encoding has to satisfy every
    checkpoint. Model seats are checked first and must agree; a disagreement is
    a configuration error rather than something to paper over, because a v1 bot
    cannot consume a v2 observation or vice versa.
    """
    versions: set[int] = set()
    schemes: set[object] = set()
    for seat in seated:
        model = seat.model
        if model is None:
            continue
        if seat.kind not in ("nfsp", "sdcfr"):
            continue
        versions.add(int(getattr(model, "obs_version", 1)))
        scheme = getattr(model, "bet_scheme", None)
        if scheme is not None:
            schemes.add(scheme)

    if len(versions) > 1:
        raise SystemExit(
            "Seated models need different observation encoders "
            f"(obs_version={sorted(versions)}). One table cannot serve both; "
            "run them in separate invocations."
        )
    if len(schemes) > 1:
        raise SystemExit(
            f"Seated models need different bet schemes ({sorted(s.value for s in schemes)}). "
            "Run them in separate invocations."
        )
    if not versions:
        return {}
    kwargs: dict = {"obs_version": versions.pop()}
    if schemes:
        kwargs["bet_scheme"] = schemes.pop()
    return kwargs


def _run_cell(
    num_players: int,
    config_name: str,
    num_hands: int,
    select_action: Callable,
    base_seed: int,
    hand_offset: int = 0,
    stats: ModelStatistics | None = None,
    env_kwargs: dict | None = None,
) -> tuple[CellStats, list[dict], list[int]]:
    """Run validation for one cell (player count + opponent config).

    Optionally collects detailed statistics if stats object provided.
    """
    cell = CellStats()
    top_wins: list[dict] = []
    cell_actions = [0] * 7

    for hand_idx in range(num_hands):
        played = False
        for attempt in range(_MAX_HAND_ATTEMPTS):
            # Same formula for every checkpoint so --compare plays the same decks.
            seed = _paired_seed(base_seed, hand_offset, hand_idx, attempt)
            opp_seed = _paired_seed(base_seed, hand_offset, hand_idx, attempt + 17)
            # Model (only at learning_seat) learns; all other seats are bots
            # Rotate learning_seat through all positions to test from all relative positions
            learning_seat = hand_idx % num_players
            # Increment hand_number at different rate than learning_seat to create
            # different relative positions (dealer_seat rotates differently)
            hand_number = hand_offset + (hand_idx * 2)
            env = PokerEnv(
                num_players=num_players,
                learning_seat=learning_seat,
                opponent_bots=_build_opponent_list(num_players, config_name, opp_seed),
                small_blind=_DEFAULT_SB,
                big_blind=_DEFAULT_BB,
                ante=0,
                seed=seed,
                **(env_kwargs or {}),
            )
            env.hand_number = hand_number

            reward, hole_cards, actions, reason = _play_hand(env, select_action, stats)
            if reason is not None:
                cell.skipped += 1
                cell.skip_reasons[reason] = cell.skip_reasons.get(reason, 0) + 1
                continue

            cell.rewards.append(reward)
            for i, count in enumerate(actions):
                cell_actions[i] += count
            if reward > 0:
                top_wins.append(
                    {"hole_cards": hole_cards, "opponent": f"{config_name}/{num_players}p", "reward": reward}
                )
            played = True
            break

        if not played:
            cell.skip_reasons["exhausted_retries"] = (
                cell.skip_reasons.get("exhausted_retries", 0) + 1
            )

    return cell, top_wins, cell_actions



def _merge_skip_summary(summary: SkipSummary, cell: CellStats) -> None:
    summary.total_skipped += cell.skipped
    for reason, count in cell.skip_reasons.items():
        summary.reasons[reason] = summary.reasons.get(reason, 0) + count


def test_nfsp(
    model_path: str,
    num_hands: int = 50,
    num_players: int = 9,
    rake_percent: float = 0.0,
    base_seed: int | None = None,
    collect_stats: bool = True,
) -> dict:
    model = NFSPModel()
    model.load(model_path)

    opp_configs = ["RandomBot", "CallBot", "FlopBot"]
    player_counts = _eval_player_counts(2, num_players)
    per_size_stats: dict[int, dict[str, CellStats]] = {}
    all_actions = [0] * 7
    all_rewards: list[float] = []
    winning_hands: list[dict] = []
    skip_summary = SkipSummary()
    if base_seed is None:
        base_seed = int(np.random.randint(0, 2**31 - 1))

    # Create stats collector if requested
    stats = ModelStatistics() if collect_stats else None

    def _select(obs, mask, _m=model):
        return _m.select_action(obs, mask, training=False)

    for n in player_counts:
        per_size_stats[n] = {}
        for config_name in opp_configs:
            cell, top_wins, actions = _run_cell(
                n, config_name, num_hands, _select, base_seed, hand_offset=n * 1000, stats=stats,
                env_kwargs=_nfsp_env_kwargs(model),
            )
            per_size_stats[n][config_name] = cell
            all_rewards.extend(cell.rewards)
            winning_hands.extend(top_wins)
            _merge_skip_summary(skip_summary, cell)
            for i, count in enumerate(actions):
                all_actions[i] += count

    winning_hands.sort(key=lambda x: x["reward"], reverse=True)
    total_actions = sum(all_actions)
    action_dist = {
        _ACTION_NAMES[i]: 100.0 * all_actions[i] / max(total_actions, 1) for i in range(7)
    }

    return {
        "per_size_stats": per_size_stats,
        "action_distribution": action_dist,
        "total_hands": len(all_rewards),
        "expected_hands": num_hands * len(opp_configs) * len(player_counts),
        "skip_summary": skip_summary,
        "win_rate": 100.0 * sum(1 for r in all_rewards if r > 0) / max(len(all_rewards), 1),
        "break_even": sum(1 for r in all_rewards if r == 0),
        "losses": sum(1 for r in all_rewards if r < 0),
        "wins": sum(1 for r in all_rewards if r > 0),
        "top_5_wins": winning_hands[:5],
        "num_players": num_players,
        "player_counts_tested": player_counts,
        "all_rewards": all_rewards,
        "base_seed": base_seed,
        "stats": stats,
    }


def _grand_bb100(results: dict) -> float:
    cells: list[CellStats] = []
    for opp_dict in results.get("per_size_stats", {}).values():
        cells.extend(opp_dict.values())
    if not cells:
        return 0.0
    return float(np.mean([c.mean_bb100() for c in cells if c.n > 0]))


def _pooled_bb100(results: dict) -> float:
    rewards = results.get("all_rewards", [])
    if not rewards:
        return 0.0
    return CellStats(rewards=rewards).mean_bb100()


def _cell_bb100(results: dict, n: int, opp: str) -> float | None:
    cell = results.get("per_size_stats", {}).get(n, {}).get(opp)
    if cell is None or cell.n == 0:
        return None
    return cell.mean_bb100()


def print_comparison_table(
    model_results: list[tuple[str, dict]],
    player_counts: list[int],
) -> None:
    """Side-by-side summary for multiple checkpoints on identical seeds."""
    print(f"\n{'='*90}")
    print("CHECKPOINT COMPARISON  (identical seeds per model)")
    print(f"{'='*90}")
    headers = [Path(p).name for p, _ in model_results]
    col_w = max(18, max(len(h) for h in headers) + 2)

    def _row(label: str, values: list[str]) -> None:
        print(f"  {label:<28}" + "".join(f"{v:>{col_w}}" for v in values))

    _row("Checkpoint", headers)
    print("  " + "-" * (28 + col_w * len(headers)))
    _row(
        "Hands played",
        [str(r["total_hands"]) for _, r in model_results],
    )
    _row(
        "Pooled BB/100",
        [f"{_pooled_bb100(r):+.1f}" for _, r in model_results],
    )
    _row(
        "Grand avg BB/100",
        [f"{_grand_bb100(r):+.1f}" for _, r in model_results],
    )
    _row(
        "FOLD %",
        [f"{r['action_distribution']['FOLD']:.1f}" for _, r in model_results],
    )
    _row(
        "RAISE %",
        [
            f"{sum(r['action_distribution'].get(n, 0) for n in _ACTION_NAMES[2:]):.1f}"
            for _, r in model_results
        ],
    )

    print("\n  Key cells (BB/100 mean):")
    print("  " + "-" * (28 + col_w * len(headers)))
    key_cells = [
        (2, "RandomBot"),
        (6, "RandomBot"),
        (6, "CallBot"),
        (9, "FlopBot"),
    ]
    for n, opp in key_cells:
        if n not in player_counts:
            continue
        label = f"{n}p vs {opp}"
        vals = []
        for _, r in model_results:
            bb = _cell_bb100(r, n, opp)
            vals.append(f"{bb:+.1f}" if bb is not None else "n/a")
        _row(label, vals)

    print(f"\n{'='*90}\n")


def test_sdcfr(
    model_path: str,
    num_hands: int = 50,
    num_players: int = 9,
    collect_stats: bool = True,
) -> dict:
    model = SDCFRModel()
    model.load(model_path)

    opp_configs = ["RandomBot", "FlopBot"]
    player_counts = _eval_player_counts(2, num_players)
    per_size_stats: dict[int, dict[str, CellStats]] = {}
    all_actions = [0] * 7
    all_rewards: list[float] = []
    skip_summary = SkipSummary()
    base_seed = int(np.random.randint(0, 2**31 - 1))

    # Create stats collector if requested
    stats = ModelStatistics() if collect_stats else None

    def _select(obs, mask, _m=model):
        strategy = _m.get_strategy(obs, mask.astype(float))
        return int(np.random.choice(7, p=strategy))

    for n in player_counts:
        per_size_stats[n] = {}
        for config_name in opp_configs:
            cell, _, actions = _run_cell(
                n, config_name, num_hands, _select, base_seed, hand_offset=n * 1000, stats=stats
            )
            per_size_stats[n][config_name] = cell
            all_rewards.extend(cell.rewards)
            _merge_skip_summary(skip_summary, cell)
            for i, count in enumerate(actions):
                all_actions[i] += count

    total_actions = sum(all_actions)
    action_dist = {
        _ACTION_NAMES[i]: 100.0 * all_actions[i] / max(total_actions, 1) for i in range(7)
    }

    return {
        "per_size_stats": per_size_stats,
        "action_distribution": action_dist,
        "total_hands": len(all_rewards),
        "expected_hands": num_hands * len(opp_configs) * len(player_counts),
        "skip_summary": skip_summary,
        "win_rate": 100.0 * sum(1 for r in all_rewards if r > 0) / max(len(all_rewards), 1),
        "break_even": sum(1 for r in all_rewards if r == 0),
        "losses": sum(1 for r in all_rewards if r < 0),
        "wins": sum(1 for r in all_rewards if r > 0),
        "num_players": num_players,
        "player_counts_tested": player_counts,
        "all_rewards": all_rewards,
        "cfr_iteration": model.cfr_iteration,
        "stats": stats,
    }


class _ActionCountBot:
    """Delegate to a bot and tally the discrete action it chooses."""

    def __init__(self, bot: object, counts: list[int], choices: _ChoiceLog | None = None) -> None:
        self._bot = bot
        self._counts = counts
        self._choices = choices
        self._name = bot.name

    @property
    def name(self) -> str:
        return self._name

    def act(self, state: GameState, legal: list[Action]) -> Action:
        action = self._bot.act(state, legal)
        seat = next(i for i, player in enumerate(state.players) if player.hole_cards)
        action_idx = action_to_action_index(action, state, seat)
        self._counts[action_idx] += 1
        if self._choices is not None:
            player = state.players[seat]
            self._choices.note(player.hole_cards, state.community_cards, action_idx)
        return action

    def observe_result(self, final_state: GameState, reward: float) -> None:
        observe = getattr(self._bot, "observe_result", None)
        if observe is not None:
            observe(final_state, reward)


@dataclass
class _Seat:
    """One player in a --bots table."""

    label: str
    kind: str
    path: Path | None = None
    model: NFSPModel | SDCFRModel | None = None


def _unique_label(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    n = 2
    while f"{base}-{n}" in used:
        n += 1
    label = f"{base}-{n}"
    used.add(label)
    return label


def _parse_bot_token(token: str, used: set[str]) -> _Seat:
    kind_token, sep, raw_path = token.partition(":")
    kind = _TABLE_KINDS.get(kind_token.strip().lower())
    if kind is None:
        known = ", ".join(sorted(set(_TABLE_KINDS)))
        raise ValueError(f"Unknown bot '{token}'. Expected one of: {known}")
    path_text = raw_path.strip() if sep else ""
    if kind in ("nfsp", "sdcfr"):
        if not path_text:
            raise ValueError(f"{kind} seat needs a checkpoint: {kind}:PATH")
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        return _Seat(_unique_label(path.stem, used), kind, path)
    if path_text:
        raise ValueError(f"{kind} does not take a checkpoint path")
    base = "Random" if kind == "random" else "Flop"
    return _Seat(_unique_label(base, used), kind)


def _parse_table(tokens: list[str]) -> list[_Seat]:
    if not 2 <= len(tokens) <= 9:
        raise ValueError(f"--bots needs 2 to 9 seats, got {len(tokens)}")
    used: set[str] = set()
    return [_parse_bot_token(token, used) for token in tokens]


def _load_table(seats: list[_Seat]) -> None:
    """Load each distinct checkpoint once and share it across seats."""
    nfsp_cache: dict[str, NFSPModel] = {}
    sdcfr_cache: dict[str, SDCFRModel] = {}
    for seat in seats:
        if seat.path is None:
            continue
        key = str(seat.path.resolve())
        if seat.kind == "nfsp":
            model = nfsp_cache.get(key)
            if model is None:
                print(f"Loading NFSP {seat.label} from {seat.path} ...")
                model = NFSPModel()
                model.load(key)
                nfsp_cache[key] = model
            seat.model = model
        else:
            model_sd = sdcfr_cache.get(key)
            if model_sd is None:
                print(f"Loading SDCFR {seat.label} from {seat.path} ...")
                model_sd = SDCFRModel()
                model_sd.load(key)
                sdcfr_cache[key] = model_sd
            seat.model = model_sd


def _rotate_seats(seats: list[_Seat], hand_idx: int) -> list[_Seat]:
    """Move the lineup so no checkpoint is stuck on one chair."""
    rot = hand_idx % len(seats)
    return seats[rot:] + seats[:rot]


def _matchup_bot(seat: _Seat, actions: list[int], choices: _ChoiceLog) -> _ActionCountBot:
    """One opponent. Every action is counted, including Random and Flop."""
    if seat.kind == "nfsp":
        inner = NFSPBot(name=seat.label, model=seat.model, training=False)
    elif seat.kind == "sdcfr":
        inner = SDCFRBot(name=seat.label, model=seat.model)
    elif seat.kind == "random":
        inner = RandomBot(name=seat.label, seed=int(np.random.randint(0, 2**31 - 1)))
    else:
        inner = FlopBot(name=seat.label)
    return _ActionCountBot(inner, actions, choices)


def _matchup_opponents(
    seated: list[_Seat],
    learning_seat: int,
    action_counts: dict[str, list[int]],
    choice_logs: dict[str, _ChoiceLog],
) -> list:
    return [
        _matchup_bot(seat, action_counts[seat.label], choice_logs[seat.label])
        for index, seat in enumerate(seated)
        if index != learning_seat
    ]


def _learning_action(env: PokerEnv, seat: _Seat, obs, mask) -> int:
    """Action for whichever player the environment is stepping."""
    if seat.kind == "nfsp":
        return seat.model.select_action(obs, mask, training=False)
    if seat.kind == "sdcfr":
        strategy = seat.model.get_strategy(obs, mask.astype(float))
        return int(np.random.choice(7, p=strategy))
    legal = legal_actions(env.state, env.learning_seat)
    if seat.kind == "random":
        bot = RandomBot(name=seat.label, seed=int(np.random.randint(0, 2**31 - 1)))
    else:
        bot = FlopBot(name=seat.label)
    action = bot.act(env.state.view_for(env.learning_seat), legal)
    # Convert the bot's concrete action back to an index the env will accept.
    # action_to_action_index assumes LEGACY bucketing, so on a POT_FRACTION
    # table it can return an index whose meaning differs; pick the legal index
    # whose concrete action matches instead.
    return _index_for_action(action, env.state, env.learning_seat, env.bet_scheme)


def _index_for_action(
    action: Action, state: GameState, seat: int, scheme: BetScheme
) -> int:
    """Index in the env's action space that reproduces ``action``.

    Prefers an exact (type, amount) match. Falls back to the kind of action so
    a size mismatch still produces a legal move rather than crashing.
    """
    for idx in range(7):
        try:
            candidate = action_index_to_action(idx, state, seat, scheme=scheme)
        except ValueError:
            continue
        if candidate.type == action.type and candidate.amount == action.amount:
            return idx
    for idx in range(7):
        try:
            candidate = action_index_to_action(idx, state, seat, scheme=scheme)
        except ValueError:
            continue
        if candidate.type == action.type:
            return idx
    # Nothing matched: check/call is the safest legal default.
    return 1


def _terminal_model_reward(env: PokerEnv, seat: int) -> float:
    """Stack-change reward using the same terminal formula as PokerEnv."""
    if env.state is None:
        return 0.0
    player = env.state.players[seat]
    stack_before = env.player_stacks_at_hand_start.get(seat, env.starting_stack)
    stack_change = player.stack - stack_before
    normalized = float(stack_change) / env.starting_stack
    return float(REWARD_SCALE * normalized)


def _hole_cards(env: PokerEnv, seat: int) -> str:
    if env.state is None:
        return "??"
    player = env.state.players[seat]
    if player.hole_cards and len(player.hole_cards) == 2:
        return "".join(_card_str(card) for card in player.hole_cards)
    return "??"


def _play_matchup_hand(
    env: PokerEnv,
    select_action: Callable,
    learner_actions: list[int],
    learner_choices: _ChoiceLog,
) -> str | None:
    """Play one hand. Returns a skip reason, or None when the hand finished."""
    obs, _ = env.reset()
    reason = _skip_reason(env)
    if reason is not None:
        return reason

    done = False
    while not done:
        mask = env.get_action_mask()
        action = select_action(obs, mask)
        learner_actions[action] += 1
        if env.state is not None:
            player = env.state.players[env.learning_seat]
            learner_choices.note(player.hole_cards, env.state.community_cards, action)
        obs, _, done, _, _ = env.step(action)
    return None


def test_bot_table(
    seats: list[_Seat],
    num_hands: int,
    base_seed: int | None,
) -> dict:
    """Play the --bots lineup. Every seat is scored."""
    if base_seed is not None:
        np.random.seed(base_seed)
    _load_table(seats)
    labels = [seat.label for seat in seats]
    rewards: dict[str, list[float]] = {label: [] for label in labels}
    action_counts: dict[str, list[int]] = {label: [0] * 7 for label in labels}
    choice_logs: dict[str, _ChoiceLog] = {label: _ChoiceLog() for label in labels}
    wins: dict[str, list[dict]] = {label: [] for label in labels}
    skip_summary = SkipSummary()

    for hand_idx in range(num_hands):
        played = False
        for _attempt in range(_MAX_HAND_ATTEMPTS):
            seated = _rotate_seats(seats, hand_idx)
            learning_seat = hand_idx % len(seated)
            learner = seated[learning_seat]
            seed = int(np.random.randint(0, 2**31 - 1))
            # The env's encoding drives every bot at the table, so it cannot be
            # taken from the learner alone: if the learner is `random`/`flop`
            # this was `{}`, the env defaulted to obs v1 (98-wide), and any
            # v2 checkpoint at the table then received a 98-wide observation
            # ("mat1 and mat2 shapes cannot be multiplied 1x91 and 311x256").
            env_kwargs = _table_encoding(seated)
            learner_encoding = (
                _nfsp_env_kwargs(learner.model) if learner.kind == "nfsp" and learner.model is not None else {}
            )
            # A model's own encoding always wins for that model; the table
            # encoding only fills in the case where no model specifies one.
            encoding = {**env_kwargs, **learner_encoding}
            env = PokerEnv(
                num_players=len(seated),
                learning_seat=learning_seat,
                opponent_bots=_matchup_opponents(
                    seated, learning_seat, action_counts, choice_logs
                ),
                small_blind=_DEFAULT_SB,
                big_blind=_DEFAULT_BB,
                ante=0,
                seed=seed,
                **encoding,
            )
            env.hand_number = hand_idx
            saved_counts = {label: counts.copy() for label, counts in action_counts.items()}
            saved_choices = {label: log.snapshot() for label, log in choice_logs.items()}

            def _select(obs, mask, _env=env, _learner=learner):
                return _learning_action(_env, _learner, obs, mask)

            reason = _play_matchup_hand(
                env, _select, action_counts[learner.label], choice_logs[learner.label]
            )
            if reason is not None:
                for label, counts in saved_counts.items():
                    action_counts[label][:] = counts
                for label, snap in saved_choices.items():
                    choice_logs[label].restore(snap)
                skip_summary.record(reason)
                continue

            for index, seat in enumerate(seated):
                reward = _terminal_model_reward(env, index)
                rewards[seat.label].append(reward)
                if reward > 0:
                    wins[seat.label].append(
                        {"hole_cards": _hole_cards(env, index), "reward": reward}
                    )
            played = True
            break

        if not played:
            skip_summary.record("exhausted_retries")

    def _pack(label: str) -> dict:
        role_rewards = rewards[label]
        actions = action_counts[label]
        total_actions = sum(actions)
        return {
            "rewards": role_rewards,
            "wins": sum(1 for reward in role_rewards if reward > 0),
            "losses": sum(1 for reward in role_rewards if reward < 0),
            "break_even": sum(1 for reward in role_rewards if reward == 0),
            "action_distribution": {
                _ACTION_NAMES[i]: 100.0 * actions[i] / max(total_actions, 1) for i in range(7)
            },
            "top_wins": sorted(wins[label], key=lambda hand: hand["reward"], reverse=True)[:3],
            "choices": choice_logs[label],
        }

    return {
        "lineup": labels,
        "hands_requested": num_hands,
        "hands_played": len(rewards[labels[0]]),
        "skip_summary": skip_summary,
        "players": {label: _pack(label) for label in labels},
    }


def _print_model_matchup_block(title: str, block: dict) -> None:
    stats = CellStats(rewards=block["rewards"])
    total = stats.n
    print(f"\n{title}")
    print("-" * 70)
    print(f"  {_format_bb100(stats)}")
    print(f"  Wins  (profit>0):  {block['wins']:5d}  ({100 * block['wins'] / max(total, 1):.1f}%)")
    print(f"  Even  (profit=0):  {block['break_even']:5d}  ({100 * block['break_even'] / max(total, 1):.1f}%)")
    print(f"  Losses(profit<0):  {block['losses']:5d}  ({100 * block['losses'] / max(total, 1):.1f}%)")
    ad = block["action_distribution"]
    raise_total = sum(ad.get(name, 0) for name in _ACTION_NAMES[2:])
    print(f"  FOLD:              {ad['FOLD']:6.2f}%")
    print(f"  CHECK/CALL:        {ad['CHECK/CALL']:6.2f}%")
    print(f"  RAISE (all sizes): {raise_total:6.2f}%")
    if block["top_wins"]:
        print("  Top winning hands:")
        for idx, hand in enumerate(block["top_wins"], 1):
            bb = (hand["reward"] / REWARD_SCALE) * _DEFAULT_STACK / _DEFAULT_BB
            print(f"    {idx}. {hand['hole_cards']:6s}  +{bb:.1f} bb")


def _print_table_summary(players: dict[str, dict]) -> None:
    """One row per seated player, best BB/100 first."""
    ranked = sorted(
        players.items(),
        key=lambda item: CellStats(rewards=item[1]["rewards"]).mean_bb100(),
        reverse=True,
    )
    print("\nTable summary (every seat):")
    print("-" * 78)
    print(
        f"  {'Player':<22} {'BB/100':>10} {'Win%':>7} {'Fold%':>7} "
        f"{'Call%':>7} {'Raise%':>8} {'Sense%':>8}"
    )
    for role, block in ranked:
        stats = CellStats(rewards=block["rewards"])
        ad = block["action_distribution"]
        choices: _ChoiceLog = block["choices"]
        raise_pct = sum(ad.get(name, 0.0) for name in _ACTION_NAMES[2:])
        win_pct = 100.0 * block["wins"] / max(stats.n, 1)
        sense_pct = 100.0 * choices.sensible / max(choices.decisions, 1)
        print(
            f"  {role:<22} {stats.mean_bb100():>+10.1f} {win_pct:6.1f}%"
            f" {ad['FOLD']:6.1f}% {ad['CHECK/CALL']:6.1f}% {raise_pct:7.1f}%"
            f" {sense_pct:7.1f}%"
        )
    leader, leader_block = ranked[0]
    leader_bb = CellStats(rewards=leader_block["rewards"]).mean_bb100()
    print(f"\n  Score: {leader} leads the table at {leader_bb:+.1f} BB/100.")
    print("\n  Choice notes (folded a strong hand, or a large bet with a weak one):")
    any_notes = False
    for role, block in ranked:
        examples = block["choices"].examples
        if not examples:
            continue
        any_notes = True
        print(f"    {role}: {'; '.join(examples)}")
    if not any_notes:
        print("    None.")


def print_matchup_report(results: dict) -> None:
    skip_summary: SkipSummary = results["skip_summary"]
    players: dict[str, dict] = results["players"]
    lineup = ", ".join(results["lineup"])
    print(f"\n{'='*78}")
    print(f"{len(results['lineup'])}-max table: {lineup}")
    print(f"{'='*78}")
    print("\nSample Coverage:")
    print("-" * 78)
    print(f"  Hands played:      {results['hands_played']} / {results['hands_requested']} requested")
    if skip_summary.total_skipped:
        print(f"  Skipped (retried): {skip_summary.total_skipped}")
        for reason, count in sorted(skip_summary.reasons.items()):
            print(f"    {reason:22s}: {count:4d}")

    _print_table_summary(players)
    print(f"{'='*78}\n")


def _format_bb100(stats: CellStats) -> str:
    mean = stats.mean_bb100()
    lo, hi = stats.ci_bb100()
    if stats.n < 2:
        return f"{mean:+.1f} BB/100  (n={stats.n})"
    return f"{mean:+.1f} BB/100  [95% CI: {lo:+.1f}, {hi:+.1f}]  (n={stats.n})"


def print_report(model_path: str, model_type: str, results: dict) -> None:
    total = results["total_hands"]
    expected = results.get("expected_hands", total)
    player_counts = results.get("player_counts_tested", [results["num_players"]])
    skip_summary: SkipSummary = results.get("skip_summary", SkipSummary())

    all_cells: list[CellStats] = []
    for opp_dict in results.get("per_size_stats", {}).values():
        all_cells.extend(opp_dict.values())
    grand_bb100 = (
        float(np.mean([c.mean_bb100() for c in all_cells if c.n > 0]))
        if all_cells else 0.0
    )

    print(f"\n{'='*70}")
    print(f"Validation Report: {Path(model_path).name}")
    print(f"Model Type: {model_type}  |  Player counts: {player_counts}")
    print(f"Reward display: BB/100  (env scale {REWARD_SCALE}x converted to big blinds)")
    if "cfr_iteration" in results:
        print(f"CFR Iteration: {results['cfr_iteration']}")
    print(f"{'='*70}")

    print("\nSample Coverage:")
    print("-" * 70)
    print(f"  Hands played:      {total} / {expected} requested")
    if skip_summary.total_skipped:
        print(f"  Skipped (retried): {skip_summary.total_skipped}")
        descriptions = {
            "won_without_acting": "agent never acted (fold-around / blind walk)",
            "null_state": "env state missing after reset",
            "wrong_turn": "action not on learning seat after reset",
            "exhausted_retries": "no actionable hand after max retries",
        }
        for reason, count in sorted(skip_summary.reasons.items()):
            print(f"    {reason:22s}: {count:4d}  ({descriptions.get(reason, reason)})")

    print("\nOverall Hand Results:")
    print("-" * 70)
    print(f"  Wins  (profit>0):  {results['wins']:5d}  ({100*results['wins']/max(total,1):.1f}%)")
    print(f"  Even  (profit=0):  {results['break_even']:5d}  ({100*results['break_even']/max(total,1):.1f}%)")
    print(f"  Losses(profit<0):  {results['losses']:5d}  ({100*results['losses']/max(total,1):.1f}%)")

    print("\nPerformance by Player Count and Opponent (BB/100):")
    print("-" * 70)
    for n in player_counts:
        opp_dict = results.get("per_size_stats", {}).get(n, {})
        if not opp_dict:
            continue
        cell_means = [s.mean_bb100() for s in opp_dict.values() if s.n > 0]
        size_avg = float(np.mean(cell_means)) if cell_means else 0.0
        tag = "[+]" if size_avg > 0 else ("[-]" if size_avg < -5 else "[~]")
        print(f"  {n:2d}-player  avg={size_avg:+.1f} BB/100  {tag}")
        for opp, stats in opp_dict.items():
            print(f"      vs {opp:12s}: {_format_bb100(stats)}")

    all_rewards = results.get("all_rewards", [])
    if all_rewards:
        pooled = CellStats(rewards=all_rewards)
        print(f"\n  Pooled (all cells):  {_format_bb100(pooled)}")
    print(f"  Grand avg (cells):   {grand_bb100:+.1f} BB/100")

    if results.get("top_5_wins"):
        print("\nTop 5 Winning Hands (bb won):")
        print("-" * 70)
        for idx, hand in enumerate(results["top_5_wins"], 1):
            bb = (hand["reward"] / REWARD_SCALE) * _DEFAULT_STACK / _DEFAULT_BB
            print(f"  {idx}. {hand['hole_cards']:6s} vs {hand['opponent']:20s}  +{bb:.1f} bb")

    print("\nAction Distribution (7 discrete actions):")
    print("-" * 70)
    ad = results["action_distribution"]
    raise_total = sum(ad.get(n, 0) for n in _ACTION_NAMES[2:])
    print(f"  FOLD:              {ad['FOLD']:6.2f}%")
    print(f"  CHECK/CALL:        {ad['CHECK/CALL']:6.2f}%")
    print(f"  RAISE (all sizes): {raise_total:6.2f}%")
    for name in _ACTION_NAMES[2:]:
        print(f"    {name:12s}:    {ad.get(name, 0):6.2f}%")

    print("\nDiagnosis:")
    print("-" * 70)
    fold_pct = ad["FOLD"]
    if fold_pct > 80:
        print("  [WARN] FOLD >80%: model may have collapsed to fold-everything.")
    elif fold_pct > 60:
        print("  [WARN] FOLD >60%: still early in training; expected to improve.")
    elif fold_pct < 10:
        print(f"  [WARN] FOLD {fold_pct:.1f}%: hyper-aggressive; exploitable by tight play.")
    elif raise_total > 65:
        print(f"  [WARN] RAISE {raise_total:.1f}%: very high aggression rate.")
    else:
        print(f"  [OK]  FOLD {fold_pct:.1f}% / RAISE {raise_total:.1f}%: balanced range.")

    if grand_bb100 > 0:
        print("  [OK]  Profitable overall — training is working.")
    elif grand_bb100 > -5:
        print("  [OK]  Near break-even — continue training.")
    else:
        print("  [WARN] Negative avg BB/100 — model needs more training or debugging.")

    # Print detailed statistics if available
    stats = results.get("stats")
    if stats is not None:
        print()
        stats.print_all()

    print(f"{'='*70}\n")


def _detect_model_type(path: Path, explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    return "SDCFR" if "sdcfr" in path.name.lower() else "NFSP"


def _run_one(
    model_path: Path,
    model_type: str,
    hands: int,
    players: int,
    rake: float,
    base_seed: int | None,
) -> dict:
    if model_type.upper() == "NFSP":
        return test_nfsp(str(model_path), hands, players, rake, base_seed=base_seed)
    if model_type.upper() == "SDCFR":
        return test_sdcfr(str(model_path), hands, players)
    raise ValueError(f"Unknown model type '{model_type}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate NFSP or SD-CFR model training progress"
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        help="Path to a single model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="MODEL",
        help="Compare multiple checkpoints on the same seeds (e.g. gen_5 gen_10 final)",
    )
    parser.add_argument(
        "--bots",
        nargs="+",
        metavar="BOT",
        help=(
            "Seat list for one table, 2-9 players. "
            "Tokens: nfsp:PATH, sdcfr:PATH, random, flop. "
            "Every seat is scored."
        ),
    )
    parser.add_argument("--hands", type=int, default=100, help="Test hands per opponent cell")
    parser.add_argument("--type", default=None, help="Model type: nfsp | sdcfr (auto-detect)")
    parser.add_argument(
        "--players", type=int, default=9,
        help="Maximum player count to test (always tests 2..players spread, default 9)",
    )
    parser.add_argument("--rake", type=float, default=0.0, help="Rake percentage")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base RNG seed for reproducible eval (default None for random; compare mode uses one seed for all)",
    )
    parser.add_argument(
        "--full-reports",
        action="store_true",
        help="With --compare, print full per-model reports after the summary table",
    )
    args = parser.parse_args()

    if args.bots:
        if args.model_path or args.compare:
            parser.error("--bots is a full table and cannot be combined with a model path or --compare")
        try:
            seats = _parse_table(args.bots)
            results = test_bot_table(seats, args.hands, args.seed)
            print_matchup_report(results)
        except Exception as exc:
            import traceback
            print(f"Error: {exc}")
            traceback.print_exc()
            sys.exit(1)
        return

    if not (2 <= args.players <= 9):
        parser.error("--players must be between 2 and 9")

    compare_paths = args.compare
    if compare_paths:
        paths = [Path(p) for p in compare_paths]
    elif args.model_path:
        paths = [Path(args.model_path)]
    else:
        parser.error("Provide model_path or --compare with one or more checkpoints")

    for path in paths:
        if not path.exists():
            print(f"Error: model not found at {path}")
            sys.exit(1)

    model_type = _detect_model_type(paths[0], args.type)
    print(
        f"\nConfig: {model_type}, up to {args.players} players, "
        f"{args.hands} hands/cell, seed={args.seed}, metrics in BB/100"
    )

    try:
        if len(paths) > 1:
            model_results: list[tuple[str, dict]] = []
            shared_seed = args.seed
            if shared_seed is None:
                shared_seed = int(np.random.randint(0, 2**31 - 1))
            print(f"  Paired seed: {shared_seed}")
            for path in paths:
                print(f"  Evaluating {path.name} ...", flush=True)
                results = _run_one(
                    path, model_type, args.hands, args.players, args.rake, shared_seed
                )
                model_results.append((str(path), results))
            player_counts = model_results[0][1].get(
                "player_counts_tested", [args.players]
            )
            print_comparison_table(model_results, player_counts)
            if args.full_reports:
                for path_str, results in model_results:
                    print_report(path_str, model_type.upper(), results)
        else:
            path = paths[0]
            print(f"Loading {model_type} model from {path} ...")
            results = _run_one(
                path, model_type, args.hands, args.players, args.rake, args.seed
            )
            print_report(str(path), model_type.upper(), results)
    except Exception as exc:
        import traceback
        print(f"Error: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
