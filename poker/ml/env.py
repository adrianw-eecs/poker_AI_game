"""Gymnasium-compatible poker environment for RL training.

The betting loop uses the same pure rules as ``play_hand``. ``PokerEnv`` keeps
the single-seat Gymnasium API (``learning_seat``). ``MultiSeatPokerEnv`` and
``reset_multi`` / ``step_multi`` expose one decision for every learning seat.
"""

from dataclasses import replace as dc_replace
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from poker.bots.base import Bot
from poker.bots.progress import chips_still_owed, ensure_progressing_action
from poker.bots.random_bot import RandomBot
from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.domain.action import Action
from poker.domain.deck import Deck
from poker.engine.action_validator import legal_actions, validate
from poker.engine.dealer import deal_flop, deal_hole_cards, deal_river, deal_turn
from poker.engine.rules import (
    advance_action,
    apply_action,
    blind_seats,
    count_active_players,
    first_preflop_actor,
    next_occupied,
    post_antes,
    post_blinds,
    reopens_betting,
    reset_street_action,
    round_is_closed,
)
from poker.engine.showdown import resolve
from poker.evaluation.evaluator import evaluate
from poker.exceptions import IllegalActionError
from poker.logging.logger import NullLogger
from poker.ml.action_space import (
    BetScheme,
    action_index_to_action,
    build_action_mask,
    cap_action,
)
from poker.ml.observation import OBS_DIM, build_observation
from poker.ml.observation_v2 import OBS_V2_DIM, build_observation_v2
from poker.ml.observation_v3 import OBS_V3_DIM, build_observation_v3
from poker.ml.observation_v4 import OBS_V4_DIM, build_observation_v4
from poker.ml.observation_v5 import OBS_V5_DIM, build_observation_v5
from poker.ml.tendencies import TendencyTracker
from poker.rng import RNG
from poker.state.game_state import GameState, Street
from poker.state.player_state import PlayerState
from poker.state.pot import Pot
from poker.state.pot_manager import rebuild_pots, return_uncalled_bet

# Training reward amplification (see docs/TRAINING.md). The reward is
# REWARD_SCALE times the change in winner-take-all equity (stack / chips in
# play). bb/hand is (reward / REWARD_SCALE) * chips_in_play / big_blind.
REWARD_SCALE = 0.5

_NEXT_STREET = {
    Street.PREFLOP: Street.FLOP,
    Street.FLOP: Street.TURN,
    Street.TURN: Street.RIVER,
}


class PokerEnv(gym.Env):
    """Single-seat Gymnasium wrapper around the engine betting rules.

    ``learning_seat`` is the seat whose observation ``reset`` and ``step``
    return. Other seats in ``learning_seats`` are also paused for a decision
    when using ``reset_multi`` / ``step_multi``; the Gymnasium methods still
    score ``learning_seat``.
    """

    metadata = {"render_modes": ["text"], "render_fps": 4}

    def __init__(
        self,
        num_players: int = 2,
        starting_stack: int = 1000,
        small_blind: int = 10,
        big_blind: int = 20,
        ante: int = 0,
        learning_seat: int = 0,
        learning_seats: set[int] | None = None,
        opponent_bots: list[Bot] | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
        max_bet: int = 0,
        stall_guard: bool = True,
        stall_counts: dict[str, int] | None = None,
        rake_percent: float = 0.0,
        rake_cap: int | None = None,
        run_it_twice: bool = False,
        bet_scheme: BetScheme = BetScheme.LEGACY,
        dedup_mask: bool = True,
        obs_version: int = 1,
        blind_schedule: BlindSchedule | None = None,
        hand_cap: int = 160,
        reward_exponent: float = 1.0,
        tendency_half_life: float | None = None,
    ) -> None:
        if num_players < 2 or num_players > 10:
            raise ValueError(f"num_players must be in [2, 10], got {num_players}")
        if starting_stack <= 0:
            raise ValueError(f"starting_stack must be positive, got {starting_stack}")
        if blind_schedule is not None:
            opening_level = blind_schedule.level_for_hand(0)
            small_blind = opening_level.small
            big_blind = opening_level.big
            ante = opening_level.ante
        if small_blind <= 0 or big_blind <= small_blind:
            raise ValueError("Blind amounts must be positive and ordered")
        if learning_seats is None:
            if learning_seat < 0 or learning_seat >= num_players:
                raise ValueError(
                    f"learning_seat must be in [0, {num_players - 1}], got {learning_seat}"
                )
            seats = {learning_seat}
        else:
            seats = set(learning_seats)
            if not seats or any(seat < 0 or seat >= num_players for seat in seats):
                raise ValueError(f"learning_seats must be a non-empty subset of seats, got {seats}")
            if learning_seat not in seats:
                learning_seat = min(seats)
        if max_bet < 0:
            raise ValueError(f"max_bet must be >= 0, got {max_bet}")
        if max_bet > 0 and max_bet < big_blind:
            raise ValueError(
                f"max_bet ({max_bet}) must be 0 or at least the big blind ({big_blind})"
            )
        if rake_percent < 0 or rake_percent > 10:
            raise ValueError(f"rake_percent must be in [0, 10], got {rake_percent}")
        if obs_version not in (1, 2, 3, 4, 5):
            raise ValueError(f"obs_version must be 1, 2, 3, 4 or 5, got {obs_version}")
        if hand_cap < 1:
            raise ValueError(f"hand_cap must be positive, got {hand_cap}")
        if reward_exponent <= 0:
            raise ValueError(f"reward_exponent must be positive, got {reward_exponent}")

        self.num_players = num_players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.ante = ante
        self.learning_seat = learning_seat
        self.learning_seats = seats
        self.max_bet = max_bet
        self.stall_guard = stall_guard
        self.stall_counts = stall_counts if stall_counts is not None else {}
        self._stall_key: tuple | None = None
        self._stall_repeats = 0
        self.rake_percent = rake_percent
        self.rake_cap = rake_cap
        self.bet_scheme = bet_scheme
        self.dedup_mask = dedup_mask
        self.obs_version = obs_version
        self.hand_cap = hand_cap
        self.reward_exponent = float(reward_exponent)
        self.tendency_half_life = tendency_half_life
        self._obs_dim = {
            1: OBS_DIM,
            2: OBS_V2_DIM,
            3: OBS_V3_DIM,
            4: OBS_V4_DIM,
            5: OBS_V5_DIM,
        }[obs_version]
        # v3 and v4 append rolling opponent tendencies, which persist across
        # hands within one sitting. Reset by reset_multi / start_next_hand_multi.
        self.tendencies = (
            TendencyTracker(num_players=num_players, half_life=tendency_half_life)
            if obs_version >= 3
            else None
        )

        self.config = GameConfig(
            num_players=num_players,
            starting_stack=starting_stack,
            small_blind=small_blind,
            big_blind=big_blind,
            ante=ante,
            rake_percent=rake_percent,
            rake_cap=rake_cap,
            blind_schedule=(
                blind_schedule
                if blind_schedule is not None
                else BlindSchedule(
                    levels=[BlindLevel(small_blind, big_blind, ante)],
                    hands_per_level=1000,
                    fixed=True,
                )
            ),
            run_it_twice=run_it_twice,
        )
        self.blind_level = BlindLevel(small_blind, big_blind, ante)

        opponent_seats = [seat for seat in range(num_players) if seat not in self.learning_seats]
        if opponent_bots is None:
            opponent_bots = [RandomBot(seed=seed) for _ in opponent_seats]
        if len(opponent_bots) != len(opponent_seats):
            raise ValueError(
                f"opponent_bots must have {len(opponent_seats)} bots, got {len(opponent_bots)}"
            )
        self.opponent_bots = opponent_bots
        self._bot_for_seat = dict(zip(opponent_seats, opponent_bots))

        self.action_space = spaces.Discrete(7)
        # Card ids occupy 0..52. v5 keeps those slots, so the box high stays 52.
        obs_high = 52.0 if obs_version in (2, 4, 5) else 1.0
        self.observation_space = spaces.Box(
            low=0.0, high=obs_high, shape=(self._obs_dim,), dtype=np.float32
        )

        self.state: GameState | None = None
        self.hand_number = 0
        self.player_stacks_at_hand_start: dict[int, int] = {}
        self._players_acted: set[int] = set()
        self._max_commitment = 0
        self.prev_equity: float = 0.5
        self.render_mode = render_mode
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.logger = NullLogger()
        self.current_hand_deck: Deck | None = None
        self._deck_seed_override: int | None = None
        self._dealer_override: int | None = None

    def _prepare_rng(self, seed: int | None) -> None:
        if seed is not None:
            self.rng = np.random.Generator(np.random.PCG64(seed))

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        """Reset to a new hand and return the learning seat's observation."""
        super().reset(seed=seed)
        self._prepare_rng(seed)
        self.reset_multi(options)
        obs = self._current_obs()
        return obs, self._get_info()

    def reset_multi(
        self, options: dict[str, Any] | None = None
    ) -> tuple[int, npt.NDArray[np.float32], npt.NDArray[np.int32]]:
        """Deal a hand and return ``(seat, obs, mask)`` for the next learner."""
        self._deck_seed_override = None if options is None else options.get("deck_seed")
        self._dealer_override = None if options is None else options.get("dealer_seat")
        opening = None if options is None else options.get("stacks")
        stacks = None if opening is None else tuple(int(stack) for stack in opening)
        return self._begin_hand(stacks)

    def start_next_hand_multi(
        self, seed: int | None = None
    ) -> tuple[int, npt.NDArray[np.float32], npt.NDArray[np.int32]] | None:
        """Deal the next hand and return ``(seat, obs, mask)`` for the next learner.

        Stacks carry and the button moves to the next live seat. A seat with
        no chips is eliminated and stays out. Returns None when fewer than
        two live seats remain, which ends the sitting. A stack below the big
        blind stays in and posts a short all-in blind.
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        self._mark_busted()
        live = [player for player in self.state.players if not player.is_eliminated]
        if len(live) < 2:
            return None
        previous_button = self.state.dealer_seat
        stacks = tuple(player.stack for player in self.state.players)
        live_mask = tuple(not player.is_eliminated for player in self.state.players)
        self.hand_number += 1
        self._prepare_rng(seed)
        return self._begin_hand(
            stacks,
            live_mask=live_mask,
            button_from=(previous_button + 1) % self.num_players,
        )

    def start_next_hand(
        self, seed: int | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]] | None:
        """Deal another hand from the stacks left after the previous one.

        The button moves to the next live seat. Returns None when fewer than
        two seats still have chips. ``seed`` redraws the deck.
        """
        opened = self.start_next_hand_multi(seed)
        if opened is None:
            return None
        return self._current_obs(), self._get_info()

    def step(
        self, action: int
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Apply one learning-seat action and return the Gymnasium step tuple."""
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self.state.action_on_seat != self.learning_seat:
            raise RuntimeError(
                f"Not learning agent's turn. Expected seat {self.learning_seat}, "
                f"action on seat {self.state.action_on_seat}"
            )
        _seat, obs, _mask, _rewards, done = self.step_multi(action)
        reward = self._compute_reward(self.learning_seat)
        if done:
            obs = np.zeros(self._obs_dim, dtype=np.float32)
        return obs, reward, done, False, self._get_info()

    def step_multi(
        self, action: int
    ) -> tuple[int, npt.NDArray[np.float32], npt.NDArray[np.int32], dict[int, float], bool]:
        """Apply ``action`` for the seat to act and advance to the next learner.

        Returns ``(seat, obs, mask, rewards_by_seat, done)``. On a finished
        hand ``seat`` is -1 and the observation and mask are zeros. Rewards are
        each seat's scaled chip delta at the end of the hand, and zeros before that.
        """
        if self.state is None or self.state.action_on_seat is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        seat = self.state.action_on_seat
        if seat not in self.learning_seats:
            raise RuntimeError(
                f"Not a learning seat's turn. Action is on seat {seat}, "
                f"learning seats are {sorted(self.learning_seats)}"
            )
        try:
            concrete = action_index_to_action(
                action,
                self.state,
                seat,
                max_bet=self.max_bet,
                scheme=self.bet_scheme,
            )
            concrete = cap_action(self.state, seat, concrete, self.max_bet)
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Invalid action {action}: {exc}") from exc
        try:
            validate(self.state, seat, concrete)
        except IllegalActionError as exc:
            raise ValueError(f"Action {action} is illegal: {exc}") from exc

        self._commit(seat, concrete)
        done = self._run_until_decision()
        if done:
            rewards = self._terminal_rewards()
            return (
                -1,
                np.zeros(self._obs_dim, dtype=np.float32),
                np.zeros(7, dtype=np.int32),
                rewards,
                True,
            )
        assert self.state is not None and self.state.action_on_seat is not None
        actor = self.state.action_on_seat
        return actor, self._observe(actor), self.get_action_mask(), self._zero_rewards(), False

    def _apply_blind_level(self) -> None:
        """Post the schedule's blinds for ``hand_number``.

        ``rules.py`` reads ``state.config`` for the blind and ante amounts, so
        the level has to be copied onto the config. A fixed one-level schedule
        leaves the config object unchanged.
        """
        level = self.config.blind_schedule.level_for_hand(self.hand_number)
        self.blind_level = level
        self.small_blind = level.small
        self.big_blind = level.big
        self.ante = level.ante
        self.config = self.config.with_blinds(level.small, level.big, level.ante)

    def _mark_busted(self) -> None:
        """Mark every empty stack eliminated so the next hand cannot reseat it."""
        assert self.state is not None
        players = tuple(
            player.with_stack(0).with_eliminated(True)
            if player.stack <= 0 or player.is_eliminated
            else player
            for player in self.state.players
        )
        self.state = dc_replace(self.state, players=players)

    def _begin_hand(
        self,
        stacks: tuple[int, ...] | None,
        live_mask: tuple[bool, ...] | None = None,
        button_from: int | None = None,
    ) -> tuple[int, npt.NDArray[np.float32], npt.NDArray[np.int32]]:
        self._apply_blind_level()
        players_list: list[PlayerState] = []
        for seat in range(self.num_players):
            stack = self.starting_stack if stacks is None else int(stacks[seat])
            if live_mask is None:
                eliminated = stack <= 0
            else:
                eliminated = (not live_mask[seat]) or stack <= 0
            players_list.append(
                PlayerState(
                    seat=seat,
                    name=f"Player{seat}",
                    stack=0 if eliminated else stack,
                    hole_cards=(),
                    committed_this_street=0,
                    committed_this_hand=0,
                    has_folded=False,
                    is_all_in=False,
                    is_eliminated=eliminated,
                )
            )
        players = tuple(players_list)
        self.state = GameState(
            hand_number=self.hand_number,
            street=Street.PREFLOP,
            dealer_seat=0,
            players=players,
            community_cards=(),
            pots=[Pot(0, frozenset(range(self.num_players)))],
            current_bet_to_call=0,
            last_raise_size=0,
            action_history_this_street=[],
            action_history_this_hand=[],
            deck_remaining_count=52,
            config=self.config,
            blind_level=self.blind_level,
            action_on_seat=None,
        )
        if self._dealer_override is not None:
            button_start = int(self._dealer_override)
        elif button_from is not None:
            button_start = button_from
        else:
            button_start = self.hand_number % self.num_players
        self.state = dc_replace(self.state, dealer_seat=next_occupied(self.state, button_start))
        self.player_stacks_at_hand_start = {player.seat: player.stack for player in players}
        if self.config.ante > 0:
            self.state = post_antes(self.state)
        sb_seat, bb_seat = blind_seats(self.state)
        self.state = post_blinds(self.state, sb_seat, bb_seat)
        self.state = rebuild_pots(self.state)

        if self._deck_seed_override is not None:
            deck_seed = int(self._deck_seed_override)
        else:
            deck_seed = int(self.rng.integers(0, 2**31 - 1))
        self.current_hand_deck = Deck(rng=RNG(seed=deck_seed))
        dealt = deal_hole_cards(self.current_hand_deck, self.state.players, self.state.dealer_seat)
        actor = first_preflop_actor(self.state, sb_seat, bb_seat)
        self.state = dc_replace(
            self.state,
            players=dealt,
            deck_remaining_count=self.current_hand_deck.remaining(),
            action_on_seat=actor,
        )
        self._players_acted = set()
        self._max_commitment = max(player.committed_this_street for player in self.state.players)
        self._stall_key = None
        self._stall_repeats = 0
        self.prev_equity = 0.5
        if self.tendencies is not None:
            self.tendencies.start_hand()
        self._run_until_decision()
        if self.state.action_on_seat is None:
            return -1, np.zeros(self._obs_dim, dtype=np.float32), np.zeros(7, dtype=np.int32)
        actor = self.state.action_on_seat
        return actor, self._observe(actor), self.get_action_mask()

    def _commit(self, seat: int, action: Action) -> None:
        """Apply one action with the engine rules and the half-raise bookkeeping."""
        assert self.state is not None
        prior_max = self._max_commitment
        prior_min = max(self.state.config.big_blind, self.state.last_raise_size)
        self._players_acted.add(seat)
        if self.tendencies is not None:
            # Feed the rolling tracker before the state changes, so "facing a
            # bet" reflects what the actor actually saw.
            self.tendencies.record_action(
                seat,
                action.type,
                street_is_preflop=self.state.street == Street.PREFLOP,
                facing_bet=self.state.current_bet_to_call
                > self.state.players[seat].committed_this_street,
            )
        state = apply_action(self.state, seat, action)
        state = rebuild_pots(state)
        self._max_commitment = max(
            self._max_commitment,
            max((player.committed_this_street for player in state.players), default=0),
        )
        if reopens_betting(action, prior_max, prior_min):
            self._players_acted = {seat}
        self.state = self._advance_action(state)

    def _advance_action(self, state: GameState) -> GameState:
        """Move action off the player who just acted.

        Kept as a one-argument method so tests can replace a stuck opponent.
        """
        if state.action_on_seat is None:
            return state
        return advance_action(state, state.action_on_seat)

    def _street_is_finished(self) -> bool:
        assert self.state is not None
        if count_active_players(self.state) <= 1 or self.state.action_on_seat is None:
            return True
        return round_is_closed(self.state, self._players_acted, self._max_commitment)

    def _advance_street(self) -> bool:
        """Close the current street. Return True when the hand is over."""
        assert self.state is not None and self.current_hand_deck is not None
        state = return_uncalled_bet(self.state)
        state = rebuild_pots(state)
        self.state = state
        if count_active_players(state) <= 1 or state.street == Street.RIVER:
            return True
        nxt = _NEXT_STREET[state.street]
        if nxt == Street.FLOP:
            board = tuple(deal_flop(self.current_hand_deck))
        elif nxt == Street.TURN:
            board = state.community_cards + (deal_turn(self.current_hand_deck),)
        else:
            board = state.community_cards + (deal_river(self.current_hand_deck),)
        state = dc_replace(
            state,
            street=nxt,
            community_cards=board,
            deck_remaining_count=self.current_hand_deck.remaining(),
        )
        self.state = reset_street_action(state)
        self._players_acted = set()
        self._max_commitment = max(
            (player.committed_this_street for player in self.state.players), default=0
        )
        return False

    def _run_until_decision(self) -> bool:
        """Auto-play opponents and finished streets. Return True if the hand ended."""
        steps = 0
        while True:
            steps += 1
            if steps > 500:
                self._escape_stuck_round()
                steps = 0
            if self._street_is_finished():
                if self._advance_street():
                    self.state = self._resolve_hand()
                    return True
                continue
            assert self.state is not None and self.state.action_on_seat is not None
            seat = self.state.action_on_seat
            if seat in self.learning_seats and self._seat_can_decide(seat):
                return False
            if seat in self.learning_seats:
                # A busted or all-in learner is not a decision. Skipping them
                # here used to freeze the hand and censor the sitting.
                self.state = advance_action(self.state, seat)
                continue
            self._play_opponent_turn()

    def _seat_can_decide(self, seat: int) -> bool:
        """True when ``seat`` can still put chips in voluntarily."""
        assert self.state is not None
        player = self.state.players[seat]
        return (
            not player.has_folded
            and not player.is_eliminated
            and not player.is_all_in
            and player.stack > 0
        )

    def _stall_signature(self, seat: int) -> tuple:
        assert self.state is not None
        return (
            seat,
            self.state.street,
            self.state.current_bet_to_call,
            tuple(player.committed_this_street for player in self.state.players),
            tuple(player.has_folded for player in self.state.players),
            tuple(player.stack for player in self.state.players),
        )

    def _note_stall(self, bot: Bot) -> None:
        name = getattr(bot, "name", None) or type(bot).__name__
        self.stall_counts[str(name)] = self.stall_counts.get(str(name), 0) + 1

    def _forced_progress(self, seat: int) -> Action:
        assert self.state is not None
        if chips_still_owed(self.state):
            player = self.state.players[seat]
            call_to = min(
                max(
                    self.state.current_bet_to_call,
                    max(p.committed_this_street for p in self.state.players),
                ),
                player.stack + player.committed_this_street,
            )
            if call_to > player.committed_this_street and player.stack > 0:
                return Action.call(call_to)
            return Action.fold()
        return Action.check()

    def _play_opponent_turn(self) -> None:
        if self.state is None or self.state.action_on_seat is None:
            return
        seat = self.state.action_on_seat
        bot = self._get_bot_for_seat(seat)
        legal = legal_actions(self.state, seat)
        stall_key = self._stall_signature(seat)
        if stall_key == self._stall_key:
            self._stall_repeats += 1
        else:
            self._stall_key = stall_key
            self._stall_repeats = 0
        if self.stall_guard and self._stall_repeats >= 2:
            self._note_stall(bot)
            action = self._forced_progress(seat)
        else:
            action = ensure_progressing_action(
                self.state,
                bot.act(self.state.view_for(seat), legal),
                legal,
            )
        action = cap_action(self.state, seat, action, self.max_bet)
        try:
            validate(self.state, seat, action)
        except IllegalActionError:
            action = Action.fold()
        self._commit(seat, action)

    def _escape_stuck_round(self) -> None:
        if self.state is None or self.state.action_on_seat is None:
            return
        seat = self.state.action_on_seat
        if seat not in self.learning_seats:
            self._note_stall(self._get_bot_for_seat(seat))
        if not self.stall_guard:
            raise RuntimeError(
                "Betting round did not close within 500 actions "
                f"(street={self.state.street}, bet={self.state.current_bet_to_call})"
            )
        self._commit(seat, Action.fold())

    def _get_bot_for_seat(self, seat: int) -> Bot:
        if seat in self.learning_seats:
            raise RuntimeError(f"Seat {seat} is a learning seat, not an opponent")
        return self._bot_for_seat[seat]

    def _resolve_hand(self) -> GameState:
        """Award pots with the engine showdown and add the chips to each stack."""
        if self.state is None:
            raise RuntimeError("State not initialized")
        state = self.state
        if self.tendencies is not None:
            # Hand is over: fold per-hand flags into the rolling counters. Done
            # before the awards so a feature never depends on the outcome.
            self.tendencies.end_hand()
        if len(state.community_cards) > 5:
            non_folded = [i for i, player in enumerate(state.players) if not player.has_folded]
            total_pot = sum(player.committed_this_hand for player in state.players)
            if len(non_folded) == 1:
                awards = {i: (total_pot if i == non_folded[0] else 0) for i in range(len(state.players))}
            else:
                split_amount = total_pot // max(len(non_folded), 1)
                awards = {seat: split_amount for seat in non_folded}
                for i in range(len(state.players)):
                    awards.setdefault(i, 0)
        else:
            deck_rng = RNG(seed=42)
            minimal_deck = Deck(rng=deck_rng)
            try:
                awards, _rake = resolve(state, minimal_deck)
            except Exception:
                non_folded = [i for i, player in enumerate(state.players) if not player.has_folded]
                total_pot = sum(player.committed_this_hand for player in state.players)
                if len(non_folded) == 1:
                    awards = {
                        i: (total_pot if i == non_folded[0] else 0) for i in range(len(state.players))
                    }
                else:
                    split_amount = total_pot // max(len(non_folded), 1)
                    awards = {seat: split_amount for seat in non_folded}
                    for i in range(len(state.players)):
                        awards.setdefault(i, 0)

        players = []
        for seat, player in enumerate(state.players):
            award = awards.get(seat, 0)
            new_stack = player.stack + award
            players.append(
                player.with_stack(max(0, new_stack))
                .with_committed_this_street(0)
                .with_committed_this_hand(0)
                .with_all_in(False)
                .with_eliminated(new_stack <= 0)
            )
        return dc_replace(state, players=tuple(players), pots=[], street=Street.SHOWDOWN, action_on_seat=None)

    def _table_total(self) -> int:
        """Chips that started the hand. Winner-take-all share uses this pool."""
        if self.player_stacks_at_hand_start:
            return sum(self.player_stacks_at_hand_start.values())
        if self.state is None:
            return 0
        return sum(player.stack + player.committed_this_hand for player in self.state.players)

    def _prize_share(self, stack: int, total: int) -> float:
        """Stack's fraction of the chips in play. Equals P(win) under winner-take-all."""
        if total <= 0:
            return 0.0
        share = max(stack, 0) / total
        if self.reward_exponent != 1.0:
            share = share ** self.reward_exponent
        return share

    def _prize_delta(self, seat: int) -> float:
        """Change in winner-take-all equity across the hand, scaled for training."""
        if self.state is None:
            return 0.0
        total = max(self._table_total(), 1)
        before = self.player_stacks_at_hand_start.get(seat, 0)
        player = self.state.players[seat]
        after = player.stack + player.committed_this_hand
        return REWARD_SCALE * (self._prize_share(after, total) - self._prize_share(before, total))

    def _terminal_rewards(self) -> dict[int, float]:
        rewards: dict[int, float] = {}
        if self.state is None:
            return rewards
        for player in self.state.players:
            rewards[player.seat] = self._prize_delta(player.seat)
        return rewards

    def _zero_rewards(self) -> dict[int, float]:
        return {seat: 0.0 for seat in range(self.num_players)}

    def _observe(self, seat: int) -> npt.NDArray[np.float32]:
        assert self.state is not None
        if self.obs_version >= 5:
            return build_observation_v5(
                self.state, seat, self.tendencies, hand_cap=self.hand_cap
            )
        if self.obs_version >= 4:
            return build_observation_v4(self.state, seat, self.tendencies)
        if self.obs_version == 3:
            return build_observation_v3(self.state, seat, self.tendencies)
        if self.obs_version == 2:
            return build_observation_v2(self.state, seat)
        return build_observation(self.state, seat)

    def _current_obs(self) -> npt.NDArray[np.float32]:
        if (
            self.state is None
            or self.state.action_on_seat is None
            or self.state.street == Street.SHOWDOWN
        ):
            return np.zeros(self._obs_dim, dtype=np.float32)
        return self._observe(self.state.action_on_seat)

    def _estimate_hand_equity(self, obs: npt.NDArray[np.float32]) -> float:
        try:
            if self.state is None or not self.state.players:
                return 0.5
            player = self.state.players[self.learning_seat]
            if not player.hole_cards or len(player.hole_cards) != 2:
                return 0.5
            all_cards = list(player.hole_cards) + list(self.state.community_cards)
            if self.state.street == Street.PREFLOP:
                card_strength = (float(obs[0]) + float(obs[1])) / 2.0
                return float(np.clip(0.15 + 0.7 * card_strength, 0.0, 1.0))
            if len(all_cards) < 5:
                card_strength = (float(obs[0]) + float(obs[1])) / 2.0
                community_strength = float(np.mean(obs[2:7])) if len(obs) > 6 else 0.5
                return float(np.clip(0.6 * card_strength + 0.4 * community_strength, 0.0, 1.0))
            current_hand_rank = evaluate(all_cards)
            hand_value = float(current_hand_rank.value)
            hand_strength_map = {
                8: 1.0,
                7: 0.95,
                6: 0.85,
                5: 0.75,
                4: 0.6,
                3: 0.45,
                2: 0.3,
                1: 0.15,
                0: 0.05,
            }
            hand_type = (int(hand_value) >> 20) & 0xF
            base_equity = hand_strength_map.get(hand_type, 0.5)
            community_strength = float(np.mean(obs[2:7])) if len(obs) > 6 else 0.5
            return float(np.clip(base_equity + 0.1 * (community_strength - 0.5), 0.0, 1.0))
        except (AttributeError, IndexError, ValueError, TypeError):
            card_strength = (float(obs[0]) + float(obs[1])) / 2.0
            community_strength = float(np.mean(obs[2:7])) if len(obs) > 6 else 0.5
            return float(np.clip(0.6 * card_strength + 0.4 * community_strength, 0.0, 1.0))

    def _compute_reward(self, seat: int) -> float:
        """Winner-take-all equity change, paid once when the hand ends.

        Mid-hand the return is zero (obs v1 still adds a small equity term).
        A chip won in a short-stack profile is a larger share of the pool than
        the same chip won when the big blind is small, because the pot itself
        is sized in blinds.
        """
        if self.state is None:
            return 0.0
        player = self.state.players[seat]
        active_players = count_active_players(self.state)
        is_hand_ended = active_players <= 1 or self.state.street == Street.SHOWDOWN
        if not is_hand_ended:
            if self.obs_version == 1:
                try:
                    obs = build_observation(self.state, seat)
                    current_equity = self._estimate_hand_equity(obs)
                    delta = current_equity - self.prev_equity
                    self.prev_equity = current_equity
                    return 0.05 * delta
                except Exception:
                    return 0.0
            return 0.0
        self.prev_equity = 0.5
        return self._prize_delta(seat)

    def _get_info(self) -> dict[str, Any]:
        if self.state is None:
            return {}
        return {
            "hand_number": self.state.hand_number,
            "street": self.state.street.value,
            "action_on_seat": self.state.action_on_seat,
            "stacks": {player.seat: player.stack for player in self.state.players},
            "active_players": sum(1 for player in self.state.players if not player.is_eliminated),
        }

    def render(self) -> None:
        if self.render_mode is None:
            return
        if self.render_mode == "text" and self.state is not None:
            print(f"Hand {self.state.hand_number} - Street: {self.state.street.value}")
            print(f"Stacks: {[player.stack for player in self.state.players]}")
            print(f"Action on: {self.state.action_on_seat}")
        else:
            raise ValueError(f"Unknown render_mode: {self.render_mode}")

    def close(self) -> None:
        return None

    def get_action_mask(self) -> npt.NDArray[np.int32]:
        """Legal action mask for the seat that is about to act."""
        if self.state is None or self.state.action_on_seat is None:
            raise RuntimeError("No player to act")
        return build_action_mask(
            self.state,
            self.state.action_on_seat,
            max_bet=self.max_bet,
            scheme=self.bet_scheme,
            dedup=self.dedup_mask,
        )


class MultiSeatPokerEnv(PokerEnv):
    """Decision API for several learning seats sharing one hand.

    ``reset`` returns ``(seat, obs, mask)`` and ``step`` returns
    ``(seat, obs, mask, rewards_by_seat, done)``.
    """

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[int, npt.NDArray[np.float32], npt.NDArray[np.int32]]:
        super(PokerEnv, self).reset(seed=seed)
        self._prepare_rng(seed)
        return self.reset_multi(options)

    def step(
        self, action: int
    ) -> tuple[int, npt.NDArray[np.float32], npt.NDArray[np.int32], dict[int, float], bool]:
        return self.step_multi(action)
