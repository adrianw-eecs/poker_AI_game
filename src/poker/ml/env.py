"""Gymnasium-compatible poker environment for RL training."""

from dataclasses import replace as dc_replace
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from poker.bots.base import Bot
from poker.bots.random_bot import RandomBot
from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.domain.deck import Deck
from poker.engine.action_validator import legal_actions, validate
from poker.engine.dealer import deal_flop, deal_hole_cards, deal_river, deal_turn
from poker.engine.showdown import resolve
from poker.exceptions import IllegalActionError
from poker.logging.logger import NullLogger
from poker.ml.action_space import action_index_to_action, build_action_mask
from poker.ml.observation import build_observation
from poker.rng import RNG
from poker.state.game_state import GameState, Street
from poker.state.player_state import PlayerState
from poker.state.pot import Pot


class PokerEnv(gym.Env):
    """Gymnasium-compatible Texas Hold'em poker environment.

    This environment provides:
    - Fixed-length observation vectors (155 features) for each player
      * Original 142 features (cards, stacks, positions, action history, etc.)
      * Enhanced with: hand strength bucket (8), SPR (1), aggression metrics (4)
    - Discrete action space (7 actions: fold, check/call, 5 raise buckets)
    - Legal action masking to prevent illegal moves
    - Dense reward shaping (equity bonuses during hand + stack delta at end)
    - Support for multi-player, variable player count games

    Each step represents one player action (not a full hand).
    Terminal state is when all but one player are eliminated.
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
        opponent_bots: list[Bot] | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the poker environment.

        Args:
            num_players: Number of players in the game.
            starting_stack: Starting stack for each player.
            small_blind: Small blind amount.
            big_blind: Big blind amount.
            ante: Ante per player (0 if no antes).
            learning_seat: Seat of the learning agent (default 0).
            opponent_bots: List of Bot instances for other seats (default: all RandomBot).
            render_mode: Rendering mode ("text" or None for no rendering).
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If parameters are invalid.
        """
        if num_players < 2 or num_players > 10:
            raise ValueError(f"num_players must be in [2, 10], got {num_players}")
        if starting_stack <= 0:
            raise ValueError(f"starting_stack must be positive, got {starting_stack}")
        if small_blind <= 0 or big_blind <= small_blind:
            raise ValueError("Blind amounts must be positive and ordered")
        if learning_seat < 0 or learning_seat >= num_players:
            raise ValueError(f"learning_seat must be in [0, {num_players - 1}], got {learning_seat}")

        self.num_players = num_players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.ante = ante
        self.learning_seat = learning_seat

        # Game configuration
        self.config = GameConfig(
            num_players=num_players,
            starting_stack=starting_stack,
            small_blind=small_blind,
            big_blind=big_blind,
            ante=ante,
            rake_percent=0,
            rake_cap=None,
            blind_schedule=BlindSchedule(
                levels=[BlindLevel(small_blind, big_blind, ante)],
                hands_per_level=1000,
                fixed=True,
            ),
            run_it_twice=False,
        )
        self.blind_level = BlindLevel(small_blind, big_blind, ante)

        # Set up opponent bots (default: RandomBot for all non-learning seats)
        if opponent_bots is None:
            opponent_bots = [RandomBot(seed=seed) for _ in range(num_players - 1)]
        if len(opponent_bots) != num_players - 1:
            raise ValueError(
                f"opponent_bots must have {num_players - 1} bots, got {len(opponent_bots)}"
            )
        self.opponent_bots = opponent_bots

        # Action and observation spaces
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(155,), dtype=np.float32
        )

        # Environment state
        self.state: GameState | None = None
        self.hand_number = 0
        self.player_stacks_at_hand_start: dict[int, int] = {}
        self.prev_equity: float = 0.5  # For equity delta shaping
        self.render_mode = render_mode
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.logger = NullLogger()  # Silent logging

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        """Reset the environment to start a new hand.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options (unused).

        Returns:
            Tuple of (initial_observation_for_learning_agent, info_dict).
        """
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.Generator(np.random.PCG64(seed))

        # Initialize game state with all players at starting stack
        dealer_seat = self.hand_number % self.num_players
        players = tuple(
            PlayerState(
                seat=i,
                name=f"Player{i}",
                stack=self.starting_stack,
                hole_cards=(),
                committed_this_street=0,
                committed_this_hand=0,
                has_folded=False,
                is_all_in=False,
                is_eliminated=False,
            )
            for i in range(self.num_players)
        )

        self.state = GameState(
            hand_number=self.hand_number,
            street=Street.PREFLOP,
            dealer_seat=dealer_seat,
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

        # Post antes and blinds
        self.state = self._post_antes_and_blinds()

        # Deal hole cards
        deck_rng = RNG(seed=int(self.rng.integers(0, 2**31 - 1)))
        deck = Deck(rng=deck_rng)
        players_after_deal = deal_hole_cards(deck, self.state.players, dealer_seat)
        self.state = dc_replace(self.state, players=players_after_deal, deck_remaining_count=deck.remaining())

        # Set action to first player (UTG in multi-way, button in heads-up)
        num_players = len(self.state.players)
        if num_players == 2:
            action_on_seat = dealer_seat
        else:
            action_on_seat = (dealer_seat + 3) % num_players

        self.state = dc_replace(self.state,action_on_seat=action_on_seat)

        # Store stacks at the start of the hand (for reward calculation)
        self.player_stacks_at_hand_start = {p.seat: p.stack for p in self.state.players}

        # Initialize equity estimate for reward shaping
        self.prev_equity = 0.5  # Neutral starting point

        # Get the first observation (for the learning agent)
        obs = build_observation(self.state, self.learning_seat)
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Execute one player action and auto-play until next learning agent turn or hand end.

        Args:
            action: Discrete action index 0-6.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
            - observation: Next player's observation (zeros if hand ended)
            - reward: Stack change normalized by starting_stack (0 mid-hand, end-of-hand delta at terminal)
            - terminated: True if hand is over (all but one player eliminated or hand showdown)
            - truncated: False (not used in poker)
            - info: Additional information dict

        Raises:
            RuntimeError: If environment not initialized.
            ValueError: If action is invalid.
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Validate that it's the learning agent's turn
        if self.state.action_on_seat != self.learning_seat:
            raise RuntimeError(
                f"Not learning agent's turn. Expected seat {self.learning_seat}, "
                f"action on seat {self.state.action_on_seat}"
            )

        # Convert discrete action to concrete Action
        try:
            concrete_action = action_index_to_action(action, self.state, self.learning_seat)
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid action {action}: {e}") from e

        # Validate the action is legal
        try:
            validate(self.state, self.learning_seat, concrete_action)
        except IllegalActionError as e:
            raise ValueError(f"Action {action} is illegal: {e}") from e

        # Apply the learning agent's action
        self.state = self._apply_action(self.state, self.learning_seat, concrete_action)

        # Auto-play other bots until it's the learning agent's turn again or hand ends
        hand_ended, learning_agent_turn = False, False
        while not hand_ended and not learning_agent_turn:
            # Check if hand is over
            active_players = sum(1 for p in self.state.players if not p.has_folded and not p.is_eliminated)
            if active_players <= 1:
                # Hand is over (everyone else folded or eliminated)
                hand_ended = True
                break

            # Check if we're back on a new street and need to deal cards
            if self.state.street == Street.FLOP and len(self.state.community_cards) == 0:
                deck = Deck(rng=RNG(seed=int(self.rng.integers(0, 2**31 - 1))))
                self.state = dc_replace(self.state,
                    community_cards=deal_flop(deck),
                    action_on_seat=None,
                    action_history_this_street=[],
                )
                # Reset action to first active player
                self.state = self._set_action_to_first_player()

            elif self.state.street == Street.TURN and len(self.state.community_cards) == 4:
                deck = Deck(rng=RNG(seed=int(self.rng.integers(0, 2**31 - 1))))
                self.state = dc_replace(self.state,
                    community_cards=self.state.community_cards + (deal_turn(deck),),
                    action_on_seat=None,
                    action_history_this_street=[],
                )
                self.state = self._set_action_to_first_player()

            elif self.state.street == Street.RIVER and len(self.state.community_cards) == 5:
                # Showdown
                hand_ended = True
                break

            # If no one is set to act, try to advance to next player
            if self.state.action_on_seat is None:
                # All remaining players either folded or all-in; go to showdown or next street
                if active_players <= 1:
                    hand_ended = True
                    break

                # Move to next street if not at river
                if self.state.street != Street.RIVER:
                    next_street_map = {
                        Street.PREFLOP: Street.FLOP,
                        Street.FLOP: Street.TURN,
                        Street.TURN: Street.RIVER,
                    }
                    self.state = dc_replace(self.state, street=next_street_map[self.state.street])
                    if self.state.street == Street.FLOP:
                        deck = Deck(rng=RNG(seed=int(self.rng.integers(0, 2**31 - 1))))
                        self.state = dc_replace(self.state, community_cards=deal_flop(deck))
                else:
                    hand_ended = True
                    break
            else:
                # Someone needs to act
                seat = self.state.action_on_seat
                if seat == self.learning_seat:
                    # Back to learning agent
                    learning_agent_turn = True
                    break
                else:
                    # Get action from opponent bot
                    bot = self._get_bot_for_seat(seat)
                    legal = legal_actions(self.state, seat)
                    opponent_action = bot.act(self.state.view_for(seat), legal)

                    # Apply opponent's action
                    self.state = self._apply_action(self.state, seat, opponent_action)

        # Resolve hand if ended
        if hand_ended:
            self.state = self._resolve_hand()

        # Compute reward and next observation
        reward = self._compute_reward(self.learning_seat)
        obs = np.zeros(155, dtype=np.float32)
        if not hand_ended and self.state.action_on_seat == self.learning_seat:
            obs = build_observation(self.state, self.learning_seat)

        info = self._get_info()
        return obs, reward, hand_ended, False, info

    def _post_antes_and_blinds(self) -> GameState:
        """Post antes and blinds, return updated state."""
        if self.state is None:
            raise RuntimeError("State not initialized")

        state = self.state
        players = list(state.players)

        # Post antes
        if state.config.ante > 0:
            for i, p in enumerate(players):
                players[i] = dc_replace(
                    p,
                    stack=max(0, p.stack - state.config.ante),
                    committed_this_hand=p.committed_this_hand + state.config.ante,
                )

        # Post blinds
        sb_seat = (state.dealer_seat + 1) % len(players)
        bb_seat = (state.dealer_seat + 2) % len(players)

        if len(players) == 2:
            # Heads-up: dealer is small blind
            sb_seat = state.dealer_seat
            bb_seat = (state.dealer_seat + 1) % len(players)

        sb_amount = min(state.config.small_blind, players[sb_seat].stack)
        bb_amount = min(state.config.big_blind, players[bb_seat].stack)

        players[sb_seat] = dc_replace(
            players[sb_seat],
            stack=players[sb_seat].stack - sb_amount,
            committed_this_street=sb_amount,
            committed_this_hand=players[sb_seat].committed_this_hand + sb_amount,
        )
        players[bb_seat] = dc_replace(
            players[bb_seat],
            stack=players[bb_seat].stack - bb_amount,
            committed_this_street=bb_amount,
            committed_this_hand=players[bb_seat].committed_this_hand + bb_amount,
        )

        # Create pot
        total_pot = sum(p.committed_this_street for p in players)
        pot = Pot(total_pot, frozenset(i for i in range(len(players))))

        return dc_replace(state,
            players=tuple(players),
            pots=[pot],
            current_bet_to_call=bb_amount,
            last_raise_size=bb_amount,
        )

    def _apply_action(self, state: GameState, seat: int, action) -> GameState:  # type: ignore[no-untyped-def]
        """Apply an action to the state, return updated state."""
        players = list(state.players)
        player = players[seat]

        if action.type.value == "fold":
            players[seat] = dc_replace(player, has_folded=True)
        elif action.type.value == "check":
            pass  # No stack change
        elif action.type.value == "call":
            amount_to_call = state.current_bet_to_call - player.committed_this_street
            actual_amount = min(amount_to_call, player.stack)
            players[seat] = dc_replace(
                player,
                stack=player.stack - actual_amount,
                committed_this_street=player.committed_this_street + actual_amount,
                committed_this_hand=player.committed_this_hand + actual_amount,
                is_all_in=(player.stack - actual_amount == 0),
            )
            # Update current_bet_to_call
            state = dc_replace(state,current_bet_to_call=player.committed_this_street + actual_amount)
        elif action.type.value == "raise":
            amount_to_call = state.current_bet_to_call - player.committed_this_street
            actual_amount = min(action.amount - player.committed_this_street, player.stack)
            players[seat] = dc_replace(
                player,
                stack=player.stack - actual_amount,
                committed_this_street=player.committed_this_street + actual_amount,
                committed_this_hand=player.committed_this_hand + actual_amount,
                is_all_in=(player.stack - actual_amount == 0),
            )
            state = dc_replace(state,
                current_bet_to_call=action.amount,
                last_raise_size=action.amount - amount_to_call - player.committed_this_street,
            )
        elif action.type.value == "all_in":
            actual_amount = player.stack
            players[seat] = dc_replace(
                player,
                stack=0,
                committed_this_street=player.committed_this_street + actual_amount,
                committed_this_hand=player.committed_this_hand + actual_amount,
                is_all_in=True,
            )
            state = dc_replace(state,current_bet_to_call=max(state.current_bet_to_call, players[seat].committed_this_street))

        # Update pots (simplified: just main pot for now)
        total_committed = sum(p.committed_this_street for p in players)
        if state.pots:
            state = dc_replace(state,pots=[Pot(total_committed, frozenset(range(len(players))))])

        # Move to next player
        state = dc_replace(state,players=tuple(players))
        state = self._advance_action(state)
        return state

    def _advance_action(self, state: GameState) -> GameState:
        """Advance action_on_seat to the next player who can act."""
        if state.action_on_seat is None:
            return state

        num_players = len(state.players)
        current_seat = state.action_on_seat
        max_commitment = max((p.committed_this_street for p in state.players), default=0)

        # Find the next player who can act
        for _ in range(num_players):
            next_seat = (current_seat + 1) % num_players
            next_player = state.players[next_seat]

            # Can act if: not folded, not all-in, has stack left, and hasn't matched the bet
            if (not next_player.has_folded and
                not next_player.is_all_in and
                next_player.stack > 0 and
                next_player.committed_this_street < max_commitment):
                return dc_replace(state,action_on_seat=next_seat)

            current_seat = next_seat

        # No one can act (all folded, all-in, or matched)
        return dc_replace(state,action_on_seat=None)

    def _set_action_to_first_player(self) -> GameState:
        """Set action_on_seat to the first player who can act post-flop."""
        if self.state is None:
            raise RuntimeError("State not initialized")

        state = self.state
        num_players = len(state.players)
        first_active = (state.dealer_seat + 1) % num_players

        # Find first player who can act
        for i in range(num_players):
            seat = (first_active + i) % num_players
            if not state.players[seat].has_folded and not state.players[seat].is_all_in:
                return dc_replace(state,action_on_seat=seat, action_history_this_street=[])

        return dc_replace(state,action_on_seat=None, action_history_this_street=[])

    def _get_bot_for_seat(self, seat: int) -> Bot:
        """Get the bot for a given seat (accounting for learning_seat)."""
        if seat == self.learning_seat:
            raise RuntimeError(f"Seat {seat} is the learning agent, not an opponent")

        # Map seat to opponent_bots index (skip learning_seat)
        opponent_index = seat if seat < self.learning_seat else seat - 1
        return self.opponent_bots[opponent_index]

    def _resolve_hand(self) -> GameState:
        """Resolve the hand via showdown, return final state with stacks updated."""
        if self.state is None:
            raise RuntimeError("State not initialized")

        # For now, simple resolution: just run showdown
        # TODO: Implement full showdown logic
        state = dc_replace(self.state, street=Street.SHOWDOWN)

        # Mark all non-folded players as having shown (simplified)
        players = list(state.players)
        for i, p in enumerate(players):
            if not p.has_folded:
                # Winners (simplified: just split among non-folded)
                pass

        # TODO: Implement actual winner determination and stack updates
        # For now, just return state as-is; actual reward will reflect stack changes
        return state

    def _estimate_hand_equity(self, obs: npt.NDArray[np.float32]) -> float:
        """Estimate hand equity from observation features.

        Uses a simple heuristic: hole card strength + community card strength.
        Returns value in [0, 1] where 0.5 = break-even vs random.

        Args:
            obs: Observation vector (142 dims)

        Returns:
            Estimated equity in [0, 1]
        """
        # Extract hole card indices (first 2 elements, normalized to [0, 1])
        card1_val = float(obs[0])
        card2_val = float(obs[1])
        card_strength = (card1_val + card2_val) / 2.0

        # Community cards (elements 2-7, 5 cards normalized to [0, 1])
        community_vals = obs[2:7]
        community_strength = float(np.mean(community_vals)) if len(community_vals) > 0 else 0.5

        # Blended estimate: more weight to hole cards early, more to community later
        hand_equity = 0.6 * card_strength + 0.4 * community_strength

        # Clamp to [0, 1]
        return float(np.clip(hand_equity, 0.0, 1.0))

    def _compute_reward(self, seat: int) -> float:
        """Compute reward for a player with dense reward shaping.

        During hand: intrinsic reward from equity improvement + stack penalties
        At hand end: extrinsic reward from stack change

        Args:
            seat: The seat of the player.

        Returns:
            Total reward combining intrinsic and extrinsic signals.
        """
        if self.state is None:
            return 0.0

        player = self.state.players[seat]
        stack_before = self.player_stacks_at_hand_start.get(seat, self.starting_stack)
        stack_change = player.stack - stack_before

        # Check if hand is still ongoing (not all opponents folded/eliminated)
        active_players = sum(1 for p in self.state.players if not p.has_folded and not p.is_eliminated)
        is_hand_ended = active_players <= 1

        reward = 0.0

        # Intrinsic reward during hand: equity bonus shaping
        if not is_hand_ended and self.state.street != Street.SHOWDOWN:
            # Estimate current hand strength
            try:
                obs = build_observation(self.state, seat)
                current_equity = self._estimate_hand_equity(obs)
                equity_delta = current_equity - self.prev_equity

                # FIX 3: Boost intrinsic reward 5x (0.01 → 0.05)
                # This makes during-hand progress signals matter more
                intrinsic_reward = 0.05 * equity_delta
                reward += intrinsic_reward

                # Update previous equity for next step
                self.prev_equity = current_equity
            except Exception:
                # Fallback if observation building fails
                pass

            # Short-stack penalty (encourage stack preservation)
            big_blind = self.state.config.big_blind
            if player.stack < 3 * big_blind:
                reward -= 0.02  # Small penalty for being short-stacked

        else:
            # Extrinsic reward at hand end
            # Reset equity for next hand
            self.prev_equity = 0.5

            # Main reward: stack change
            # FIX 1: Scale reward 10x (optimal signal-to-noise ratio)
            # This makes win/loss differences visible to the network
            normalized_change = float(stack_change) / self.starting_stack
            reward = 10.0 * normalized_change  # 10x amplification

            # Bonus for winning pot (positive result)
            if stack_change > 0:
                # Small additional bonus (also scaled)
                reward += 0.2 * min(normalized_change, 0.1)

        return reward

    def _get_info(self) -> dict[str, Any]:
        """Get info dictionary for this step.

        Returns:
            A dict with game state information.
        """
        if self.state is None:
            return {}

        return {
            "hand_number": self.state.hand_number,
            "street": self.state.street.value,
            "action_on_seat": self.state.action_on_seat,
            "stacks": {p.seat: p.stack for p in self.state.players},
            "active_players": sum(1 for p in self.state.players if not p.is_eliminated),
        }

    def render(self) -> None:
        """Render the current game state.

        Currently supports "text" mode with basic terminal output.
        """
        if self.render_mode is None:
            return

        if self.render_mode == "text" and self.state is not None:
            print(f"Hand {self.state.hand_number} - Street: {self.state.street.value}")
            print(f"Stacks: {[p.stack for p in self.state.players]}")
            print(f"Action on: {self.state.action_on_seat}")
        else:
            raise ValueError(f"Unknown render_mode: {self.render_mode}")

    def close(self) -> None:
        """Close the environment and clean up resources."""
        pass

    def get_action_mask(self) -> npt.NDArray[np.int32]:
        """Get the valid action mask for the current player.

        Returns:
            A binary array of shape (7,) with 1 for legal, 0 for illegal.

        Raises:
            RuntimeError: If no player has action or game is not initialized.
        """
        if self.state is None or self.state.action_on_seat is None:
            raise RuntimeError("No player to act")

        return build_action_mask(self.state, self.state.action_on_seat)
