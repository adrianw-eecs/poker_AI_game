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
from poker.evaluation.evaluator import evaluate
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
        rake_percent: float = 0.0,
        rake_cap: int | None = None,
        run_it_twice: bool = False,
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
            rake_percent: Rake percentage (0.0 to 5.0, default 0.0).
            rake_cap: Maximum rake per pot, or None for no cap.
            run_it_twice: Enable run-it-twice for all-in scenarios.

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
        if rake_percent < 0 or rake_percent > 10:
            raise ValueError(f"rake_percent must be in [0, 10], got {rake_percent}")

        self.num_players = num_players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.ante = ante
        self.learning_seat = learning_seat
        self.rake_percent = rake_percent
        self.rake_cap = rake_cap

        # Game configuration
        self.config = GameConfig(
            num_players=num_players,
            starting_stack=starting_stack,
            small_blind=small_blind,
            big_blind=big_blind,
            ante=ante,
            rake_percent=rake_percent,
            rake_cap=rake_cap,
            blind_schedule=BlindSchedule(
                levels=[BlindLevel(small_blind, big_blind, ante)],
                hands_per_level=1000,
                fixed=True,
            ),
            run_it_twice=run_it_twice,
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
        self.current_hand_deck: Deck | None = None  # Deck for current hand (shuffled once at start)

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

        # Create deck once for the entire hand (shuffle it once here)
        deck_rng = RNG(seed=int(self.rng.integers(0, 2**31 - 1)))
        self.current_hand_deck = Deck(rng=deck_rng)

        # Deal hole cards from the deck
        players_after_deal = deal_hole_cards(self.current_hand_deck, self.state.players, dealer_seat)
        self.state = dc_replace(self.state, players=players_after_deal, deck_remaining_count=self.current_hand_deck.remaining())

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

        # Auto-play bots until it's the learning agent's turn
        while self.state.action_on_seat != self.learning_seat and self.state.action_on_seat is not None:
            seat = self.state.action_on_seat
            bot = self._get_bot_for_seat(seat)
            legal = legal_actions(self.state, seat)
            opponent_action = bot.act(self.state.view_for(seat), legal)
            self.state = self._apply_action(self.state, seat, opponent_action)

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
                # Deal FLOP from the current hand's deck
                self.state = dc_replace(self.state,
                    community_cards=deal_flop(self.current_hand_deck),
                    action_on_seat=None,
                    action_history_this_street=[],
                )
                # Reset action to first active player
                self.state = self._set_action_to_first_player()

            elif self.state.street == Street.TURN and len(self.state.community_cards) == 3:
                # Deal TURN from the current hand's deck
                self.state = dc_replace(self.state,
                    community_cards=self.state.community_cards + (deal_turn(self.current_hand_deck),),
                    action_on_seat=None,
                    action_history_this_street=[],
                )
                self.state = self._set_action_to_first_player()

            elif self.state.street == Street.RIVER and len(self.state.community_cards) == 4:
                # Deal RIVER from the current hand's deck
                self.state = dc_replace(self.state,
                    community_cards=self.state.community_cards + (deal_river(self.current_hand_deck),),
                    action_on_seat=None,
                    action_history_this_street=[],
                )
                self.state = self._set_action_to_first_player()

            # If no one is set to act, try to advance to next player
            if self.state.action_on_seat is None:
                # All remaining players either folded or all-in; go to showdown or next street
                if active_players <= 1:
                    hand_ended = True
                    break

                # Check if all remaining players are all-in or folded
                # If so, auto-deal remaining streets until river
                non_folded_non_all_in = sum(
                    1 for p in self.state.players
                    if not p.has_folded and not p.is_eliminated and not p.is_all_in
                )

                if non_folded_non_all_in == 0 and active_players > 1:
                    # All remaining players are either all-in
                    # Auto-deal all remaining streets from the same deck
                    while self.state.street != Street.RIVER:
                        next_street_map = {
                            Street.PREFLOP: Street.FLOP,
                            Street.FLOP: Street.TURN,
                            Street.TURN: Street.RIVER,
                        }
                        self.state = dc_replace(self.state, street=next_street_map[self.state.street])

                        # Deal the card for this street from the current hand's deck
                        if self.state.street == Street.FLOP:
                            if len(self.state.community_cards) == 0:
                                self.state = dc_replace(self.state, community_cards=deal_flop(self.current_hand_deck), action_history_this_street=[])
                        elif self.state.street == Street.TURN:
                            if len(self.state.community_cards) == 3:
                                self.state = dc_replace(self.state, community_cards=self.state.community_cards + (deal_turn(self.current_hand_deck),), action_history_this_street=[])
                        elif self.state.street == Street.RIVER:
                            if len(self.state.community_cards) == 4:
                                self.state = dc_replace(self.state, community_cards=self.state.community_cards + (deal_river(self.current_hand_deck),), action_history_this_street=[])

                    hand_ended = True
                    break

                # Move to next street if not at river
                elif self.state.street != Street.RIVER:
                    next_street_map = {
                        Street.PREFLOP: Street.FLOP,
                        Street.FLOP: Street.TURN,
                        Street.TURN: Street.RIVER,
                    }
                    self.state = dc_replace(self.state, street=next_street_map[self.state.street])

                    # Deal cards for the new street from the current hand's deck
                    if self.state.street == Street.FLOP and len(self.state.community_cards) == 0:
                        self.state = dc_replace(self.state, community_cards=deal_flop(self.current_hand_deck), action_history_this_street=[])
                    elif self.state.street == Street.TURN and len(self.state.community_cards) == 3:
                        self.state = dc_replace(self.state, community_cards=self.state.community_cards + (deal_turn(self.current_hand_deck),), action_history_this_street=[])
                    elif self.state.street == Street.RIVER and len(self.state.community_cards) == 4:
                        self.state = dc_replace(self.state, community_cards=self.state.community_cards + (deal_river(self.current_hand_deck),), action_history_this_street=[])

                    # Always set action to first player after transitioning streets
                    self.state = self._set_action_to_first_player()
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
        """Advance action_on_seat to the next player who can act.

        Key logic:
        - If there are unacted players who haven't matched the current bet, they act next
        - If all active players have acted once and matched, move to next street (return None)
        """
        if state.action_on_seat is None:
            return state

        num_players = len(state.players)
        current_seat = state.action_on_seat
        max_commitment = max((p.committed_this_street for p in state.players), default=0)

        # Count active players and how many have acted
        active_players = [
            i for i in range(num_players)
            if not state.players[i].has_folded and not state.players[i].is_all_in
        ]

        # Find players who haven't matched the current bet
        need_to_act = [
            i for i in active_players
            if state.players[i].committed_this_street < max_commitment and state.players[i].stack > 0
        ]

        # If someone needs to act, find the next one in rotation
        if need_to_act:
            for _ in range(num_players):
                next_seat = (current_seat + 1) % num_players
                if next_seat in need_to_act:
                    return dc_replace(state, action_on_seat=next_seat)
                current_seat = next_seat

        # No one needs to act (everyone matched or folded/all-in)
        return dc_replace(state, action_on_seat=None)

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
        """Resolve the hand via showdown using proper hand evaluation.

        Uses showdown.resolve() to:
        - Evaluate actual hole cards vs community cards
        - Build main pots and side pots correctly
        - Apply rake according to game configuration
        - Distribute chips to winners based on hand ranks
        """
        if self.state is None:
            raise RuntimeError("State not initialized")

        state = self.state

        # Validate the state has reasonable community cards (0-5)
        # If more than 5, there's a corruption issue
        if len(state.community_cards) > 5:
            # This shouldn't happen in normal play
            # Fall back to simplified resolution
            non_folded = [i for i in range(len(state.players)) if not state.players[i].has_folded]

            if len(non_folded) == 1:
                # Everyone else folded, winner takes pot
                winner_seat = non_folded[0]
                total_pot = sum(p.committed_this_hand for p in state.players)
                awards = {i: (total_pot if i == winner_seat else 0) for i in range(len(state.players))}
            else:
                # Multiple players: determine winner by hand strength
                # Since we can't evaluate corrupted state, split equally
                total_pot = sum(p.committed_this_hand for p in state.players)
                split_amount = total_pot // len(non_folded)
                remainder = total_pot % len(non_folded)
                awards = {}
                for i, seat in enumerate(non_folded):
                    bonus = remainder if i == 0 else 0
                    awards[seat] = split_amount + bonus
                for i in range(len(state.players)):
                    if i not in awards:
                        awards[i] = 0
            rake_taken = 0
        else:
            # Normal case: use proper hand evaluation
            # Create a minimal deck for run-it-twice (if needed)
            deck_rng = RNG(seed=42)
            minimal_deck = Deck(rng=deck_rng)

            try:
                # Use the proper showdown resolution with hand evaluation
                awards, rake_taken = resolve(state, minimal_deck)
            except Exception as e:
                # If evaluation fails, fall back to simple winner determination
                print(f"[WARNING] Hand evaluation failed: {e}")
                non_folded = [i for i in range(len(state.players)) if not state.players[i].has_folded]

                if len(non_folded) == 1:
                    winner_seat = non_folded[0]
                    total_pot = sum(p.committed_this_hand for p in state.players)
                    awards = {i: (total_pot if i == winner_seat else 0) for i in range(len(state.players))}
                else:
                    total_pot = sum(p.committed_this_hand for p in state.players)
                    split_amount = total_pot // len(non_folded)
                    awards = {seat: split_amount for seat in non_folded}
                    for i in range(len(state.players)):
                        if i not in awards:
                            awards[i] = 0
                rake_taken = 0

        # Apply awards to player stacks
        players = list(state.players)
        for seat, award in awards.items():
            if award > 0:  # Player won chips
                players[seat] = players[seat].__class__(
                    seat=players[seat].seat,
                    name=players[seat].name,
                    stack=players[seat].stack + award,
                    hole_cards=players[seat].hole_cards,
                    committed_this_street=0,
                    committed_this_hand=players[seat].committed_this_hand,
                    has_folded=players[seat].has_folded,
                    is_all_in=players[seat].is_all_in,
                    is_eliminated=players[seat].is_eliminated,
                )

        # Return updated state with empty pots and showdown street
        state = dc_replace(state, players=tuple(players), pots=[], street=Street.SHOWDOWN)
        return state

    def _estimate_hand_equity(self, obs: npt.NDArray[np.float32]) -> float:
        """Estimate hand equity from observation features using hand evaluation.

        For preflop hands: Estimates equity based on hole card strength.
        For postflop hands: Evaluates current hand strength vs potential future cards.
        Returns value in [0, 1] where 0.5 = break-even vs random.

        This is a fast approximation for during-hand reward shaping. For exact equity,
        would require Monte Carlo simulation (expensive). This heuristic provides
        reasonable signal without computational overhead.

        Args:
            obs: Observation vector (155 dims)

        Returns:
            Estimated equity in [0, 1]
        """
        try:
            # Get actual hole cards and community from game state
            if self.state is None or not self.state.players:
                return 0.5  # Neutral if no state

            player = self.state.players[self.learning_seat]
            if not player.hole_cards or len(player.hole_cards) != 2:
                return 0.5  # No hole cards yet

            # Estimate equity based on street and available cards
            all_cards = list(player.hole_cards) + list(self.state.community_cards)

            if self.state.street == Street.PREFLOP:
                # Preflop: estimate based on hole card strength
                # Pocket aces = ~85% equity, worst hand = ~15%
                card_strength = (float(obs[0]) + float(obs[1])) / 2.0
                return float(np.clip(0.15 + 0.7 * card_strength, 0.0, 1.0))

            elif len(all_cards) < 5:
                # Incomplete board: use heuristic
                card_strength = (float(obs[0]) + float(obs[1])) / 2.0
                community_strength = float(np.mean(obs[2:7])) if len(obs) > 6 else 0.5
                hand_equity = 0.6 * card_strength + 0.4 * community_strength
                return float(np.clip(hand_equity, 0.0, 1.0))

            else:
                # Postflop with 5+ cards: evaluate actual hand
                current_hand_rank = evaluate(all_cards)
                hand_value = float(current_hand_rank.value)

                # Normalize to [0, 1] range
                # Flush = 1.0, Straight = 0.8, Three of a kind = 0.6, etc.
                hand_strength_map = {
                    8: 1.0,     # Straight Flush
                    7: 0.95,    # Four of a Kind
                    6: 0.85,    # Full House
                    5: 0.75,    # Flush
                    4: 0.6,     # Straight
                    3: 0.45,    # Three of a Kind
                    2: 0.3,     # Two Pair
                    1: 0.15,    # One Pair
                    0: 0.05,    # High Card
                }

                # Get hand type (first 3 bits of rank value)
                hand_type = (int(hand_value) >> 20) & 0xF
                base_equity = hand_strength_map.get(hand_type, 0.5)

                # Add small adjustment based on kicker strength (obs features)
                community_strength = float(np.mean(obs[2:7])) if len(obs) > 6 else 0.5
                kicker_adjustment = 0.1 * (community_strength - 0.5)

                return float(np.clip(base_equity + kicker_adjustment, 0.0, 1.0))

        except (AttributeError, IndexError, ValueError, TypeError):
            # Fallback to simple heuristic if anything goes wrong
            card_strength = (float(obs[0]) + float(obs[1])) / 2.0
            community_strength = float(np.mean(obs[2:7])) if len(obs) > 6 else 0.5
            hand_equity = 0.6 * card_strength + 0.4 * community_strength
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
