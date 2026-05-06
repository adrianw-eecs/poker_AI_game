"""Gymnasium-compatible poker environment for RL training."""

from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.ml.action_space import build_action_mask
from poker.ml.observation import build_observation
from poker.state.game_state import GameState, Street
from poker.state.player_state import PlayerState
from poker.state.pot import Pot


class PokerEnv(gym.Env):
    """Gymnasium-compatible Texas Hold'em poker environment.

    This environment provides:
    - Fixed-length observation vectors (142 features) for each player
    - Discrete action space (7 actions: fold, check/call, 5 raise buckets)
    - Legal action masking to prevent illegal moves
    - Rewards based on stack changes (normalized by starting stack)
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
        render_mode: str | None = None,
    ) -> None:
        """Initialize the poker environment.

        Args:
            num_players: Number of players in the game.
            starting_stack: Starting stack for each player.
            small_blind: Small blind amount.
            big_blind: Big blind amount.
            ante: Ante per player (0 if no antes).
            render_mode: Rendering mode ("text" or None for no rendering).

        Raises:
            ValueError: If parameters are invalid.
        """
        if num_players < 2 or num_players > 10:
            raise ValueError(f"num_players must be in [2, 10], got {num_players}")
        if starting_stack <= 0:
            raise ValueError(f"starting_stack must be positive, got {starting_stack}")
        if small_blind <= 0 or big_blind <= small_blind:
            raise ValueError("Blind amounts must be positive and ordered")

        self.num_players = num_players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.ante = ante

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

        # Action and observation spaces
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(142,), dtype=np.float32
        )

        # Environment state
        self.state: GameState | None = None
        self.hand_number = 0
        self.player_stacks_before_action: dict[int, int] = {}
        self.render_mode = render_mode

        # Tracking variables
        self._action_count = 0

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        """Reset the environment to the initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options (unused).

        Returns:
            Tuple of (initial_observation, info_dict).
        """
        super().reset(seed=seed)

        # Initialize game state with all players at starting stack
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
            dealer_seat=self.hand_number % self.num_players,
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
            action_on_seat=0,  # Will be set by play_hand or Session
        )

        self._action_count = 0
        self.player_stacks_before_action = {
            p.seat: p.stack for p in self.state.players
        }

        # Note: In a full implementation, we would call play_hand() here via a Session
        # to advance to the first player decision. For now, we return the initial state.
        # get_action_mask() will fail gracefully if called before step() is ready.

        # Get the first observation
        if self.state.action_on_seat is not None:
            acting_seat = self.state.action_on_seat
            obs = build_observation(self.state, acting_seat)
        else:
            obs = np.zeros(142, dtype=np.float32)

        info = self._get_info()
        return obs, info

    def step(self, action: int) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Execute one step (one player action) in the environment.

        Args:
            action: Discrete action index 0-6.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
            - observation: The next player's observation (or dummy if terminal)
            - reward: Stack change for the *previous* acting player
            - terminated: True if game is over (one player left)
            - truncated: False (not used in poker env)
            - info: Additional information dict

        Raises:
            NotImplementedError: Requires Session integration (Wave 5 Worker A, T22).
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        raise NotImplementedError(
            "step() requires Session integration (T22 - Wave 5 Worker A). "
            "PokerEnv.reset() works to initialize the environment. "
            "Full step() will be available after Session is implemented."
        )

    def _compute_reward(self, seat: int) -> float:
        """Compute reward for a player.

        Reward is the stack change since before the action,
        normalized by the starting stack.

        Args:
            seat: The seat of the player.

        Returns:
            Normalized stack change (-1.0 to 1.0 typical).
        """
        if self.state is None:
            return 0.0

        stack_before = self.player_stacks_before_action.get(seat, self.starting_stack)
        stack_after = self.state.players[seat].stack
        stack_change = stack_after - stack_before

        # Normalize by starting stack
        return float(stack_change) / self.starting_stack

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
