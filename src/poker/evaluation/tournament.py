"""Tournament and match evaluation harness."""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from poker.bots.base import Bot
from poker.engine.action_validator import legal_actions
from poker.ml.env import PokerEnv


@dataclass
class MatchStats:
    """Statistics from a single match between two bots."""

    player1_name: str
    player2_name: str
    num_hands: int
    player1_chip_change: float
    player2_chip_change: float
    player1_wins: int
    player2_wins: int
    draws: int

    @property
    def player1_win_rate(self) -> float:
        """Return player 1's win rate."""
        total = self.player1_wins + self.player2_wins
        return self.player1_wins / total if total > 0 else 0.0

    @property
    def player2_win_rate(self) -> float:
        """Return player 2's win rate."""
        total = self.player1_wins + self.player2_wins
        return self.player2_wins / total if total > 0 else 0.0

    def __str__(self) -> str:
        """Return human-readable string representation."""
        return (
            f"{self.player1_name} vs {self.player2_name}\n"
            f"  Hands: {self.num_hands}\n"
            f"  {self.player1_name}: {self.player1_wins} wins ({self.player1_win_rate:.1%}), "
            f"{self.player1_chip_change:+.2f} chips\n"
            f"  {self.player2_name}: {self.player2_wins} wins ({self.player2_win_rate:.1%}), "
            f"{self.player2_chip_change:+.2f} chips"
        )


@dataclass
class TournamentStats:
    """Statistics from a round-robin tournament."""

    bot_names: list[str]
    matches: list[MatchStats] = field(default_factory=list)

    def add_match(self, stats: MatchStats) -> None:
        """Add match statistics."""
        self.matches.append(stats)

    def get_standings(self) -> list[tuple[str, int, float]]:
        """Return standings as (name, wins, avg_chips_per_hand)."""
        standings = {}
        for match in self.matches:
            if match.player1_name not in standings:
                standings[match.player1_name] = {"wins": 0, "chips": 0.0, "hands": 0}
            if match.player2_name not in standings:
                standings[match.player2_name] = {"wins": 0, "chips": 0.0, "hands": 0}

            standings[match.player1_name]["wins"] += match.player1_wins
            standings[match.player1_name]["chips"] += match.player1_chip_change
            standings[match.player1_name]["hands"] += match.num_hands

            standings[match.player2_name]["wins"] += match.player2_wins
            standings[match.player2_name]["chips"] += match.player2_chip_change
            standings[match.player2_name]["hands"] += match.num_hands

        # Sort by wins
        result = []
        for name in sorted(standings.keys()):
            stats = standings[name]
            avg_chips = stats["chips"] / stats["hands"] if stats["hands"] > 0 else 0.0
            result.append((name, stats["wins"], avg_chips))

        return sorted(result, key=lambda x: x[1], reverse=True)

    def __str__(self) -> str:
        """Return human-readable tournament summary."""
        lines = ["Tournament Results:\n"]
        standings = self.get_standings()
        for rank, (name, wins, avg_chips) in enumerate(standings, 1):
            lines.append(f"  {rank}. {name}: {wins} wins, {avg_chips:+.3f} chips/hand")

        lines.append("\nDetailed Matches:")
        for match in self.matches:
            lines.append(f"  {match}")

        return "\n".join(lines)


class MatchEvaluator:
    """Evaluates a match between two bots."""

    def __init__(self, num_hands: int = 100, starting_stack: int = 1000) -> None:
        """Initialize the evaluator.

        Args:
            num_hands: Number of hands to play.
            starting_stack: Starting chip stack per player.
        """
        self.num_hands = num_hands
        self.starting_stack = starting_stack

    def evaluate(self, bot1: Bot, bot2: Bot) -> MatchStats:
        """Run a match between two bots.

        Args:
            bot1: First bot to evaluate.
            bot2: Second bot to evaluate.

        Returns:
            Statistics from the match.
        """
        # Track cumulative chip changes
        bot1_chip_change = 0.0
        bot2_chip_change = 0.0

        # Track hand outcomes
        bot1_wins = 0
        bot2_wins = 0
        draws = 0

        # Play hands
        for hand_idx in range(self.num_hands):
            # Run one hand with alternating dealer positions
            dealer = hand_idx % 2
            learning_bot = bot1 if dealer == 0 else bot2
            opponent_bot = bot2 if dealer == 0 else bot1

            env = PokerEnv(
                num_players=2,
                starting_stack=self.starting_stack,
                learning_seat=0,
                opponent_bots=[opponent_bot],
            )

            obs, info = env.reset()
            done = False

            while not done:
                # Get legal actions for the learning agent
                legal_acts = legal_actions(env.state, env.learning_seat)
                # Get the learning bot's action
                action = learning_bot.act(env.state.view_for(env.learning_seat), legal_acts)
                # Convert to action index and step
                from poker.ml.action_space import action_to_action_index
                action_idx = action_to_action_index(action, env.state, env.learning_seat)
                obs, reward, done, _, info = env.step(action_idx)

            # Record outcome
            learning_bot_chips = env.state.players[0].stack
            opponent_bot_chips = env.state.players[1].stack
            learning_bot_change = learning_bot_chips - self.starting_stack

            if learning_bot_change > 0:
                # Learning bot won this hand
                if learning_bot == bot1:
                    bot1_wins += 1
                    bot1_chip_change += learning_bot_change
                    bot2_chip_change -= learning_bot_change
                else:
                    bot2_wins += 1
                    bot2_chip_change += learning_bot_change
                    bot1_chip_change -= learning_bot_change
            elif learning_bot_change < 0:
                # Opponent won this hand
                opponent_change = -learning_bot_change
                if learning_bot == bot1:
                    bot2_wins += 1
                    bot2_chip_change += opponent_change
                    bot1_chip_change -= opponent_change
                else:
                    bot1_wins += 1
                    bot1_chip_change += opponent_change
                    bot2_chip_change -= opponent_change
            else:
                draws += 1

        return MatchStats(
            player1_name=bot1.name,
            player2_name=bot2.name,
            num_hands=self.num_hands,
            player1_chip_change=bot1_chip_change,
            player2_chip_change=bot2_chip_change,
            player1_wins=bot1_wins,
            player2_wins=bot2_wins,
            draws=draws,
        )


class TournamentEvaluator:
    """Runs a round-robin tournament between multiple bots."""

    def __init__(self, num_hands_per_match: int = 100, starting_stack: int = 1000) -> None:
        """Initialize the tournament.

        Args:
            num_hands_per_match: Number of hands per match.
            starting_stack: Starting chip stack per player.
        """
        self.num_hands_per_match = num_hands_per_match
        self.starting_stack = starting_stack
        self.match_evaluator = MatchEvaluator(num_hands_per_match, starting_stack)

    def run_tournament(self, bots: Sequence[Bot]) -> TournamentStats:
        """Run a round-robin tournament.

        Args:
            bots: List of bots to compete.

        Returns:
            Tournament statistics.
        """
        bot_names = [bot.name for bot in bots]
        stats = TournamentStats(bot_names=bot_names)

        # Play all matches
        for i in range(len(bots)):
            for j in range(i + 1, len(bots)):
                print(f"Playing {bots[i].name} vs {bots[j].name}...")
                match_stats = self.match_evaluator.evaluate(bots[i], bots[j])
                stats.add_match(match_stats)
                print(f"  {match_stats.player1_name}: {match_stats.player1_wins} wins, "
                      f"{match_stats.player1_chip_change:+.2f} chips")
                print(f"  {match_stats.player2_name}: {match_stats.player2_wins} wins, "
                      f"{match_stats.player2_chip_change:+.2f} chips")

        return stats
