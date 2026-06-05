#!/usr/bin/env python
"""Interactive poker game: Human vs NFSP Model, RandomBot, and FlopBot."""

import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from poker.ml.env import PokerEnv
from poker.ml.models.nfsp_model import NFSPModel
from poker.bots.base import Bot
from poker.bots.random_bot import RandomBot
from poker.bots.flop_bot import FlopBot


class NFSPBot(Bot):
    """Bot wrapper for NFSP model."""

    def __init__(self, model: NFSPModel, seat: int = 1):
        """Initialize with trained NFSP model.

        Args:
            model: Trained NFSPModel instance
            seat: Seat number for this bot (for observation building)
        """
        super().__init__()
        self.model = model
        self.seat = seat

    def act(self, game_state, legal_actions_list):
        """Choose action using NFSP model.

        Args:
            game_state: Game state view for this bot
            legal_actions_list: List of legal Action objects

        Returns:
            Action object (best action from legal actions)
        """
        # Convert game state to observation
        from poker.ml.observation import build_observation
        from poker.ml.action_space import build_action_mask, action_index_to_action

        obs = build_observation(game_state, self.seat)

        # Build binary mask from legal actions list
        mask = build_action_mask(game_state, self.seat)

        # Get model's action (returns index 0-6)
        action_index = self.model.select_action(obs, mask, training=False)

        # Convert action index to Action object
        action = action_index_to_action(action_index, game_state, self.seat)

        # If action is not in legal actions, fall back to first legal action
        if action not in legal_actions_list:
            return legal_actions_list[0]
        return action


class InteractivePokerGame:
    """Interactive 4-player poker game with human player."""

    def __init__(self, model_path: str, num_hands: int = 3):
        """Initialize the game.

        Args:
            model_path: Path to trained NFSP model checkpoint
            num_hands: Number of hands to play
        """
        self.model_path = model_path
        self.num_hands = num_hands
        self.hands_played = 0
        self.cumulative_profit = 0.0

        # Load trained model
        print("\n[LOADING] Loading trained model...")
        nfsp_model = NFSPModel()
        nfsp_model.load(model_path)
        print(f"[OK] Model loaded from {model_path}\n")

        # Create environment with human at seat 0
        self.env = PokerEnv(
            num_players=4,
            learning_seat=0,  # Human is at seat 0
            opponent_bots=[
                NFSPBot(nfsp_model),  # Seat 1: NFSP Model
                RandomBot(seed=None),  # Seat 2: RandomBot
                FlopBot(seed=None),  # Seat 3: FlopBot
            ],
            small_blind=25,
            big_blind=50,
            starting_stack=1000,
            seed=None,
        )

    def get_player_name(self, seat: int) -> str:
        """Get friendly name for a player seat."""
        names = {
            0: "YOU (Human)",
            1: "NFSP Model",
            2: "RandomBot",
            3: "FlopBot",
        }
        return names.get(seat, f"Player {seat}")

    def cards_to_string(self, cards) -> str:
        """Convert card objects to readable string with suit symbols."""
        if not cards:
            return "(empty)"

        card_strs = []
        for card in cards:
            card_str = str(card)
            # Remove ANSI color codes (keep suit symbols)
            card_str = re.sub(r'\x1b\[[0-9;]*m', '', card_str)
            card_strs.append(card_str)

        return " ".join(card_strs)

    def display_game_state(self):
        """Display current game state."""
        state = self.env.state
        if state is None:
            return

        print("\n" + "=" * 70)
        print(f"STREET: {state.street.name}")
        print("=" * 70)

        # Show community cards
        community_str = self.cards_to_string(state.community_cards) if state.community_cards else "(not dealt)"
        print(f"Community Cards: {community_str}")

        # Show pot
        total_pot = sum(p.committed_this_hand for p in state.players)
        print(f"Current Pot: ${total_pot}")

        # Show all players
        print("\nPlayers:")
        print("-" * 70)
        for i, player in enumerate(state.players):
            if player.is_eliminated:
                status = "[ELIMINATED]"
            elif player.has_folded:
                status = "[FOLDED]"
            elif player.is_all_in:
                status = "[ALL-IN]"
            else:
                status = ""

            seat_marker = " <-- YOUR SEAT" if i == 0 else ""
            print(f"Seat {i}: {self.get_player_name(i):20s} Stack: ${player.stack:6d}  {status}{seat_marker}")

            # Show hole cards
            # - Always show human's cards
            # - At showdown, show everyone's cards (if not folded)
            # - Before showdown, show only human's cards
            if player.hole_cards and (i == 0 or state.street.name == "SHOWDOWN"):
                if not player.has_folded or state.street.name == "SHOWDOWN":
                    cards_str = self.cards_to_string(player.hole_cards)
                    if i == 0:
                        print(f"         Your cards: {cards_str}")
                    else:
                        print(f"         Cards: {cards_str}")

        print("-" * 70)

    def get_human_action(self) -> int:
        """Get action from human player.

        Returns:
            Action index (0-6)
        """
        mask = self.env.get_action_mask()
        legal_actions = [i for i in range(7) if mask[i]]

        action_names = ["FOLD", "CHECK", "CALL", "RAISE-0.5x", "RAISE-POT", "RAISE-2x", "ALL-IN"]
        legal_names = [action_names[i] for i in legal_actions]

        print(f"\nYour legal actions: {', '.join(legal_names)}")
        print("Action options:")
        for idx, (action_idx, action_name) in enumerate(zip(legal_actions, legal_names), 1):
            print(f"  {idx}. {action_name}")

        while True:
            choice = input("\nEnter action number (1-{}): ".format(len(legal_actions))).strip()
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(legal_actions):
                    return legal_actions[choice_num - 1]
                else:
                    print(f"Invalid choice. Please enter 1-{len(legal_actions)}")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def play_hand(self):
        """Play a single hand."""
        print(f"\n\n{'#' * 70}")
        print(f"HAND {self.hands_played + 1} of {self.num_hands}")
        print(f"{'#' * 70}")

        # Reset environment for new hand
        obs, _ = self.env.reset()

        # Play until hand is done
        done = False
        action_count = 0

        while not done:
            # Display game state
            self.display_game_state()

            # Check whose turn it is
            if self.env.state.action_on_seat == 0:
                # Human's turn
                print("\n[YOUR TURN]")
                action = self.get_human_action()
                action_name = ["FOLD", "CHECK", "CALL", "RAISE-0.5x", "RAISE-POT", "RAISE-2x", "ALL-IN"][action]
                print(f"You chose: {action_name}")
            else:
                # Bot's turn
                bot_seat = self.env.state.action_on_seat
                bot_name = self.get_player_name(bot_seat)
                mask = self.env.get_action_mask()
                action = self.env.opponent_bots[bot_seat - 1].act(obs, mask, training=False)
                action_name = ["FOLD", "CHECK", "CALL", "RAISE-0.5x", "RAISE-POT", "RAISE-2x", "ALL-IN"][action]
                print(f"\n[{bot_name.upper()}]")
                print(f"{bot_name} chose: {action_name}")
                input("Press Enter to continue...")

            # Execute action
            obs, reward, done, _, _ = self.env.step(action)
            action_count += 1

            if done:
                # Hand is over
                self.display_game_state()
                break

        # Show hand result
        print("\n" + "=" * 70)
        print("HAND RESULT")
        print("=" * 70)

        # Calculate final stacks
        for i, player in enumerate(self.env.state.players):
            stack_change = player.stack - 1000  # Starting stack was 1000
            if stack_change > 0:
                result = f"WON ${stack_change}"
            elif stack_change < 0:
                result = f"LOST ${abs(stack_change)}"
            else:
                result = "BREAK-EVEN"

            marker = " <-- YOU" if i == 0 else ""
            print(f"{self.get_player_name(i):20s} Stack: ${player.stack:6d}  ({result}){marker}")

        your_stack = self.env.state.players[0].stack
        self.cumulative_profit += (your_stack - 1000)

        print("=" * 70)
        print(f"Your cumulative profit: ${self.cumulative_profit:.2f}")

        self.hands_played += 1

    def run(self):
        """Play multiple hands."""
        print("\n" + "=" * 70)
        print("INTERACTIVE POKER GAME")
        print("=" * 70)
        print(f"Players: Human vs NFSP Model vs RandomBot vs FlopBot")
        print(f"Hands to play: {self.num_hands}")
        print(f"Starting stack: $1000")
        print(f"Blinds: $25/$50")
        print("=" * 70)

        try:
            for _ in range(self.num_hands):
                self.play_hand()

                if self.hands_played < self.num_hands:
                    cont = input(f"\nContinue to hand {self.hands_played + 1}? (y/n): ").strip().lower()
                    if cont != 'y':
                        break

        except KeyboardInterrupt:
            print("\n\n[INTERRUPTED] Game stopped by user")

        # Show summary
        print("\n" + "=" * 70)
        print("GAME SUMMARY")
        print("=" * 70)
        print(f"Hands played: {self.hands_played}")
        print(f"Your total profit: ${self.cumulative_profit:.2f}")
        if self.hands_played > 0:
            print(f"Profit per hand: ${self.cumulative_profit / self.hands_played:.2f}")
        print("=" * 70)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Play interactive poker vs trained model")
    parser.add_argument("model_path", help="Path to trained NFSP model checkpoint (.pt file)")
    parser.add_argument("--hands", type=int, default=3, help="Number of hands to play (default: 3)")

    args = parser.parse_args()

    # Verify model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    # Run game
    game = InteractivePokerGame(str(model_path), num_hands=args.hands)
    game.run()


if __name__ == "__main__":
    main()
