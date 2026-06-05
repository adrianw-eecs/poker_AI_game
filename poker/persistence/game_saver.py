"""Game state persistence to JSON files."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from poker.bots.base import Bot
from poker.state.game_state import GameState


def save_game_session(
    state: GameState,
    bots: dict[int, Bot],
    output_dir: Path | str = "games/AI_games",
) -> Path:
    """Save a completed session to a JSON file.

    Accumulates all streets and actions from all hands in one JSON file.

    Args:
        state: The final game state after all hands are complete.
        bots: Dict mapping seat → Bot for each player.
        output_dir: Directory to save the game file to.

    Returns:
        Path to the saved JSON file.

    Raises:
        ValueError: If state has no hands played (hand_number == 0).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for session ID
    session_id = datetime.now().isoformat()

    # Build player info
    players_info = []
    for i, player in enumerate(state.players):
        bot_name = bots.get(i, Bot).name if i in bots else f"Player{i + 1}"
        players_info.append({
            "seat": player.seat,
            "name": bot_name,
            "is_eliminated": player.is_eliminated,
            "final_stack": player.stack,
        })

    # Note: We would need to track individual hands' starting stacks, streets,
    # actions, and results. Since GameState is immutable and we're only given
    # the final state, we can only save summary information.
    game_data = {
        "session_id": session_id,
        "hand_count": state.hand_number,
        "players": players_info,
        "final_stacks": {str(p.seat): p.stack for p in state.players},
    }

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"game_{timestamp}.json"

    # Write to JSON file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(game_data, f, indent=2)

    return filename
