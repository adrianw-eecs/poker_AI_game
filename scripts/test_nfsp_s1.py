#!/usr/bin/env python
"""Test NFSP Scenario 1: 4 players, 10 hands, rebuy ON."""

import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poker.bots.nfsp_bot import NFSPBot
from poker.bots.random_bot import RandomBot
from poker.bots.flop_bot import FlopBot
from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.domain.deck import Deck
from poker.engine.session import Session, SessionConfig
from poker.logging.logger import GameLogger
from poker.logging.session_text_logger import SessionTextLogger
from poker.logging.multi_logger import MultiLogger
from poker.stats.game_analyzer import GameStatsAnalyzer
from datetime import datetime
from poker.ml.models.nfsp_model import NFSPModel
from poker.rng import RNG


def main():
    # Load NFSP model
    nfsp_model = NFSPModel()
    nfsp_model.load("models/nfsp.pt")

    # Create config for 4 players
    schedule = BlindSchedule(
        levels=[BlindLevel(small=5, big=10, ante=0)] * 20,
        hands_per_level=20,
        fixed=True,
    )
    cfg = GameConfig(
        num_players=4,
        starting_stack=1000,
        small_blind=5,
        big_blind=10,
        ante=0,
        rake_percent=0.0,
        rake_cap=None,
        blind_schedule=schedule,
        run_it_twice=False,
    )

    # Create loggers
    games_dir = Path("games/AI_games")
    games_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    event_logger = GameLogger(games_dir / f"nfsp_s1_{ts}.jsonl")
    transcript_logger = SessionTextLogger(games_dir / f"nfsp_s1_{ts}.txt")
    logger = MultiLogger([event_logger, transcript_logger])

    # Create session
    session = Session(
        config=cfg,
        blind_schedule=schedule,
        session_config=SessionConfig(
            duration_hands=10,
            rebuy_enabled=True,
            rebuy_stack=cfg.starting_stack,
        ),
        logger=logger,
    )

    # Create bots
    bots = {
        0: NFSPBot(name="NFSPBot", model=nfsp_model),
        1: RandomBot(name="RandomBot1", seed=101),
        2: RandomBot(name="RandomBot2", seed=102),
        3: FlopBot(name="FlopBot1", seed=103),
    }

    # Create initial state and run
    state = session.create_initial_state(4)
    rng = RNG(seed=1)
    final = session.run(state, bots, lambda: Deck(rng=RNG(seed=rng.randint(0, 2**31 - 1))))

    # Print basic results
    print("\n" + "=" * 80)
    print("NFSP SCENARIO 1 — 4 players | 10 hands | rebuy ON")
    print("=" * 80)
    print(f"{'Player':<22} {'Final Stack':>15} {'Change':>12}")
    print("-" * 80)
    for p in final.players:
        name = bots[p.seat].name if p.seat in bots else f"Player{p.seat+1}"
        change = p.stack - cfg.starting_stack
        change_str = f"+{change}" if change > 0 else str(change)
        print(f"{name:<22} {p.stack:>15}  {change_str:>12}")
    print("=" * 80)

    # Analyze game statistics from logs
    log_file = games_dir / f"nfsp_s1_{ts}.jsonl"
    if log_file.exists():
        seat_to_bot = {
            0: "NFSPBot",
            1: "RandomBot1",
            2: "RandomBot2",
            3: "FlopBot1",
        }
        final_stacks = {p.seat: p.stack for p in final.players}
        analyzer = GameStatsAnalyzer(log_file, seat_to_bot, final_stacks)
        print(analyzer.get_summary())


if __name__ == "__main__":
    main()
