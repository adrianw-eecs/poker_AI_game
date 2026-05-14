"""Command-line entry point for starting and managing poker games."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Enable UTF-8 output on Windows for Unicode suit symbols
if sys.platform == "win32":
    # Reconfigure stdout to use UTF-8 encoding
    sys.stdout.reconfigure(encoding="utf-8")

from poker.bots.base import Bot
from poker.bots.flop_bot import FlopBot
from poker.bots.human_bot import HumanBot
from poker.bots.random_bot import RandomBot
from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.domain.deck import Deck
from poker.engine.action_handler import ActionHandler
from poker.engine.session import Session, SessionConfig
from poker.interface.text_ui import render
from poker.logging.logger import GameLogger, NullLogger
from poker.logging.multi_logger import MultiLogger
from poker.logging.session_text_logger import SessionTextLogger
from poker.persistence import save_game_session
from poker.rng import RNG
from poker.state.game_state import GameState


def _create_blind_schedule(
    initial_small_blind: int,
    initial_big_blind: int,
    blind_increase_factor: float = 1.0,
    hands_per_level: int = 10,
    num_levels: int = 5,
) -> BlindSchedule:
    """Create a blind schedule with escalating levels.

    Args:
        initial_small_blind: Starting small blind.
        initial_big_blind: Starting big blind.
        blind_increase_factor: Multiplier for each level (1.0 = fixed).
        hands_per_level: Hands played before advancing to next level.
        num_levels: Number of blind levels to create.

    Returns:
        A BlindSchedule instance.
    """
    levels = []
    sb = initial_small_blind
    bb = initial_big_blind

    for _ in range(num_levels):
        levels.append(BlindLevel(small=sb, big=bb, ante=0))
        if blind_increase_factor != 1.0:
            sb = max(1, int(sb * blind_increase_factor))
            bb = max(sb + 1, int(bb * blind_increase_factor))

    # If fixed, use only the first level
    is_fixed = blind_increase_factor == 1.0

    return BlindSchedule(
        levels=levels, hands_per_level=hands_per_level, fixed=is_fixed
    )


def main() -> int:
    """Main entry point for the poker game CLI.

    Parses command-line arguments, initializes bots and config,
    creates a session, runs it, and displays results.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(
        description="Texas Hold'em Poker Engine with ML Self-Play"
    )

    # Game setup arguments
    parser.add_argument(
        "-n",
        "--num-players",
        type=int,
        default=2,
        help="Number of players (2-6, default: 2)",
    )
    parser.add_argument(
        "-s",
        "--starting-stack",
        type=int,
        default=1000,
        help="Starting chip stack per player (default: 1000)",
    )
    parser.add_argument(
        "-sb",
        "--small-blind",
        type=int,
        default=5,
        help="Small blind in chips (default: 5)",
    )
    parser.add_argument(
        "-bb",
        "--big-blind",
        type=int,
        default=10,
        help="Big blind in chips (default: 10)",
    )
    parser.add_argument(
        "-a",
        "--ante",
        type=int,
        default=0,
        help="Ante per player in chips (default: 0)",
    )

    # Session duration
    parser.add_argument(
        "-hh",
        "--hands",
        type=int,
        default=None,
        help="Maximum hands to play (default: unlimited)",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=float,
        default=None,
        help="Maximum session duration in seconds (default: unlimited)",
    )

    # Logging
    parser.add_argument(
        "-l",
        "--log",
        type=str,
        default=None,
        help="Log file path (JSONL format). If not set, no logging.",
    )

    # Blind schedule
    parser.add_argument(
        "--blind-increase",
        type=float,
        default=1.0,
        help="Blind increase factor per level (default: 1.0 = fixed)",
    )
    parser.add_argument(
        "--hands-per-level",
        type=int,
        default=10,
        help="Hands before advancing blind level (default: 10)",
    )

    # Bot configuration
    parser.add_argument(
        "-b",
        "--bots",
        type=str,
        nargs="*",
        default=None,
        help="Bot types for each seat: random, human, flop_bot (default: all random)",
    )

    # Other options
    parser.add_argument(
        "--run-it-twice",
        action="store_true",
        help="Enable run-it-twice for all-in situations",
    )

    # Rebuy options
    parser.add_argument(
        "--rebuy",
        action="store_true",
        help="Enable automatic rebuy when players hit 0 chips",
    )
    parser.add_argument(
        "--rebuy-stack",
        type=int,
        default=None,
        help="Stack amount to rebuy to (default: starting stack)",
    )

    args = parser.parse_args()

    try:
        # Validate player count
        if not 2 <= args.num_players <= 10:
            print(f"Error: num_players must be 2-10, got {args.num_players}")
            return 1

        # Create game config
        blind_schedule = _create_blind_schedule(
            initial_small_blind=args.small_blind,
            initial_big_blind=args.big_blind,
            blind_increase_factor=args.blind_increase,
            hands_per_level=args.hands_per_level,
        )

        config = GameConfig(
            num_players=args.num_players,
            starting_stack=args.starting_stack,
            small_blind=args.small_blind,
            big_blind=args.big_blind,
            ante=args.ante,
            rake_percent=5.0,
            rake_cap=None,
            blind_schedule=blind_schedule,
            run_it_twice=args.run_it_twice,
        )

        # Create session config
        session_config = SessionConfig(
            duration_hands=args.hands,
            duration_seconds=args.time,
            rebuy_enabled=args.rebuy,
            rebuy_stack=args.rebuy_stack
        )

        # Create logger
        if args.log:
            event_logger = GameLogger(Path(args.log))
        else:
            event_logger = NullLogger()

        # Create action handler and per-session transcript in games/
        games_dir = Path("games/AI_games")
        games_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        action_handler = ActionHandler()

        transcript_path = games_dir / f"session_{ts}.txt"
        transcript_logger = SessionTextLogger(transcript_path)

        logger = MultiLogger([event_logger, transcript_logger])

        # Create session
        session = Session(
            config=config,
            blind_schedule=blind_schedule,
            session_config=session_config,
            logger=logger,
            action_handler=action_handler,
        )

        # Create bots
        bots: dict[int, Bot] = {}
        bot_types = args.bots or ["random"] * args.num_players

        for seat, bot_type in enumerate(bot_types[: args.num_players]):
            if bot_type == "human":
                bots[seat] = HumanBot(seat=seat)
            elif bot_type == "flop_bot":
                bots[seat] = FlopBot(name=f"FlopBot{seat + 1}")
            elif bot_type == "linear_bot":
                from poker.bots.linear_bot import LinearBot
                from poker.ml.models.linear_q import LinearQModel
                model = LinearQModel()
                model.load("models/linear_q.pkl")
                bots[seat] = LinearBot(name=f"LinearBot{seat + 1}", model=model)
            elif bot_type == "deep_bot":
                from poker.bots.deep_bot import DeepBot
                from poker.ml.models.deep_q import DeepQModel
                model = DeepQModel()
                model.load("models/deep_q.pt")
                bots[seat] = DeepBot(name=f"DeepBot{seat + 1}", model=model)
            elif bot_type == "tree_bot":
                from poker.bots.tree_bot import TreeBot
                from poker.ml.models.tree_q import TreeQModel
                model = TreeQModel()
                model.load("models/tree_q.pkl")
                bots[seat] = TreeBot(name=f"TreeBot{seat + 1}", model=model)
            elif bot_type == "nfsp_bot":
                from poker.bots.nfsp_bot import NFSPBot
                from poker.ml.models.nfsp_model import NFSPModel
                model = NFSPModel()
                model.load("models/nfsp.pt")
                bots[seat] = NFSPBot(name=f"NFSPBot{seat + 1}", model=model)
            elif bot_type == "sdcfr_bot":
                from poker.bots.sdcfr_bot import SDCFRBot
                from poker.ml.models.sdcfr_model import SDCFRModel
                model = SDCFRModel()
                model.load("models/sdcfr.pt")
                bots[seat] = SDCFRBot(name=f"SDCFRBot{seat + 1}", model=model)
            else:  # default to random
                bots[seat] = RandomBot(name=f"RandomBot{seat + 1}")

        # Create initial state
        state = session.create_initial_state(args.num_players)

        print("=" * 60)
        print("Texas Hold'em Poker Engine")
        print("=" * 60)
        print(f"Players: {args.num_players}")
        print(f"Starting stack: {args.starting_stack}")
        print(f"Blinds: {args.small_blind}/{args.big_blind}")
        if args.ante:
            print(f"Ante: {args.ante}")
        if args.hands:
            print(f"Duration: {args.hands} hands")
        if args.time:
            print(f"Duration: {args.time} seconds")
        print("=" * 60)
        print()

        # Run session
        rng = RNG()  # Create RNG for deck shuffling

        def deck_factory() -> Deck:
            """Create a new shuffled deck."""
            deck = Deck(rng=rng)
            return deck

        final_state = session.run(state, bots, deck_factory)

        # Save game session to JSON
        save_game_session(final_state, bots, output_dir="games/AI_games")

        # Display final results
        print()
        print("=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        results = session.get_final_results(final_state)
        for seat in sorted(results.keys()):
            stack = results[seat]
            bot_name = bots[seat].name if seat in bots else f"Player{seat + 1}"
            change = stack - args.starting_stack
            change_str = f"+{change}" if change > 0 else str(change)
            print(f"{bot_name:20} {stack:6} ({change_str:>6})")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
