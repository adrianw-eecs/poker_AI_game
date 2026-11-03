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
from poker.bots.nn_bot import NNBot, load_model as load_nn_model
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


_BOT_KINDS = {
    "human": "human",
    "random": "random",
    "random_bot": "random",
    "flop": "flop",
    "flop_bot": "flop",
    "nfsp": "nfsp",
    "nfsp_bot": "nfsp",
    "sdcfr": "sdcfr",
    "sdcfr_bot": "sdcfr",
    "nn": "nn",
    "nn_bot": "nn",
    "linear_bot": "linear",
    "deep_bot": "deep",
    "tree_bot": "tree",
}


def _split_bot_token(token: str) -> tuple[str, str | None]:
    """Return (kind, inline checkpoint path or None)."""
    kind_token, sep, path = token.partition(":")
    kind = _BOT_KINDS.get(kind_token.strip().lower())
    if kind is None:
        known = ", ".join(sorted(set(_BOT_KINDS)))
        raise ValueError(f"Unknown bot '{token}'. Expected one of: {known}")
    inline = path.strip() if sep else None
    if inline == "":
        raise ValueError(f"Bot '{token}' is missing a checkpoint path after ':'")
    return kind, inline


def _take_checkpoint(kind: str, inline: str | None, queued: list[str], seat: int) -> str:
    if inline is not None:
        return inline
    if queued:
        return queued.pop(0)
    flag = {"nfsp": "--nfsp-model", "sdcfr": "--sdcfr-model", "nn": "--nn-model"}[kind]
    raise ValueError(
        f"Seat {seat} ({kind}) needs a checkpoint. "
        f"Use {kind}:PATH or {flag}."
    )


def _require_checkpoint(path: str) -> Path:
    checkpoint = Path(path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")
    return checkpoint


class _ModelCache:
    """Load each checkpoint path once and reuse it across seats."""

    def __init__(self) -> None:
        self._nfsp: dict[str, object] = {}
        self._sdcfr: dict[str, object] = {}

    def nfsp(self, path: str):
        from poker.ml.models.nfsp_model import NFSPModel

        key = str(_require_checkpoint(path).resolve())
        model = self._nfsp.get(key)
        if model is None:
            model = NFSPModel()
            model.load(key)
            self._nfsp[key] = model
        return model

    def sdcfr(self, path: str):
        from poker.ml.models.sdcfr_model import SDCFRModel

        key = str(_require_checkpoint(path).resolve())
        model = self._sdcfr.get(key)
        if model is None:
            model = SDCFRModel()
            model.load(key)
            self._sdcfr[key] = model
        return model


def _build_bots(
    tokens: list[str],
    nfsp_paths: list[str],
    sdcfr_paths: list[str],
    nn_paths: list[str],
    seed: int | None,
) -> dict[int, Bot]:
    """Build one bot per seat token. Identical checkpoint paths share one model."""
    cache = _ModelCache()
    bots: dict[int, Bot] = {}
    human_count = 0

    for seat, token in enumerate(tokens):
        kind, inline = _split_bot_token(token)
        if kind == "human":
            human_count += 1
            name = "Human" if human_count == 1 else f"Human{human_count}"
            bots[seat] = HumanBot(seat=seat, name=name)
        elif kind == "random":
            bots[seat] = RandomBot(name=f"RandomBot{seat + 1}", seed=seed)
        elif kind == "flop":
            bots[seat] = FlopBot(name=f"FlopBot{seat + 1}")
        elif kind == "nfsp":
            from poker.bots.nfsp_bot import NFSPBot

            path = _take_checkpoint(kind, inline, nfsp_paths, seat)
            model = cache.nfsp(path)
            bots[seat] = NFSPBot(name=f"NFSPBot{seat + 1}", model=model, training=False)
        elif kind == "sdcfr":
            from poker.bots.sdcfr_bot import SDCFRBot

            path = _take_checkpoint(kind, inline, sdcfr_paths, seat)
            model = cache.sdcfr(path)
            bots[seat] = SDCFRBot(name=f"SDCFRBot{seat + 1}", model=model)
        elif kind == "nn":
            path = _take_checkpoint(kind, inline, nn_paths, seat)
            model = load_nn_model(path)
            bots[seat] = NNBot(name=f"NNBot{seat + 1}", model=model)
        elif kind == "linear":
            from poker.bots.linear_bot import LinearBot
            from poker.ml.models.linear_q import LinearQModel

            model = LinearQModel()
            model.load("models/linear_q.pkl")
            bots[seat] = LinearBot(name=f"LinearBot{seat + 1}", model=model)
        elif kind == "deep":
            from poker.bots.deep_bot import DeepBot
            from poker.ml.models.deep_q import DeepQModel

            model = DeepQModel()
            model.load("models/deep_q.pt")
            bots[seat] = DeepBot(name=f"DeepBot{seat + 1}", model=model)
        else:
            from poker.bots.tree_bot import TreeBot
            from poker.ml.models.tree_q import TreeQModel

            model = TreeQModel()
            model.load("models/tree_q.pkl")
            bots[seat] = TreeBot(name=f"TreeBot{seat + 1}", model=model)

    return bots


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
        help="Number of players (2-9, default: 2)",
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
        help=(
            "One seat token per player, in seat order. "
            "Kinds: human, random, flop, nfsp, sdcfr, nn. "
            "Model seats accept an inline checkpoint (nfsp:PATH, sdcfr:PATH, nn:PATH). "
            "Default: every seat is random."
        ),
    )
    parser.add_argument(
        "--nfsp-model",
        action="append",
        default=None,
        help="NFSP checkpoint for the next nfsp seat that has no inline path. Repeat for several models.",
    )
    parser.add_argument(
        "--sdcfr-model",
        action="append",
        default=None,
        help="SDCFR checkpoint for the next sdcfr seat that has no inline path. Repeat for several models.",
    )
    parser.add_argument(
        "--nn-model",
        action="append",
        default=None,
        help="Checkpoint for the next nn seat that has no inline path. The network is not implemented yet.",
    )
    parser.add_argument(
        "--rake",
        type=float,
        default=5.0,
        help="Rake percentage (0.0 to 10.0, default: 5.0)",
    )

    # Other options
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. If not set, a random seed is used.",
    )
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
        if not 2 <= args.num_players <= 9:
            print(f"Error: num_players must be 2-9, got {args.num_players}")
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
            rake_percent=args.rake,
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

        games_dir = Path("games/AI_games")
        games_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcript_logger = SessionTextLogger(games_dir / f"session_{ts}.txt", seed=args.seed)

        logger = MultiLogger([event_logger, transcript_logger])
        action_handler = ActionHandler(logger)

        # Create session
        session = Session(
            config=config,
            blind_schedule=blind_schedule,
            session_config=session_config,
            logger=logger,
            action_handler=action_handler,
            seed=args.seed,
        )

        # Create bots. Model checkpoints are loaded once per distinct path.
        bot_tokens = args.bots if args.bots else ["random"] * args.num_players
        if len(bot_tokens) != args.num_players:
            print(
                f"Error: --num-players is {args.num_players} but --bots has "
                f"{len(bot_tokens)} seat(s)"
            )
            return 1
        bots = _build_bots(
            bot_tokens,
            list(args.nfsp_model or []),
            list(args.sdcfr_model or []),
            list(args.nn_model or []),
            args.seed,
        )

        # Create initial state
        state = session.create_initial_state(args.num_players)

        print("=" * 60)
        print("Texas Hold'em Poker Engine")
        print("=" * 60)
        print(f"Players: {args.num_players}")
        for seat in range(args.num_players):
            print(f"  Seat {seat}: {bots[seat].name}")
        print(f"Starting stack: {args.starting_stack}")
        print(f"Blinds: {args.small_blind}/{args.big_blind}")
        if args.ante:
            print(f"Ante: {args.ante}")
        if args.hands:
            print(f"Duration: {args.hands} hands")
        if args.time:
            print(f"Duration: {args.time} seconds")
        if args.seed is not None:
            print(f"Random Seed: {args.seed}")
        print("=" * 60)
        print()

        # Run session
        rng = RNG(seed=args.seed)  # Create RNG for deck shuffling with optional seed

        def deck_factory() -> Deck:
            """Create a new shuffled deck."""
            deck = Deck(rng=rng)
            return deck

        final_state = session.run(state, bots, deck_factory)

        # Display final results
        print()
        print("=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        results = session.get_final_results(final_state)

        # Format final results for both console and log file
        results_lines = [
            "",
            "=" * 60,
            "FINAL RESULTS",
            "=" * 60,
        ]
        for seat in sorted(results.keys()):
            stack = results[seat]
            bot_name = bots[seat].name if seat in bots else f"Player{seat + 1}"
            change = stack - args.starting_stack
            change_str = f"+{change}" if change > 0 else str(change)
            line = f"{bot_name:20} {stack:6} ({change_str:>6})"
            results_lines.append(line)
            print(line)

        results_lines.append("=" * 60)
        print("=" * 60)

        # Log final results to transcript file
        transcript_logger.log_text("\n".join(results_lines))

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
