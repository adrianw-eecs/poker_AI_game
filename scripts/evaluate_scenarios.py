"""Evaluate NFSP and SD-CFR bots across three benchmark scenarios."""

import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poker.bots.flop_bot import FlopBot
from poker.bots.nfsp_bot import NFSPBot
from poker.bots.random_bot import RandomBot
from poker.bots.sdcfr_bot import SDCFRBot
from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.domain.deck import Deck
from poker.engine.session import Session, SessionConfig
from poker.logging.logger import NullLogger
from poker.ml.models.nfsp_model import NFSPModel
from poker.ml.models.sdcfr_model import SDCFRModel
from poker.rng import RNG


def make_config(num_players: int) -> tuple[GameConfig, BlindSchedule]:
    schedule = BlindSchedule(
        levels=[BlindLevel(small=5, big=10, ante=0)] * 20,
        hands_per_level=20,
        fixed=True,
    )
    cfg = GameConfig(
        num_players=num_players,
        starting_stack=1000,
        small_blind=5,
        big_blind=10,
        ante=0,
        rake_percent=0.0,
        rake_cap=None,
        blind_schedule=schedule,
        run_it_twice=False,
    )
    return cfg, schedule


def run_scenario(chosen_bot, bot_label: str, opponents: dict, num_hands: int,
                 rebuy: bool, seed: int = 42) -> dict:
    num_players = 1 + len(opponents)
    cfg, schedule = make_config(num_players)

    session = Session(
        config=cfg,
        blind_schedule=schedule,
        session_config=SessionConfig(
            duration_hands=num_hands,
            rebuy_enabled=rebuy,
            rebuy_stack=cfg.starting_stack,
        ),
        logger=NullLogger(),
    )

    bots = {0: chosen_bot}
    bots.update(opponents)

    state = session.create_initial_state(num_players)
    rng = RNG(seed=seed)
    final = session.run(state, bots, lambda: Deck(rng=RNG(seed=rng.randint(0, 2**31 - 1))))

    results = {}
    for p in final.players:
        name = bots[p.seat].name if p.seat in bots else f"Player{p.seat+1}"
        results[name] = {
            "final_stack": p.stack,
            "change": p.stack - cfg.starting_stack,
        }
    return results


def print_results(scenario_name: str, results: dict, starting_stack: int = 1000) -> None:
    print(f"\n  {scenario_name}")
    print(f"  {'Player':<22} {'Final':>7} {'Change':>8}")
    print(f"  {'-'*40}")
    for name, data in results.items():
        ch = data["change"]
        sign = "+" if ch > 0 else ""
        print(f"  {name:<22} {data['final_stack']:>7}  {sign}{ch:>6}")


def main() -> None:
    # Load models
    nfsp_model = NFSPModel()
    nfsp_model.load("models/nfsp.pt")

    sdcfr_model = SDCFRModel()
    sdcfr_model.load("models/sdcfr.pt")

    print("=" * 60)
    print("BENCHMARK RESULTS — NFSP Bot")
    print("=" * 60)

    # ── Scenario 1: 4 players, 10 hands, rebuy ON ──────────────
    nfsp_bot = NFSPBot(name="NFSPBot", model=nfsp_model)
    opponents = {
        1: RandomBot(name="RandomBot1", seed=101),
        2: RandomBot(name="RandomBot2", seed=102),
        3: FlopBot(name="FlopBot1", seed=103),
    }
    r = run_scenario(nfsp_bot, "NFSPBot", opponents,
                     num_hands=10, rebuy=True, seed=1)
    print_results("Scenario 1 — 4 players | 10 hands | rebuy ON", r)

    # ── Scenario 2: 6 players, 20 hands, rebuy ON ──────────────
    nfsp_bot = NFSPBot(name="NFSPBot", model=nfsp_model)
    opponents = {
        1: RandomBot(name="RandomBot1", seed=201),
        2: RandomBot(name="RandomBot2", seed=202),
        3: RandomBot(name="RandomBot3", seed=203),
        4: RandomBot(name="RandomBot4", seed=204),
        5: FlopBot(name="FlopBot1", seed=205),
    }
    r = run_scenario(nfsp_bot, "NFSPBot", opponents,
                     num_hands=20, rebuy=True, seed=2)
    print_results("Scenario 2 — 6 players | 20 hands | rebuy ON", r)

    # ── Scenario 3: 8 players, 10 hands, rebuy OFF ─────────────
    nfsp_bot = NFSPBot(name="NFSPBot", model=nfsp_model)
    opponents = {
        1: RandomBot(name="RandomBot1", seed=301),
        2: RandomBot(name="RandomBot2", seed=302),
        3: RandomBot(name="RandomBot3", seed=303),
        4: RandomBot(name="RandomBot4", seed=304),
        5: RandomBot(name="RandomBot5", seed=305),
        6: FlopBot(name="FlopBot1", seed=306),
        7: FlopBot(name="FlopBot2", seed=307),
    }
    r = run_scenario(nfsp_bot, "NFSPBot", opponents,
                     num_hands=10, rebuy=False, seed=3)
    print_results("Scenario 3 — 8 players | 10 hands | rebuy OFF", r)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS — SD-CFR Bot")
    print("=" * 60)

    # ── Scenario 1: 4 players, 10 hands, rebuy ON ──────────────
    sdcfr_bot = SDCFRBot(name="SDCFRBot", model=sdcfr_model)
    opponents = {
        1: RandomBot(name="RandomBot1", seed=101),
        2: RandomBot(name="RandomBot2", seed=102),
        3: FlopBot(name="FlopBot1", seed=103),
    }
    r = run_scenario(sdcfr_bot, "SDCFRBot", opponents,
                     num_hands=10, rebuy=True, seed=1)
    print_results("Scenario 1 — 4 players | 10 hands | rebuy ON", r)

    # ── Scenario 2: 6 players, 20 hands, rebuy ON ──────────────
    sdcfr_bot = SDCFRBot(name="SDCFRBot", model=sdcfr_model)
    opponents = {
        1: RandomBot(name="RandomBot1", seed=201),
        2: RandomBot(name="RandomBot2", seed=202),
        3: RandomBot(name="RandomBot3", seed=203),
        4: RandomBot(name="RandomBot4", seed=204),
        5: FlopBot(name="FlopBot1", seed=205),
    }
    r = run_scenario(sdcfr_bot, "SDCFRBot", opponents,
                     num_hands=20, rebuy=True, seed=2)
    print_results("Scenario 2 — 6 players | 20 hands | rebuy ON", r)

    # ── Scenario 3: 8 players, 10 hands, rebuy OFF ─────────────
    sdcfr_bot = SDCFRBot(name="SDCFRBot", model=sdcfr_model)
    opponents = {
        1: RandomBot(name="RandomBot1", seed=301),
        2: RandomBot(name="RandomBot2", seed=302),
        3: RandomBot(name="RandomBot3", seed=303),
        4: RandomBot(name="RandomBot4", seed=304),
        5: RandomBot(name="RandomBot5", seed=305),
        6: FlopBot(name="FlopBot1", seed=306),
        7: FlopBot(name="FlopBot2", seed=307),
    }
    r = run_scenario(sdcfr_bot, "SDCFRBot", opponents,
                     num_hands=10, rebuy=False, seed=3)
    print_results("Scenario 3 — 8 players | 10 hands | rebuy OFF", r)

    print()


if __name__ == "__main__":
    main()
