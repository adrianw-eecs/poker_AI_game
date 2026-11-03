"""Tests for hand replay from JSONL logs."""

import json
from pathlib import Path

import pytest

from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.logging.events import HandEnded, HandStarted
from poker.logging.replay import Replay


def test_replay_empty_log(tmp_path: Path) -> None:
    log_file = tmp_path / "empty.jsonl"
    log_file.write_text("")
    replay = Replay(log_file)
    assert len(replay.events) == 0
    assert len(replay.get_available_hands()) == 0


def test_replay_reconstructs_chip_totals(tmp_path: Path) -> None:
    log_file = tmp_path / "test.jsonl"
    events = [
        {
            "type": "HandStarted",
            "hand_number": 0,
            "dealer_seat": 0,
            "small_blind": 5,
            "big_blind": 10,
        },
        {
            "type": "HandEnded",
            "hand_number": 0,
            "chip_distribution": {"0": 10, "1": -10},
        },
    ]
    with open(log_file, "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    replay = Replay(log_file)
    blind_schedule = BlindSchedule(
        levels=[BlindLevel(small=5, big=10)],
        hands_per_level=10,
        fixed=True,
    )
    config = GameConfig(
        num_players=2,
        starting_stack=1000,
        small_blind=5,
        big_blind=10,
        ante=0,
        rake_percent=0.0,
        rake_cap=None,
        blind_schedule=blind_schedule,
    )
    result = replay.replay_hand(0, config)
    assert result["outcome"] == {"0": 10, "1": -10}


def test_replay_summary_skips_timing_and_shows_actions(tmp_path: Path) -> None:
    log_file = tmp_path / "hand.jsonl"
    events = [
        {
            "type": "HandStarted",
            "hand_number": 0,
            "dealer_seat": 0,
            "small_blind": 25,
            "big_blind": 50,
            "stacks": {"0": 1000, "1": 1000, "2": 40},
            "names": {"0": "P1", "1": "P2", "2": "P3"},
        },
        {
            "type": "ActionTaken",
            "hand_number": 0,
            "street": "preflop",
            "seat": 1,
            "action_type": "fold",
            "amount": 0,
            "label": "fold",
            "name": "P2",
            "position": "SB",
        },
        {
            "type": "TimingEvent",
            "hand_number": 0,
            "phase": "bot_action",
            "seat": 1,
            "elapsed_us": 999,
            "seq": 3,
        },
        {
            "type": "ActionTaken",
            "hand_number": 0,
            "street": "preflop",
            "seat": 2,
            "action_type": "call",
            "amount": 80,
            "chips_added": 40,
            "pot_after": 120,
            "label": "call",
            "all_in": True,
            "stack_after": 0,
            "name": "P3",
            "position": "BB",
        },
        {
            "type": "PotsBuilt",
            "hand_number": 0,
            "street": "showdown",
            "pots": [
                {"amount": 100, "eligible_seats": [0, 1, 2]},
                {"amount": 50, "eligible_seats": [0, 1]},
            ],
        },
        {
            "type": "PlayerOut",
            "hand_number": 0,
            "seat": 2,
            "reason": "busted",
            "stack": 0,
            "name": "P3",
            "position": "BB",
        },
        {
            "type": "HandEnded",
            "hand_number": 0,
            "chip_distribution": {"0": 150, "1": 0, "2": 0},
        },
    ]
    with open(log_file, "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    replay = Replay(log_file)
    config = GameConfig(
        num_players=3,
        starting_stack=1000,
        small_blind=25,
        big_blind=50,
        ante=0,
        rake_percent=0.0,
        rake_cap=None,
        blind_schedule=BlindSchedule(
            levels=[BlindLevel(small=25, big=50)],
            hands_per_level=10,
            fixed=True,
        ),
    )
    summary = replay.display_hand_summary(0, config)
    assert "P2 (SB seat 1) folds" in summary
    assert "calls 80" in summary
    assert "all-in" in summary
    assert "main 100" in summary
    assert "side 1 50" in summary
    assert "is eliminated" in summary
    assert "bot_action" not in summary
    assert "elapsed" not in summary

    text = replay.replay_hand(0, config)["text"]
    assert text == summary


def test_replay_multiple_hands(tmp_path: Path) -> None:
    log_file = tmp_path / "test.jsonl"
    events = [
        {"type": "HandStarted", "hand_number": 0, "dealer_seat": 0, "small_blind": 5, "big_blind": 10},
        {"type": "HandEnded", "hand_number": 0, "chip_distribution": {"0": 0, "1": 0}},
        {"type": "HandStarted", "hand_number": 1, "dealer_seat": 1, "small_blind": 5, "big_blind": 10},
        {"type": "HandEnded", "hand_number": 1, "chip_distribution": {"0": 0, "1": 0}},
    ]
    with open(log_file, "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    replay = Replay(log_file)
    hands = replay.get_available_hands()
    assert hands == {0, 1}
    assert len(replay.get_hand_events(0)) > 0
    assert len(replay.get_hand_events(1)) > 0
