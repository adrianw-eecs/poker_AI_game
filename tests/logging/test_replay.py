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
