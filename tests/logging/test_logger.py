"""Tests for game logging."""

import json
import tempfile
from pathlib import Path

from poker.logging.events import HandEnded, HandStarted
from poker.logging.logger import GameLogger, NullLogger


def test_null_logger_discards() -> None:
    logger = NullLogger()
    logger.log_event(HandStarted(1, 0, 5, 10))
    logger.flush()


def test_game_logger_writes_jsonl() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test.jsonl"
        logger = GameLogger(filepath)
        logger.log_event(HandStarted(1, 0, 5, 10))
        logger.log_event(HandEnded(1, {0: 100, 1: -100}))
        logger.flush()
        assert filepath.exists()
        with open(filepath) as f:
            lines = f.readlines()
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert "type" in data


def test_game_logger_creates_directories() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "logs" / "subdir" / "test.jsonl"
        logger = GameLogger(filepath)
        assert filepath.parent.exists()
        logger.log_event(HandStarted(1, 0, 5, 10))
        logger.flush()
        assert filepath.exists()
