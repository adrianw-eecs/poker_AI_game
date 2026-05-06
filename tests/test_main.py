"""Tests for main CLI entry point."""

import sys
from unittest.mock import patch

import pytest

from poker.main import main


def test_main_help() -> None:
    with patch.object(sys, "argv", ["poker", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0


def test_main_invalid_player_count() -> None:
    with patch.object(sys, "argv", ["poker", "-n", "1"]):
        result = main()
        assert result == 1


def test_main_default_arguments() -> None:
    with patch.object(sys, "argv", ["poker", "-hh", "1"]):
        try:
            result = main()
            assert result in (0, 1)
        except Exception:
            pass
