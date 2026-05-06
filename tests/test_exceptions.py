"""Tests for exception hierarchy."""

import pytest

from poker.exceptions import (
    ConfigError,
    EngineStateError,
    EvaluationError,
    IllegalActionError,
    PokerError,
)


def test_exception_hierarchy() -> None:
    assert issubclass(PokerError, Exception)
    assert issubclass(ConfigError, PokerError)
    assert issubclass(IllegalActionError, PokerError)
    assert issubclass(EngineStateError, PokerError)
    assert issubclass(EvaluationError, PokerError)


def test_catching_poker_error_catches_subclasses() -> None:
    with pytest.raises(PokerError):
        raise IllegalActionError("Cannot fold")
