"""Tests for Action and ActionType."""

import pytest

from poker.domain.action import Action, ActionType


def test_action_construction() -> None:
    assert Action.fold().type == ActionType.FOLD
    assert Action.fold().amount == 0
    assert Action.check().type == ActionType.CHECK
    assert Action.call(50).amount == 50
    assert Action.raise_to(100).amount == 100
    assert Action.all_in(500).type == ActionType.ALL_IN


def test_action_validation() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        Action(type=ActionType.CALL, amount=-10)
    with pytest.raises(ValueError, match="fold"):
        Action(type=ActionType.FOLD, amount=50)
    with pytest.raises(ValueError, match="raise"):
        Action(type=ActionType.RAISE, amount=0)
    with pytest.raises(ValueError, match="all_in"):
        Action(type=ActionType.ALL_IN, amount=0)


def test_action_str() -> None:
    assert "folds" in str(Action.fold()).lower()
    assert "checks" in str(Action.check()).lower()
    assert "50" in str(Action.call(50))
    assert "raises" in str(Action.raise_to(100)).lower()
    assert "all-in" in str(Action.all_in(500)).lower()


def test_action_equality() -> None:
    assert Action.call(50) == Action.call(50)
    assert Action.call(50) != Action.call(60)
    assert Action.fold() != Action.check()


def test_action_frozen() -> None:
    action = Action.fold()
    with pytest.raises(AttributeError):
        action.type = ActionType.CHECK  # type: ignore
