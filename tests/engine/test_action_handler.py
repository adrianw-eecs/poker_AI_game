"""Tests for ActionHandler retry logic."""

from unittest.mock import MagicMock, patch

import pytest

from poker.domain.action import Action
from poker.engine.action_handler import ActionHandler
from poker.exceptions import IllegalActionError
from poker.state.game_state import GameState, Street


class _MockBot:
    def __init__(self, name: str, actions: list[Action]):
        self.name = name
        self._actions = iter(actions)

    def act(self, state: GameState, legal: list[Action]) -> Action:
        return next(self._actions, Action.fold())

    def observe_result(self, state: GameState, reward: float) -> None:
        pass


def _mock_state() -> MagicMock:
    s = MagicMock(spec=GameState)
    s.street = Street.PREFLOP
    s.action_on_seat = 0
    s.view_for.return_value = s
    return s


def test_valid_action_passes_through() -> None:
    handler = ActionHandler()
    bot = _MockBot("Bot", [Action.fold()])
    with patch("poker.engine.action_handler.legal_actions", return_value=[Action.fold()]):
        with patch("poker.engine.action_handler.validate"):
            result = handler.get_valid_action(bot, _mock_state(), 0)
    assert result == Action.fold()


def test_retry_on_invalid_then_valid() -> None:
    handler = ActionHandler()
    bot = _MockBot("Bot", [Action.raise_to(50), Action.fold()])
    with patch("poker.engine.action_handler.legal_actions", return_value=[Action.fold()]):
        with patch("poker.engine.action_handler.validate",
                   side_effect=[IllegalActionError("bad"), None]):
            result = handler.get_valid_action(bot, _mock_state(), 0)
    assert result == Action.fold()


def test_max_retries_raises() -> None:
    handler = ActionHandler()
    bot = _MockBot("BadBot", [Action.raise_to(i * 10) for i in range(1, 21)])
    with patch("poker.engine.action_handler.legal_actions", return_value=[Action.fold()]):
        with patch("poker.engine.action_handler.validate",
                   side_effect=IllegalActionError("bad")):
            with pytest.raises(IllegalActionError, match="BadBot"):
                handler.get_valid_action(bot, _mock_state(), 0, max_retries=5)
