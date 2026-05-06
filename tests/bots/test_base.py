"""Tests for bot protocol."""

from poker.bots.base import Bot
from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.domain.action import Action
from poker.state.game_state import GameState, Street
from poker.state.player_state import PlayerState
from poker.state.pot import Pot


class NoOpBot:
    """A simple bot that satisfies the Bot protocol."""

    @property
    def name(self) -> str:
        """Return bot name."""
        return "NoOpBot"

    def act(self, state: GameState, legal: list[Action]) -> Action:
        """Always return the first legal action."""
        return legal[0] if legal else Action.fold()

    def observe_result(self, final_state: GameState, reward: float) -> None:
        """Do nothing with the result."""
        pass


def test_bot_protocol_satisfied() -> None:
    """Verify a no-op bot satisfies the Bot protocol."""
    bot: Bot = NoOpBot()
    assert bot.name == "NoOpBot"

    # Create a minimal game state
    config = GameConfig(
        num_players=2,
        starting_stack=1000,
        small_blind=5,
        big_blind=10,
        ante=0,
        rake_percent=0.0,
        rake_cap=None,
        blind_schedule=BlindSchedule(
            levels=[BlindLevel(small=5, big=10)],
            hands_per_level=10,
            fixed=True,
        ),
    )

    players = (
        PlayerState(
            seat=0,
            name="Bot",
            stack=1000,
            hole_cards=(),
            committed_this_street=0,
            committed_this_hand=0,
            has_folded=False,
            is_all_in=False,
            is_eliminated=False,
        ),
        PlayerState(
            seat=1,
            name="Other",
            stack=1000,
            hole_cards=(),
            committed_this_street=0,
            committed_this_hand=0,
            has_folded=False,
            is_all_in=False,
            is_eliminated=False,
        ),
    )

    blind_level = config.blind_schedule.level_for_hand(0)
    state = GameState(
        hand_number=0,
        street=Street.PREFLOP,
        dealer_seat=0,
        players=players,
        community_cards=(),
        pots=[Pot(amount=15, eligible_seats=frozenset({0, 1}))],
        current_bet_to_call=10,
        last_raise_size=10,
        action_history_this_street=[],
        action_history_this_hand=[],
        deck_remaining_count=50,
        config=config,
        blind_level=blind_level,
        action_on_seat=1,
    )

    # Test act method
    legal_actions = [Action.check(), Action.call(10), Action.raise_to(20)]
    action = bot.act(state, legal_actions)
    assert action in legal_actions

    # Test observe_result method
    bot.observe_result(state, 0.1)  # Should not raise


def test_bot_protocol_multiple_implementations() -> None:
    """Verify multiple implementations can satisfy the protocol."""

    class AlwaysFold:
        @property
        def name(self) -> str:
            return "AlwaysFold"

        def act(self, state: GameState, legal: list[Action]) -> Action:
            return Action.fold()

        def observe_result(self, final_state: GameState, reward: float) -> None:
            pass

    class AlwaysCall:
        @property
        def name(self) -> str:
            return "AlwaysCall"

        def act(self, state: GameState, legal: list[Action]) -> Action:
            for action in legal:
                if action.type.value == "call":
                    return action
            return legal[0]

        def observe_result(self, final_state: GameState, reward: float) -> None:
            pass

    fold_bot: Bot = AlwaysFold()
    call_bot: Bot = AlwaysCall()

    assert fold_bot.name == "AlwaysFold"
    assert call_bot.name == "AlwaysCall"
