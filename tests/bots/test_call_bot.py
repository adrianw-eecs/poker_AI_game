"""CallBot must check or call. It must not treat Action objects as integers."""

from poker.bots.call_bot import CallBot
from poker.domain.action import Action, ActionType
from poker.engine.action_validator import legal_actions
from poker.ml.env import PokerEnv


def test_callbot_checks_or_calls_from_action_objects() -> None:
    bot = CallBot()
    check = Action.check()
    call = Action.call(50)
    fold = Action.fold()
    assert bot.act(None, [fold, check, Action.raise_to(100)]).type == ActionType.CHECK
    assert bot.act(None, [fold, call, Action.raise_to(100)]).type == ActionType.CALL


def test_callbot_calls_a_raise_in_the_environment() -> None:
    env = PokerEnv(
        num_players=2,
        learning_seat=0,
        opponent_bots=[CallBot()],
        small_blind=25,
        big_blind=50,
        seed=1,
    )
    env.hand_number = 0
    env.reset()
    assert env.state is not None
    assert env.state.action_on_seat == 0
    legal = legal_actions(env.state, 0)
    assert any(action.type == ActionType.RAISE for action in legal)
    _, _, done, _, _ = env.step(2)
    assert env.state is not None
    assert not done
    assert env.state.street.value == "flop"
    assert env.state.current_bet_to_call == 0
    assert not any(player.has_folded for player in env.state.players)
    assert all(player.committed_this_hand > 50 for player in env.state.players)
