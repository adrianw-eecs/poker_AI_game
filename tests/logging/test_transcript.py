"""Transcript lines for the facts a hand history has to show."""

from poker.logging.events import ActionTaken, PlayerOut, PotsBuilt
from poker.logging.transcript import render_event


def test_fold_line_names_the_player() -> None:
    line = render_event(
        ActionTaken(
            hand_number=0,
            street="preflop",
            seat=1,
            action_type="fold",
            amount=0,
            label="fold",
            name="P2",
            position="SB",
        )
    )
    assert line is not None
    assert "P2 (SB seat 1) folds" in line


def test_short_all_in_call_line() -> None:
    line = render_event(
        ActionTaken(
            hand_number=0,
            street="preflop",
            seat=2,
            action_type="call",
            amount=80,
            chips_added=40,
            pot_after=120,
            label="call",
            stack_before=40,
            stack_after=0,
            all_in=True,
            name="P3",
            position="BB",
        )
    )
    assert line is not None
    assert "P3 (BB seat 2) calls 80 (+40) (all-in)" in line
    assert "stack 0" in line


def test_side_pot_line() -> None:
    line = render_event(
        PotsBuilt(
            hand_number=0,
            street="showdown",
            pots=((100, (0, 1, 2)), (50, (0, 1))),
        )
    )
    assert line is not None
    assert "main 100 eligible [0, 1, 2]" in line
    assert "side 1 50 eligible [0, 1]" in line


def test_bust_line() -> None:
    line = render_event(
        PlayerOut(
            hand_number=2,
            seat=2,
            reason="busted",
            street="showdown",
            stack=0,
            name="P3",
            position="BB",
        )
    )
    assert line is not None
    assert "P3 (BB seat 2) is eliminated (stack 0)" in line


def test_timing_event_is_omitted() -> None:
    from poker.logging.events import TimingEvent

    assert render_event(TimingEvent(hand_number=0, phase="bot_action", seat=0, elapsed_us=12)) is None
