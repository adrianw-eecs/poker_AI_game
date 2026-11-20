"""Pure betting and blind rules shared by the hand engine and PokerEnv.

BettingRound and play_hand call these functions so a training environment
cannot drift from the rules used at the table.
"""

from dataclasses import replace

from poker.domain.action import Action, ActionType
from poker.state.game_state import GameState
from poker.state.player_state import PlayerState
from poker.state.pot_manager import rebuild_pots


def occupied_seats(state: GameState) -> list[int]:
    """Seats still in the game. Stack-zero seats are eliminated before this."""
    return [seat for seat, player in enumerate(state.players) if not player.is_eliminated]


def next_occupied(state: GameState, start: int) -> int:
    """First non-eliminated seat at or clockwise from start."""
    num_players = len(state.players)
    seat = start % num_players
    for _ in range(num_players):
        if not state.players[seat].is_eliminated:
            return seat
        seat = (seat + 1) % num_players
    raise ValueError("No occupied seats")


def blind_seats(state: GameState) -> tuple[int, int]:
    """Small blind and big blind among live seats.

    With three or more live players, both blinds sit left of the button.
    Heads-up, the button posts the small blind and the other player the big blind.
    """
    occupied = occupied_seats(state)
    if len(occupied) < 2:
        raise ValueError("Need two occupied seats to post blinds")
    num_players = len(state.players)
    if len(occupied) == 2:
        sb_seat = (
            state.dealer_seat
            if not state.players[state.dealer_seat].is_eliminated
            else next_occupied(state, (state.dealer_seat + 1) % num_players)
        )
        bb_seat = next_occupied(state, (sb_seat + 1) % num_players)
        return sb_seat, bb_seat
    sb_seat = next_occupied(state, (state.dealer_seat + 1) % num_players)
    bb_seat = next_occupied(state, (sb_seat + 1) % num_players)
    return sb_seat, bb_seat


def first_can_act(state: GameState, start: int) -> int | None:
    """First seat at or clockwise from start that can still bet."""
    num_players = len(state.players)
    seat = start % num_players
    for _ in range(num_players):
        player = state.players[seat]
        if not player.has_folded and not player.is_eliminated and not player.is_all_in:
            return seat
        seat = (seat + 1) % num_players
    return None


def first_preflop_actor(state: GameState, sb_seat: int, bb_seat: int) -> int | None:
    """Heads-up: button acts first. Multiway: first live seat left of the big blind."""
    if len(occupied_seats(state)) == 2:
        return first_can_act(state, sb_seat)
    return first_can_act(state, (bb_seat + 1) % len(state.players))


def apply_blind(player: PlayerState, required: int) -> tuple[PlayerState, int, int, bool]:
    """Post up to ``required`` chips. A short post is an all-in.

    Returns the updated player, chips posted, stack after, and whether the post
    was short.
    """
    actual = min(required, player.stack)
    new_stack = player.stack - actual
    posted = (
        player.with_stack(new_stack)
        .with_committed_this_street(player.committed_this_street + actual)
        .with_committed_this_hand(player.committed_this_hand + actual)
    )
    short = new_stack == 0 or actual < required
    if short:
        posted = posted.with_all_in(True)
    return posted, actual, new_stack, short


def post_antes(state: GameState) -> GameState:
    """Deduct the ante from every non-eliminated player.

    A short stack posts what it has and is all-in. The ante never drives a
    stack negative, matching ``apply_blind``.
    """
    if state.config.ante <= 0:
        return state
    players_list = list(state.players)
    for seat, player in enumerate(state.players):
        if player.is_eliminated:
            continue
        actual = min(state.config.ante, player.stack)
        new_stack = player.stack - actual
        posted = player.with_stack(new_stack).with_committed_this_hand(
            player.committed_this_hand + actual
        )
        if new_stack == 0:
            posted = posted.with_all_in(True)
        players_list[seat] = posted
    return replace(state, players=tuple(players_list))


def may_raise(state: GameState, seat: int) -> bool:
    """Whether ``seat`` may raise above the current bet.

    A short all-in that does not complete a full raise does not re-open
    betting for a player who has already acted; that player may only call or
    fold. A player who has not acted yet may still raise. The first wager of
    a street is not an incomplete raise, even when it is a short all-in.
    """
    player = state.players[seat]
    to_call = state.current_bet_to_call
    if to_call <= 0 or player.committed_this_street >= to_call:
        return True
    acted = any(actor == seat for actor, _action in state.action_history_this_street)
    if not acted:
        return True
    lower = [
        other.committed_this_street
        for other in state.players
        if other.committed_this_street < to_call
    ]
    previous = max(lower) if lower else 0
    # No matched wager sits under this bet, so it is the opening wager.
    if previous <= 0:
        return True
    prior_min = max(state.config.big_blind, state.last_raise_size)
    if reopens_betting(Action.all_in(to_call), previous, prior_min):
        return True
    return not any(
        other.is_all_in and other.committed_this_street == to_call and not other.has_folded
        for other in state.players
    )


def post_blinds(state: GameState, sb_seat: int, bb_seat: int) -> GameState:
    """Post the small blind and big blind and set the opening bet."""
    players_list = list(state.players)
    sb_player = state.players[sb_seat]
    posted_sb, actual_sb, _, _ = apply_blind(sb_player, state.config.small_blind)
    players_list[sb_seat] = posted_sb

    bb_player = players_list[bb_seat]
    posted_bb, actual_bb, _, _ = apply_blind(bb_player, state.config.big_blind)
    players_list[bb_seat] = posted_bb

    ante_pot = sum(
        player.committed_this_hand - player.committed_this_street for player in players_list
    )
    return replace(
        state,
        players=tuple(players_list),
        current_bet_to_call=actual_bb,
        last_raise_size=actual_bb,
        pots_entering_street=(max(0, ante_pot),),
        street_history_ends=(),
    )


def count_active_players(state: GameState) -> int:
    """Count non-folded, non-eliminated players."""
    return sum(
        1 for player in state.players if not player.has_folded and not player.is_eliminated
    )


def reopens_betting(action: Action, prior_max: int, prior_min: int) -> bool:
    """Whether an action re-opens betting under the half-raise rule.

    A raise always re-opens. An all-in re-opens only when the extra chips are
    at least the minimum raise that was in force before the action.
    """
    if action.type == ActionType.RAISE:
        return True
    if action.type == ActionType.ALL_IN:
        return action.amount - prior_max >= prior_min
    return False


def round_is_closed(state: GameState, acted: set[int], max_commit: int) -> bool:
    """True when every player who can still bet has acted and matched.

    All-in players are excluded. A single player who still owes chips must act.
    """
    can_act_seats = [
        i
        for i, player in enumerate(state.players)
        if not player.has_folded and not player.is_eliminated and not player.is_all_in
    ]
    if len(can_act_seats) == 0:
        return True
    if len(can_act_seats) == 1:
        only = state.players[can_act_seats[0]]
        return only.committed_this_street >= max_commit

    for seat in can_act_seats:
        if seat not in acted:
            return False
    for seat in can_act_seats:
        if state.players[seat].committed_this_street < max_commit:
            return False
    return True


def apply_action(state: GameState, seat: int, action: Action) -> GameState:
    """Apply one action, append it to both history lists, and leave the actor in place.

    Action amounts are the player's total street commitment, not the extra chips.
    The caller advances the actor and decides whether the action re-opened betting.
    """
    player = state.players[seat]
    players_list = list(state.players)

    if action.type == ActionType.FOLD:
        players_list[seat] = player.with_folded(True)

    elif action.type == ActionType.CHECK:
        pass

    elif action.type == ActionType.CALL:
        additional_to_commit = action.amount - player.committed_this_street
        new_stack = player.stack - additional_to_commit
        updated_player = (
            player.with_stack(new_stack)
            .with_committed_this_street(action.amount)
            .with_committed_this_hand(player.committed_this_hand + additional_to_commit)
        )
        if new_stack == 0 or action.amount < state.current_bet_to_call:
            updated_player = updated_player.with_all_in(True)
        players_list[seat] = updated_player

    elif action.type == ActionType.RAISE:
        amount_to_raise = action.amount - state.current_bet_to_call
        prior_min = max(state.config.big_blind, state.last_raise_size)
        additional_to_commit = action.amount - player.committed_this_street
        new_stack = player.stack - additional_to_commit
        updated_player = (
            player.with_stack(new_stack)
            .with_committed_this_street(action.amount)
            .with_committed_this_hand(player.committed_this_hand + additional_to_commit)
        )
        if new_stack == 0:
            updated_player = updated_player.with_all_in(True)
        players_list[seat] = updated_player
        new_last_raise = amount_to_raise if amount_to_raise >= prior_min else state.last_raise_size
        state = replace(
            state,
            current_bet_to_call=action.amount,
            last_raise_size=new_last_raise,
            players=tuple(players_list),
        )

    elif action.type == ActionType.ALL_IN:
        additional_to_commit = action.amount - player.committed_this_street
        new_stack = player.stack - additional_to_commit
        updated_player = (
            player.with_stack(new_stack)
            .with_committed_this_street(action.amount)
            .with_committed_this_hand(player.committed_this_hand + additional_to_commit)
            .with_all_in(True)
        )
        players_list[seat] = updated_player
        if action.amount > state.current_bet_to_call:
            amount_to_raise = action.amount - state.current_bet_to_call
            prior_min = max(state.config.big_blind, state.last_raise_size)
            new_last_raise = (
                amount_to_raise if amount_to_raise >= prior_min else state.last_raise_size
            )
            state = replace(
                state,
                current_bet_to_call=action.amount,
                last_raise_size=new_last_raise,
                players=tuple(players_list),
            )
            players_list = list(state.players)

    state = replace(
        state,
        players=tuple(players_list),
        action_history_this_street=[*state.action_history_this_street, (seat, action)],
        action_history_this_hand=[*state.action_history_this_hand, (seat, action)],
    )
    return state


def advance_action(state: GameState, current_seat: int) -> GameState:
    """Move action to the next player who can still bet, or clear it."""
    num_players = len(state.players)
    next_seat = (current_seat + 1) % num_players
    can_act = [
        i
        for i, player in enumerate(state.players)
        if not player.has_folded and not player.is_eliminated and not player.is_all_in
    ]

    if len(can_act) == 0:
        return replace(state, action_on_seat=None)
    if len(can_act) == 1:
        only = can_act[0]
        if state.players[only].committed_this_street >= state.current_bet_to_call:
            return replace(state, action_on_seat=None)
        return replace(state, action_on_seat=only)

    attempts = 0
    while attempts < num_players:
        player = state.players[next_seat]
        if player.has_folded or player.is_eliminated or player.is_all_in:
            next_seat = (next_seat + 1) % num_players
            attempts += 1
            continue
        return replace(state, action_on_seat=next_seat)

    return replace(state, action_on_seat=None)


def focus_action(state: GameState) -> GameState:
    """Point action at the first player from the current seat who can still bet."""
    if state.action_on_seat is None:
        return state
    num_players = len(state.players)
    current_seat = state.action_on_seat
    for _ in range(num_players):
        player = state.players[current_seat]
        if not player.has_folded and not player.is_eliminated and not player.is_all_in:
            return replace(state, action_on_seat=current_seat)
        current_seat = (current_seat + 1) % num_players
    return replace(state, action_on_seat=None)


def reset_street_action(state: GameState) -> GameState:
    """Clear the live bet and give the first live player left of the button the action.

    Records where the previous street ended in the hand history so later
    observations can rebuild per-street summaries.
    """
    pot_now = sum(player.committed_this_hand for player in state.players)
    players_list = [player.with_committed_this_street(0) for player in state.players]
    num_players = len(state.players)
    sb_seat = (state.dealer_seat + 1) % num_players
    action_on_seat = None
    for _ in range(num_players):
        player = state.players[sb_seat]
        if not player.has_folded and not player.is_eliminated and not player.is_all_in:
            action_on_seat = sb_seat
            break
        sb_seat = (sb_seat + 1) % num_players

    state = replace(
        state,
        players=tuple(players_list),
        current_bet_to_call=0,
        last_raise_size=0,
        action_history_this_street=[],
        action_on_seat=action_on_seat,
        street_history_ends=(*state.street_history_ends, len(state.action_history_this_hand)),
        pots_entering_street=(*state.pots_entering_street, pot_now),
    )
    return rebuild_pots(state)
