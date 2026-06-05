"""Full hand orchestration from deal to payout."""

from dataclasses import replace
from typing import Optional

from poker.bots.base import Bot
from poker.domain.action import Action
from poker.domain.deck import Deck
from poker.engine.action_handler import ActionHandler
from poker.engine.action_validator import legal_actions, validate
from poker.engine.betting_round import BettingRound
from poker.engine.dealer import deal_flop, deal_hole_cards, deal_river, deal_turn
from poker.engine.showdown import resolve
from poker.logging.events import (
    AntePosted,
    BlindPosted,
    BoardCardsDealt,
    HandEnded,
    HandStarted,
    HoleCardsDealt,
    StreetEnded,
)
from poker.logging.logger import Logger
from poker.logging.state_snapshot import format_hand_replay, format_state_snapshot
from poker.state.game_state import GameState, Street
from poker.state.pot_manager import rebuild_pots


def play_hand(
    state: GameState,
    bots: dict[int, Bot],
    deck: Deck,
    logger: Logger,
    action_handler: Optional[ActionHandler] = None,
) -> GameState:
    """Play a single hand from start to finish.

    Orchestrates the full hand sequence: antes, blinds, hole cards, betting
    rounds for each street, and showdown.

    Args:
        state: Initial GameState (antes already posted if applicable).
        bots: Dict mapping seat → Bot for action selection.
        deck: Shuffled Deck for this hand.
        logger: Logger to emit hand events.
        action_handler: Optional ActionHandler for action validation with retries.

    Returns:
        Final GameState with updated stacks and all events logged.
    """
    # Emit hand start event
    logger.log_event(
        HandStarted(
            hand_number=state.hand_number,
            dealer_seat=state.dealer_seat,
            small_blind=state.config.small_blind,
            big_blind=state.config.big_blind,
        )
    )

    # Automatically eliminate (fold) players with 0 chips at start of hand
    players_list = list(state.players)
    for seat, player in enumerate(players_list):
        if player.stack == 0 and not player.is_eliminated:
            # Mark as eliminated and folded so they don't participate
            players_list[seat] = player.with_folded(True).with_eliminated(True)
    state = replace(state, players=tuple(players_list))

    # Post antes (if any)
    if state.config.ante > 0:
        state = _post_antes(state, logger)

    # Post blinds
    state = _post_blinds(state, logger)

    # Rebuild pots after blinds are posted
    state = rebuild_pots(state)

    # Deal hole cards (deck is mutated in place)
    players_after_deal = deal_hole_cards(deck, state.players, state.dealer_seat)
    state = replace(state, players=players_after_deal)
    logger.log_event(
        HoleCardsDealt(hand_number=state.hand_number, num_players=len(state.players))
    )

    # Set action to first player (UTG = small blind in heads-up, player after BB in 3+)
    num_players = len(state.players)
    if num_players == 2:
        # Heads-up: button is small blind, acts first preflop
        action_on_seat = state.dealer_seat
    else:
        # Multi-way: UTG (player after big blind) acts first
        action_on_seat = (state.dealer_seat + 3) % num_players

    state = replace(state, action_on_seat=action_on_seat)

    hand_log: list[tuple[str, int, Action]] = []

    # Pre-flop betting
    state = _run_betting_round(state, bots, logger, action_handler)
    _append_street_actions(state, hand_log)
    _log_street_snapshot(state, logger)

    # If ≤1 player remains, go straight to showdown
    if _count_active_players(state) <= 1:
        return _finish_hand(state, bots, deck, logger, hand_log)

    # Flop (deck is mutated in place)
    board_cards = deal_flop(deck)
    state = replace(state, street=Street.FLOP, community_cards=board_cards)
    logger.log_event(
        BoardCardsDealt(
            hand_number=state.hand_number,
            street=Street.FLOP.value,
            cards=tuple(str(c) for c in board_cards),
        )
    )
    state = _reset_street_action(state)
    state = _run_betting_round(state, bots, logger, action_handler)
    _append_street_actions(state, hand_log)
    _log_street_snapshot(state, logger)

    if _count_active_players(state) <= 1:
        return _finish_hand(state, bots, deck, logger, hand_log)

    # Turn (deck is mutated in place)
    turn_card = deal_turn(deck)
    new_board = (*state.community_cards, turn_card)
    state = replace(state, street=Street.TURN, community_cards=new_board)
    logger.log_event(
        BoardCardsDealt(
            hand_number=state.hand_number,
            street=Street.TURN.value,
            cards=(str(turn_card),),
        )
    )
    state = _reset_street_action(state)
    state = _run_betting_round(state, bots, logger, action_handler)
    _append_street_actions(state, hand_log)
    _log_street_snapshot(state, logger)

    if _count_active_players(state) <= 1:
        return _finish_hand(state, bots, deck, logger, hand_log)

    # River (deck is mutated in place)
    river_card = deal_river(deck)
    new_board = (*state.community_cards, river_card)
    state = replace(state, street=Street.RIVER, community_cards=new_board)
    logger.log_event(
        BoardCardsDealt(
            hand_number=state.hand_number,
            street=Street.RIVER.value,
            cards=(str(river_card),),
        )
    )
    state = _reset_street_action(state)
    state = _run_betting_round(state, bots, logger, action_handler)
    _append_street_actions(state, hand_log)
    _log_street_snapshot(state, logger)

    # Showdown
    return _finish_hand(state, bots, deck, logger, hand_log)


def _post_antes(state: GameState, logger: Logger) -> GameState:
    """Post antes for all active players.

    Args:
        state: Current game state.
        logger: Logger to emit AntePosted events.

    Returns:
        Updated game state with antes posted.
    """
    players_list = list(state.players)

    for seat, player in enumerate(state.players):
        if not player.is_eliminated:
            ante_amount = state.config.ante
            new_stack = player.stack - ante_amount
            new_committed_hand = player.committed_this_hand + ante_amount

            players_list[seat] = (
                player.with_stack(new_stack).with_committed_this_hand(new_committed_hand)
            )

            logger.log_event(
                AntePosted(hand_number=state.hand_number, seat=seat, amount=ante_amount)
            )

    return replace(state, players=tuple(players_list))


def _post_blinds(state: GameState, logger: Logger) -> GameState:
    """Post small blind and big blind.

    Args:
        state: Current game state.
        logger: Logger to emit BlindPosted events.

    Returns:
        Updated game state with blinds posted and action ready to start.
    """
    players_list = list(state.players)
    num_players = len(state.players)

    # Small blind is one seat left of button
    sb_seat = (state.dealer_seat + 1) % num_players
    # Big blind is two seats left of button
    bb_seat = (state.dealer_seat + 2) % num_players

    # Post small blind
    sb_player = state.players[sb_seat]
    sb_amount = state.config.small_blind
    actual_sb = min(sb_amount, sb_player.stack)  # Cap at available stack
    new_stack_sb = sb_player.stack - actual_sb
    players_list[sb_seat] = (
        sb_player.with_stack(new_stack_sb)
        .with_committed_this_street(actual_sb)
        .with_committed_this_hand(actual_sb)
    )
    logger.log_event(
        BlindPosted(
            hand_number=state.hand_number,
            seat=sb_seat,
            amount=actual_sb,
            is_big_blind=False,
        )
    )

    # Post big blind
    bb_player = state.players[bb_seat]
    bb_amount = state.config.big_blind
    actual_bb = min(bb_amount, bb_player.stack)  # Cap at available stack
    new_stack_bb = bb_player.stack - actual_bb
    players_list[bb_seat] = (
        bb_player.with_stack(new_stack_bb)
        .with_committed_this_street(actual_bb)
        .with_committed_this_hand(actual_bb)
    )
    logger.log_event(
        BlindPosted(
            hand_number=state.hand_number,
            seat=bb_seat,
            amount=actual_bb,
            is_big_blind=True,
        )
    )

    # Update game state with blinds posted
    state = replace(
        state,
        players=tuple(players_list),
        current_bet_to_call=actual_bb,  # Use actual amount posted, not full BB
        last_raise_size=actual_bb,
    )

    return state


def _run_betting_round(
    state: GameState,
    bots: dict[int, Bot],
    logger: Logger,
    action_handler: Optional[ActionHandler] = None,
) -> GameState:
    """Run a single betting round for the current street.

    Args:
        state: Current game state.
        bots: Dict of bots for action.
        logger: Logger for events.
        action_handler: Optional ActionHandler for action validation with retries.

    Returns:
        Updated game state after the betting round.
    """

    def get_bot_action(seat: int, s: GameState):  # type: ignore
        """Get an action from a bot."""
        from poker.domain.action import Action as ActionClass
        bot = bots[seat]

        if action_handler:
            # Use action handler with retries and logging
            action: ActionClass = action_handler.get_valid_action(bot, s, seat)
        else:
            # Fallback to direct validation (for backward compatibility)
            legal = legal_actions(s, seat)
            action: ActionClass = bot.act(s.view_for(seat), legal)
            validate(s, seat, action)

        return action

    betting_round = BettingRound(state, get_bot_action, logger)
    state = betting_round.run()

    # Rebuild pots after betting round to track pot in UI
    state = rebuild_pots(state)

    return state


def _reset_street_action(state: GameState) -> GameState:
    """Reset street-specific state for a new street.

    Clears committed_this_street and action history, sets action to first player
    who has not folded and is not eliminated.

    Args:
        state: Current game state.

    Returns:
        Updated game state with street reset.
    """
    # Reset committed_this_street and action history
    players_list = [
        p.with_committed_this_street(0) for p in state.players
    ]

    # Set action to small blind or first active player
    num_players = len(state.players)
    sb_seat = (state.dealer_seat + 1) % num_players

    # Find first player who can act (not folded, not eliminated, not all-in)
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
    )

    # Rebuild pots when moving to new street
    state = rebuild_pots(state)

    return state


def _append_street_actions(
    state: GameState,
    hand_log: list[tuple[str, int, Action]],
) -> None:
    """Append this street's actions to the full-hand replay log."""
    street = state.street.value.upper()
    for seat, action in state.action_history_this_street:
        hand_log.append((street, seat, action))


def _finish_hand(
    state: GameState,
    bots: dict[int, Bot],
    deck: Deck,
    logger: Logger,
    hand_log: list[tuple[str, int, Action]],
) -> GameState:
    """Resolve the hand via showdown and update stacks.

    Args:
        state: Current game state.
        bots: Dict of bots (for observe_result callbacks).
        deck: The remaining deck (needed for run-it-twice).
        logger: Logger for events.

    Returns:
        Final game state with stacks updated.
    """
    # Move to showdown street for display purposes
    state = replace(
        state,
        street=Street.SHOWDOWN,
        action_on_seat=None,  # Clear action - no more decisions in showdown
        action_history_this_street=[],  # Clear this street's history
    )
    _log_street_snapshot(state, logger)

    # Resolve showdown and get awards + rake
    awards, rake_taken = resolve(state, deck)

    # Display showdown information (BEFORE updating stacks so we can see committed amounts)
    from poker.interface.text_ui import display_showdown
    display_showdown(state, awards)

    # Print rake amount
    if rake_taken > 0:
        print(f"  Rake: {rake_taken} chips")

    # Update stacks and clear committed amounts (chips are now in stacks)
    # Handle negative stacks: clamp to 0 and mark as eliminated
    players_list = list(state.players)
    for seat in range(len(state.players)):
        player = state.players[seat]
        award = awards.get(seat, 0)
        new_stack = player.stack + award

        # Prevent negative stacks - clamp to 0 and mark as eliminated
        is_eliminated = new_stack <= 0
        final_stack = max(0, new_stack)

        players_list[seat] = player.with_stack(final_stack).with_committed_this_hand(0).with_eliminated(is_eliminated)

    state = replace(state, players=tuple(players_list), pots=[])

    # Log full hand replay before HandEnded
    replay = format_hand_replay(hand_log, state)
    if replay:
        logger.log_event(
            StreetEnded(
                hand_number=state.hand_number,
                street="REPLAY",
                snapshot=replay,
            )
        )

    # Emit HandEnded event
    logger.log_event(
        HandEnded(
            hand_number=state.hand_number,
            chip_distribution={seat: award for seat, award in awards.items()},
        )
    )

    # Notify bots of their result
    for seat, bot in bots.items():
        if not state.players[seat].is_eliminated:
            reward = awards.get(seat, 0) / state.config.starting_stack
            bot.observe_result(state.view_for(seat), reward)

    # Flush logger
    logger.flush()

    return state


def _log_street_snapshot(state: GameState, logger: Logger) -> None:
    """Emit a StreetEnded event with a full-state snapshot."""
    logger.log_event(
        StreetEnded(
            hand_number=state.hand_number,
            street=state.street.value.upper(),
            snapshot=format_state_snapshot(state),
        )
    )


def _count_active_players(state: GameState) -> int:
    """Count non-folded, non-eliminated players.

    Args:
        state: Current game state.

    Returns:
        Number of active players.
    """
    return sum(
        1 for p in state.players if not p.has_folded and not p.is_eliminated
    )
