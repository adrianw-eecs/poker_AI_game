"""Full hand orchestration from deal to payout."""

from dataclasses import replace
from typing import Optional

from poker.bots.base import Bot
from poker.domain.deck import Deck
from poker.engine.action_handler import ActionHandler
from poker.engine.action_validator import legal_actions, validate
from poker.engine.betting_round import BettingRound
from poker.engine.dealer import deal_flop, deal_hole_cards, deal_river, deal_turn
from poker.engine.rules import (
    blind_seats as _blind_seats,
    first_preflop_actor as _first_preflop_actor,
    occupied_seats as _occupied_seats,
    post_antes as _post_antes_state,
    post_blinds as _post_blinds_state,
    reset_street_action as _reset_street_action,
)
from poker.engine.showdown import HandResolution, resolve_hand
from poker.interface.text_ui import display_showdown, format_showdown_labels
from poker.logging.events import (
    AntePosted,
    BlindPosted,
    BoardCardsDealt,
    HandEnded,
    HandStarted,
    HoleCardsDealt,
    PlayerOut,
    PotAwarded,
    PotsBuilt,
    RakeTaken,
    RunItTwice,
    Showdown,
    StreetEnded,
)
from poker.logging.logger import Logger
from poker.logging.state_snapshot import position_label
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
    names = _actor_names(state, bots)
    num_players = len(state.players)
    players_list = list(state.players)
    for seat, player in enumerate(players_list):
        if player.stack == 0 and not player.is_eliminated:
            players_list[seat] = player.with_folded(True).with_eliminated(True)
            logger.log_event(
                PlayerOut(
                    hand_number=state.hand_number,
                    seat=seat,
                    reason="sit_out",
                    street=state.street.value,
                    stack=0,
                    name=names.get(seat, player.name),
                    position=position_label(seat, state.dealer_seat, num_players),
                )
            )
    state = replace(state, players=tuple(players_list))

    sb_seat, bb_seat = _blind_seats(state)
    active_seats = tuple(_occupied_seats(state))
    logger.log_event(
        HandStarted(
            hand_number=state.hand_number,
            dealer_seat=state.dealer_seat,
            small_blind=state.config.small_blind,
            big_blind=state.config.big_blind,
            ante=state.config.ante,
            sb_seat=sb_seat,
            bb_seat=bb_seat,
            stacks={player.seat: player.stack for player in state.players},
            names=names,
            active_seats=active_seats,
        )
    )

    if state.config.ante > 0:
        state = _post_antes(state, logger, names)

    state = _post_blinds(state, logger, names, sb_seat, bb_seat)
    state = rebuild_pots(state)

    players_after_deal = deal_hole_cards(deck, state.players, state.dealer_seat)
    state = replace(state, players=players_after_deal)
    dealt = {
        player.seat: tuple(card.code() for card in player.hole_cards)
        for player in state.players
        if player.hole_cards
    }
    logger.log_event(
        HoleCardsDealt(
            hand_number=state.hand_number,
            num_players=len(state.players),
            cards=dealt,
            names=names,
        )
    )

    state = replace(state, action_on_seat=_first_preflop_actor(state, sb_seat, bb_seat))

    state = _run_betting_round(state, bots, logger, action_handler, names)
    _log_street_ended(state, logger)

    if _count_active_players(state) <= 1:
        return _finish_hand(state, bots, deck, logger, names)

    board_cards = deal_flop(deck)
    state = replace(state, street=Street.FLOP, community_cards=board_cards)
    logger.log_event(
        BoardCardsDealt(
            hand_number=state.hand_number,
            street=Street.FLOP.value,
            cards=tuple(card.code() for card in board_cards),
            board=tuple(card.code() for card in board_cards),
        )
    )
    state = _reset_street_action(state)
    state = _run_betting_round(state, bots, logger, action_handler, names)
    _log_street_ended(state, logger)

    if _count_active_players(state) <= 1:
        return _finish_hand(state, bots, deck, logger, names)

    turn_card = deal_turn(deck)
    new_board = (*state.community_cards, turn_card)
    state = replace(state, street=Street.TURN, community_cards=new_board)
    logger.log_event(
        BoardCardsDealt(
            hand_number=state.hand_number,
            street=Street.TURN.value,
            cards=(turn_card.code(),),
            board=tuple(card.code() for card in new_board),
        )
    )
    state = _reset_street_action(state)
    state = _run_betting_round(state, bots, logger, action_handler, names)
    _log_street_ended(state, logger)

    if _count_active_players(state) <= 1:
        return _finish_hand(state, bots, deck, logger, names)

    river_card = deal_river(deck)
    new_board = (*state.community_cards, river_card)
    state = replace(state, street=Street.RIVER, community_cards=new_board)
    logger.log_event(
        BoardCardsDealt(
            hand_number=state.hand_number,
            street=Street.RIVER.value,
            cards=(river_card.code(),),
            board=tuple(card.code() for card in new_board),
        )
    )
    state = _reset_street_action(state)
    state = _run_betting_round(state, bots, logger, action_handler, names)
    _log_street_ended(state, logger)

    return _finish_hand(state, bots, deck, logger, names)


def _actor_names(state: GameState, bots: dict[int, Bot]) -> dict[int, str]:
    """Prefer the bot name so the log matches who is seated."""
    names: dict[int, str] = {}
    for seat, player in enumerate(state.players):
        bot = bots.get(seat)
        bot_name = getattr(bot, "name", "") if bot is not None else ""
        names[seat] = bot_name or player.name
    return names


def _post_antes(state: GameState, logger: Logger, names: dict[int, str]) -> GameState:
    """Post antes for all active players."""
    before = state.players
    num_players = len(state.players)
    state = _post_antes_state(state)
    for seat, player in enumerate(before):
        if player.is_eliminated:
            continue
        new_stack = state.players[seat].stack
        logger.log_event(
            AntePosted(
                hand_number=state.hand_number,
                seat=seat,
                amount=player.stack - new_stack,
                stack_before=player.stack,
                stack_after=new_stack,
                all_in=state.players[seat].is_all_in,
                name=names.get(seat, player.name),
                position=position_label(seat, state.dealer_seat, num_players),
            )
        )
    return state


def _post_blinds(
    state: GameState,
    logger: Logger,
    names: dict[int, str],
    sb_seat: int,
    bb_seat: int,
) -> GameState:
    """Post small blind and big blind from live seats."""
    before = state.players
    num_players = len(state.players)
    state = _post_blinds_state(state, sb_seat, bb_seat)
    for seat, is_big in ((sb_seat, False), (bb_seat, True)):
        required = state.config.big_blind if is_big else state.config.small_blind
        posted = before[seat].stack - state.players[seat].stack
        logger.log_event(
            BlindPosted(
                hand_number=state.hand_number,
                seat=seat,
                amount=posted,
                is_big_blind=is_big,
                required=required,
                stack_before=before[seat].stack,
                stack_after=state.players[seat].stack,
                all_in=state.players[seat].is_all_in,
                name=names.get(seat, before[seat].name),
                position=position_label(seat, state.dealer_seat, num_players),
            )
        )
    return state


def _run_betting_round(
    state: GameState,
    bots: dict[int, Bot],
    logger: Logger,
    action_handler: Optional[ActionHandler],
    names: dict[int, str],
) -> GameState:
    """Run a single betting round for the current street."""

    def get_bot_action(seat: int, s: GameState):  # type: ignore
        """Get an action from a bot."""
        from poker.domain.action import Action as ActionClass
        bot = bots[seat]

        if action_handler:
            action: ActionClass = action_handler.get_valid_action(bot, s, seat)
        else:
            legal = legal_actions(s, seat)
            action = bot.act(s.view_for(seat), legal)
            validate(s, seat, action)

        return action

    betting_round = BettingRound(state, get_bot_action, logger, names)
    state = betting_round.run()
    state = rebuild_pots(state)
    return state


def _finish_hand(
    state: GameState,
    bots: dict[int, Bot],
    deck: Deck,
    logger: Logger,
    names: dict[int, str],
) -> GameState:
    """Resolve the hand via showdown and update stacks."""
    state = replace(
        state,
        street=Street.SHOWDOWN,
        action_on_seat=None,
        action_history_this_street=[],
    )

    resolution = resolve_hand(state, deck)
    awards = resolution.awards
    rake_taken = resolution.rake
    committed = {
        seat: state.players[seat].committed_this_hand
        for seat in range(len(state.players))
    }

    display_showdown(state, awards)

    if rake_taken > 0:
        print(f"  Rake: {rake_taken} chips")

    _log_resolution(state, logger, names, resolution)

    players_list = list(state.players)
    for seat in range(len(state.players)):
        player = state.players[seat]
        award = awards.get(seat, 0)
        new_stack = player.stack + award
        is_eliminated = new_stack <= 0
        final_stack = max(0, new_stack)
        players_list[seat] = (
            player.with_stack(final_stack)
            .with_committed_this_hand(0)
            .with_eliminated(is_eliminated)
        )

    state = replace(state, players=tuple(players_list), pots=[])

    nets = {
        seat: awards.get(seat, 0) - committed[seat]
        for seat in range(len(state.players))
    }
    logger.log_event(
        HandEnded(
            hand_number=state.hand_number,
            chip_distribution={seat: award for seat, award in awards.items()},
            net=nets,
            stacks_after={player.seat: player.stack for player in state.players},
            names=names,
        )
    )

    for seat, bot in bots.items():
        if not state.players[seat].is_eliminated:
            reward = awards.get(seat, 0) / state.config.starting_stack
            bot.observe_result(state.view_for(seat), reward)

    logger.flush()
    return state


def _log_resolution(
    state: GameState,
    logger: Logger,
    names: dict[int, str],
    resolution: HandResolution,
) -> None:
    """Emit pot, showdown, and award events for a finished hand."""
    pots = tuple(
        (pot.amount, tuple(sorted(pot.eligible_seats)))
        for pot in resolution.pots
    )
    logger.log_event(
        PotsBuilt(hand_number=state.hand_number, street="showdown", pots=pots)
    )

    contenders = [
        seat
        for seat, player in enumerate(state.players)
        if not player.has_folded and not player.is_eliminated
    ]
    hole_cards = {
        seat: tuple(card.code() for card in state.players[seat].hole_cards)
        for seat in contenders
        if state.players[seat].hole_cards
    }
    hands = {
        seat: (int(rank.type), rank.kickers)
        for seat, rank in resolution.ranks.items()
    }
    labels = (
        format_showdown_labels(resolution.ranks, resolution.best_five)
        if resolution.ranks
        else {}
    )
    best_five = {
        seat: tuple(card.code() for card in cards)
        for seat, cards in resolution.best_five.items()
    }
    logger.log_event(
        Showdown(
            hand_number=state.hand_number,
            hands=hands,
            ended_early=resolution.ended_early,
            hole_cards=hole_cards,
            rank_names=labels,
            best_five=best_five,
            names=names,
        )
    )

    if (
        resolution.board1 is not None
        and resolution.board2 is not None
        and resolution.awards1 is not None
        and resolution.awards2 is not None
    ):
        ranks1 = (
            format_showdown_labels(resolution.ranks1, resolution.best1 or {})
            if resolution.ranks1
            else {}
        )
        ranks2 = (
            format_showdown_labels(resolution.ranks2, resolution.best2 or {})
            if resolution.ranks2
            else {}
        )
        logger.log_event(
            RunItTwice(
                hand_number=state.hand_number,
                board1=tuple(card.code() for card in resolution.board1),
                board2=tuple(card.code() for card in resolution.board2),
                awards1=resolution.awards1,
                awards2=resolution.awards2,
                ranks1=ranks1,
                ranks2=ranks2,
            )
        )

    pot_index_by_run: dict[int, int] = {}
    for run, amount, winners, split in resolution.pot_awards:
        pot_index = pot_index_by_run.get(run, 0)
        pot_index_by_run[run] = pot_index + 1
        logger.log_event(
            PotAwarded(
                hand_number=state.hand_number,
                pot_index=pot_index,
                amount=amount,
                winning_seats=winners,
                split=split,
                run=run,
            )
        )

    if resolution.rake > 0:
        logger.log_event(RakeTaken(hand_number=state.hand_number, amount=resolution.rake))


def _log_street_ended(state: GameState, logger: Logger) -> None:
    """Emit a structured street-close event."""
    active = tuple(
        seat
        for seat, player in enumerate(state.players)
        if not player.has_folded and not player.is_eliminated
    )
    folded = tuple(
        seat
        for seat, player in enumerate(state.players)
        if player.has_folded and not player.is_eliminated
    )
    all_in = tuple(
        seat
        for seat, player in enumerate(state.players)
        if player.is_all_in and not player.has_folded and not player.is_eliminated
    )
    logger.log_event(
        StreetEnded(
            hand_number=state.hand_number,
            street=state.street.value.upper(),
            pot=sum(pot.amount for pot in state.pots),
            board=tuple(card.code() for card in state.community_cards),
            active_seats=active,
            folded_seats=folded,
            all_in_seats=all_in,
            reason=_street_reason(state),
        )
    )


def _street_reason(state: GameState) -> str:
    """Why the betting street stopped."""
    active = [
        player
        for player in state.players
        if not player.has_folded and not player.is_eliminated
    ]
    if len(active) <= 1:
        return "uncontested"
    can_act = [player for player in active if not player.is_all_in]
    if len(can_act) <= 1:
        return "all_in"
    return "matched"


def _count_active_players(state: GameState) -> int:
    """Count non-folded, non-eliminated players."""
    return sum(
        1 for player in state.players if not player.has_folded and not player.is_eliminated
    )
