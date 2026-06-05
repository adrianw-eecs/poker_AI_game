"""Action handler with retry logic and invalid action reporting."""

import logging

from poker.bots.base import Bot
from poker.domain.action import Action
from poker.engine.action_validator import legal_actions, validate
from poker.exceptions import IllegalActionError
from poker.state.game_state import GameState, Street


_logger = logging.getLogger(__name__)


class ActionHandler:
    """Handles bot action validation with automatic retries.

    When a bot makes an invalid action, this handler automatically retries
    up to a maximum number of attempts. Invalid attempts are reported via
    standard logging. If a bot exceeds the maximum retry limit, an error is raised.

    Attributes:
        invalid_attempts: Dict tracking attempt count per bot per action.
    """

    def __init__(self) -> None:
        """Initialize action handler."""
        self.invalid_attempts: dict[str, int] = {}  # Track attempts per bot

    def get_valid_action(
        self,
        bot: Bot,
        state: GameState,
        seat: int,
        max_retries: int = 10,
    ) -> Action:
        """Get a valid action from a bot, retrying on invalid actions.

        This method:
        1. Gets legal actions for the seat
        2. Requests an action from the bot
        3. Validates the action
        4. If invalid, logs the error and retries (up to max_retries)
        5. If valid, resets the attempt counter and returns the action
        6. If max_retries exceeded, raises an error with bot name and history

        Args:
            bot: The bot to get action from.
            state: Current game state.
            seat: Seat number.
            max_retries: Maximum retry attempts (default 10).

        Returns:
            A valid Action.

        Raises:
            IllegalActionError: If bot exceeds max_retries invalid attempts.
        """
        bot_name = bot.name
        attempt = 0
        action_history: list[str] = []

        while attempt < max_retries:
            attempt += 1

            try:
                # Get legal actions for this seat
                legal = legal_actions(state, seat)

                # Ask bot for action
                action = bot.act(state.view_for(seat), legal)

                # Validate the action
                validate(state, seat, action)

                # Action is valid - reset attempt counter for this bot
                if bot_name in self.invalid_attempts:
                    del self.invalid_attempts[bot_name]

                return action

            except IllegalActionError as e:
                error_msg = str(e)
                action_str = str(action) if 'action' in locals() else "unknown"

                # Track attempt
                self.invalid_attempts[bot_name] = attempt

                # Report this invalid attempt
                _logger.error(
                    "Invalid bot action attempt=%s bot=%s seat=%s street=%s error=%s action=%s",
                    attempt,
                    bot_name,
                    seat,
                    state.street.value,
                    error_msg,
                    action_str,
                )

                # Record in history for error message
                action_history.append(
                    f"[Attempt {attempt}] {error_msg} | Action: {action_str}"
                )

                # Check if exceeded max retries
                if attempt >= max_retries:
                    # Raise error with bot name and action history
                    history_str = "\n  ".join(action_history)
                    raise IllegalActionError(
                        f"Bot {bot_name} exceeded {max_retries} invalid attempts:\n  {history_str}"
                    )

                # Continue to next attempt

        # Should not reach here, but as a safeguard
        raise IllegalActionError(
            f"Bot {bot_name} exhausted retries without valid action"
        )

    def flush_and_close(self) -> None:
        """No-op (kept for backward compatibility with Session)."""
        return

    def reset(self) -> None:
        """Reset attempt counters (call at start of new hand).

        This resets per-bot invalid attempt counters.
        """
        self.invalid_attempts.clear()
