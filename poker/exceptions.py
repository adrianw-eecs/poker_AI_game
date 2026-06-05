"""Custom exception hierarchy for poker."""


class PokerError(Exception):
    """Base exception for all poker-related errors."""

    pass


class ConfigError(PokerError):
    """Raised when game configuration is invalid."""

    pass


class IllegalActionError(PokerError):
    """Raised when a player attempts an illegal action."""

    pass


class EngineStateError(PokerError):
    """Raised when the game engine reaches an invalid state."""

    pass


class EvaluationError(PokerError):
    """Raised when hand evaluation fails."""

    pass
