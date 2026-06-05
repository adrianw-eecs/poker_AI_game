"""Linear regression bot for poker."""

from poker.bots.base_ml_bot import BaseMLBot
from poker.ml.models.linear_q import LinearQModel


class LinearBot(BaseMLBot):
    """Poker bot using linear Q-learning model.

    Uses handcrafted features (15-dim) and a linear regression model
    to predict Q-values and choose actions.
    """

    def __init__(self, name: str = "LinearBot", model: LinearQModel | None = None) -> None:
        """Initialize the bot.

        Args:
            name: Display name for the bot.
            model: Optional pre-trained LinearQModel. If None, uses unfitted model.
        """
        super().__init__(name=name, model=model or LinearQModel())
