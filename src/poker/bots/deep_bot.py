"""Deep Q-learning bot for poker."""

from poker.bots.base_ml_bot import BaseMLBot
from poker.ml.models.deep_q import DeepQModel


class DeepBot(BaseMLBot):
    """Poker bot using deep Q-learning model.

    Uses handcrafted features (15-dim) and a neural network
    to predict Q-values and choose actions.
    """

    def __init__(self, name: str = "DeepBot", model: DeepQModel | None = None) -> None:
        """Initialize the bot.

        Args:
            name: Display name for the bot.
            model: Optional pre-trained DeepQModel. If None, uses unfitted model.
        """
        super().__init__(name=name, model=model or DeepQModel())
