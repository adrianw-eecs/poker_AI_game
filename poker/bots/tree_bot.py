"""Decision tree bot for poker."""

from poker.bots.base_ml_bot import BaseMLBot
from poker.ml.models.tree_q import TreeQModel


class TreeBot(BaseMLBot):
    """Poker bot using decision tree Q-learning model.

    Uses handcrafted features (15-dim) and decision tree models
    to predict Q-values and choose actions.
    """

    def __init__(self, name: str = "TreeBot", model: TreeQModel | None = None) -> None:
        """Initialize the bot.

        Args:
            name: Display name for the bot.
            model: Optional pre-trained TreeQModel. If None, uses unfitted model.
        """
        super().__init__(name=name, model=model or TreeQModel())
