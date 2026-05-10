"""Population-based self-play management for generational NFSP training."""

from pathlib import Path
from typing import Optional

from poker.bots.nfsp_bot import NFSPBot
from poker.bots.random_bot import RandomBot
from poker.bots.flop_bot import FlopBot
from poker.ml.models.nfsp_model import NFSPModel


class PopulationManager:
    """Manages saved model generations for population-based self-play."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_generation(self, model: NFSPModel, gen_id: int) -> Path:
        """Save a model as a generation checkpoint.

        Args:
            model: NFSP model to save
            gen_id: Generation number (0-indexed)

        Returns:
            Path to saved model
        """
        gen_path = self.models_dir / f"nfsp_gen_{gen_id}.pt"
        model.save(str(gen_path))
        return gen_path

    def load_generation(self, gen_id: int) -> NFSPModel:
        """Load a saved generation model.

        Args:
            gen_id: Generation number to load

        Returns:
            Loaded NFSP model

        Raises:
            FileNotFoundError: If generation doesn't exist
        """
        gen_path = self.models_dir / f"nfsp_gen_{gen_id}.pt"
        if not gen_path.exists():
            raise FileNotFoundError(f"Generation {gen_id} not found at {gen_path}")

        model = NFSPModel()
        model.load(str(gen_path))
        return model

    def get_opponent_roster(self, current_gen: int, num_opponents: int) -> list:
        """Get opponent roster for training current generation.

        Roster composition:
        - 40% RandomBot
        - 30% FlopBot
        - 20% Self-play (previous generation if available)
        - 10% Historical generations (rotating)

        Args:
            current_gen: Current generation ID
            num_opponents: Number of opponents to create

        Returns:
            List of Bot instances
        """
        opponents = []

        # Calculate counts
        random_count = max(1, int(num_opponents * 0.40))
        flop_count = max(1, int(num_opponents * 0.30))
        self_play_count = max(0, int(num_opponents * 0.20))
        pop_count = num_opponents - random_count - flop_count - self_play_count

        # Add RandomBots
        for i in range(random_count):
            opponents.append(RandomBot(name=f"RandomBot_{i}", seed=100 + i))

        # Add FlopBots
        for i in range(flop_count):
            opponents.append(FlopBot(name=f"FlopBot_{i}", seed=200 + i))

        # Add previous generation self-play
        if self_play_count > 0 and current_gen > 0:
            prev_model = self.load_generation(current_gen - 1)
            for i in range(self_play_count):
                bot = NFSPBot(
                    name=f"GenBot_{current_gen-1}_{i}",
                    model=prev_model,
                    training=False  # Eval mode for opponents
                )
                opponents.append(bot)

        # Add rotating historical generations
        for i in range(pop_count):
            if current_gen >= 2:
                # Cycle through older generations
                hist_gen = max(0, current_gen - 2 - (i % max(1, current_gen - 1)))
                hist_model = self.load_generation(hist_gen)
                bot = NFSPBot(
                    name=f"GenBot_{hist_gen}_{i}",
                    model=hist_model,
                    training=False
                )
                opponents.append(bot)
            else:
                # Fallback: use RandomBot if not enough history
                opponents.append(RandomBot(name=f"RandomBot_hist_{i}", seed=300 + i))

        return opponents[:num_opponents]

    def refresh_opponent_roster_for_episode(self, current_gen: int, episode: int, num_opponents: int) -> list:
        """Refresh opponent roster every N episodes (allows dynamic rotation).

        Args:
            current_gen: Current generation ID
            episode: Current episode number
            num_opponents: Number of opponents

        Returns:
            List of Bot instances (same composition, fresh instances)
        """
        # Could add episode-based variation here if desired
        # For now, just return standard roster
        return self.get_opponent_roster(current_gen, num_opponents)
