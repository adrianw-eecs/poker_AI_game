"""CallBot - Always calls, never folds or raises (simple calling station).

This bot is deliberately simple and predictable - it forces the learning agent
to exploit weak play patterns (always calling). Combined with RandomBot and
FlopBot, it provides diverse opponent types for training.
"""

import numpy as np


class CallBot:
    """Calling station bot - calls every bet, never raises or folds voluntarily.

    Strategy:
    - FOLD: only if forced (e.g., facing all-in with worst hand)
    - CALL: always (any bet, any amount)
    - RAISE: never
    - CHECK: if no bet (effectively calls with 0 cost)

    This creates a weak but predictable opponent that forces the learning agent
    to learn value betting (exploit weak calling).
    """

    def __init__(self, seed: int = None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def act(self, state, legal_actions: list) -> int:
        """Return action: prefer CALL (action 2), fall back to CHECK (action 1).

        Args:
            state: Game state (unused, for interface compatibility)
            legal_actions: List of legal action indices

        Returns:
            Action index (CALL=2, CHECK=1, FOLD=0)
        """
        # Try to call (action 2)
        if 2 in legal_actions:  # CALL
            return 2

        # If can't call, try to check (action 1)
        if 1 in legal_actions:  # CHECK
            return 1

        # If forced to fold (worst case)
        if 0 in legal_actions:  # FOLD
            return 0

        # Fallback (shouldn't happen)
        return legal_actions[0] if legal_actions else 0
