"""Regret matching utilities for SD-CFR."""

import numpy as np
import numpy.typing as npt
import torch

from poker.ml.models.nfsp_networks import PokerNetwork


def regret_matching(
    advantages: npt.NDArray[np.float32],
    legal_mask: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Convert advantage values to a strategy via regret matching.

    Args:
        advantages: Raw network output of shape (7,).
        legal_mask: Binary mask of shape (7,) where 1 indicates a legal action.

    Returns:
        Probability distribution of shape (7,) summing to 1.0 over legal actions.
    """
    positive_regrets = np.maximum(advantages, 0.0) * legal_mask
    total = positive_regrets.sum()
    if total > 0.0:
        return positive_regrets / total
    # Uniform over legal actions
    legal_count = legal_mask.sum()
    return legal_mask / legal_count


def compute_strategy_from_network(
    network: PokerNetwork,
    obs_tensor: torch.Tensor,
    legal_mask: npt.NDArray[np.float32],
    device: torch.device,
) -> npt.NDArray[np.float32]:
    """Forward the network and apply regret matching to produce a strategy.

    Args:
        network: PokerNetwork to query.
        obs_tensor: Observation tensor of shape (1, 142) or (142,).
        legal_mask: Binary mask of shape (7,).
        device: Torch device for inference.

    Returns:
        Probability distribution of shape (7,).
    """
    network.eval()
    with torch.no_grad():
        x = obs_tensor.to(device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        advantages = network(x).squeeze(0).cpu().numpy().astype(np.float32)
    return regret_matching(advantages, legal_mask)
