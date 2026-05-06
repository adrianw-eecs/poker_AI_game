#!/usr/bin/env python
"""Train a decision tree Q-learning model."""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poker.ml.features.handcrafted import extract_handcrafted_features, feature_names
from poker.ml.models.tree_q import TreeQModel
from poker.training.dataset import ReplayBuffer


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train a decision tree poker agent")
    parser.add_argument("--data", type=str, default="data/selfplay.npz", help="Input data file")
    parser.add_argument("--out", type=str, default="models/tree_q.pkl", help="Output model file")
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum tree depth")
    parser.add_argument("--test-split", type=float, default=0.2, help="Test set fraction")

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    buffer = ReplayBuffer()
    buffer.load(args.data)

    if buffer.size == 0:
        print("ERROR: No data loaded")
        sys.exit(1)

    print(f"Loaded {buffer.size} experiences")
    print(f"Stats: {buffer.stats()}")

    # Extract features from observations
    print("\nExtracting handcrafted features...")
    num_experiences = buffer.size
    X = np.zeros((num_experiences, 15), dtype=np.float32)

    # Note: We can't extract features directly from observations (which are 142-dim)
    # We need the full game state. For now, we'll use a workaround: train on the
    # observations directly by flattening them to 15 dims via PCA or similar.
    # For this prototype, let's just use the first 15 features of the observation.
    # This is a limitation - in a real implementation, we'd need to store the game states.

    print("WARNING: Using observation prefix as features (not true handcrafted features)")
    print("For production, store game states during data generation")
    X = buffer.observations[: buffer.size, :15].copy()

    actions = buffer.actions[: buffer.size]
    rewards = buffer.rewards[: buffer.size]
    legal_masks = buffer.legal_masks[: buffer.size]

    # Split into train/test
    num_train = int(num_experiences * (1 - args.test_split))
    indices = np.arange(num_experiences)
    np.random.shuffle(indices)

    train_idx = indices[:num_train]
    test_idx = indices[num_train:]

    X_train = X[train_idx]
    actions_train = actions[train_idx]
    rewards_train = rewards[train_idx]
    masks_train = legal_masks[train_idx]

    X_test = X[test_idx]
    actions_test = actions[test_idx]
    rewards_test = rewards[test_idx]
    masks_test = legal_masks[test_idx]

    print(f"\nTrain set: {len(train_idx)} examples")
    print(f"Test set: {len(test_idx)} examples")

    # Train model
    print(f"\nTraining TreeQModel (max_depth={args.max_depth})...")
    model = TreeQModel(num_actions=7, max_depth=args.max_depth)
    model.fit(X_train, actions_train, rewards_train, masks_train)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_q_values = model.predict_q_values(X_test)

    # Compute accuracy: fraction of times the model picks the action that was taken
    correct = 0
    for i in range(len(test_idx)):
        predicted_action = model.predict_best_action(X_test[i], masks_test[i])
        if predicted_action == actions_test[i]:
            correct += 1

    accuracy = correct / len(test_idx)
    print(f"Action accuracy: {accuracy:.2%}")

    # Compute MSE on test rewards
    test_rewards_pred = np.array(
        [test_q_values[i, actions_test[i]] for i in range(len(test_idx))]
    )
    mse = np.mean((test_rewards_pred - rewards_test) ** 2)
    print(f"Reward MSE: {mse:.6f}")

    # Save model
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"\nModel saved to {args.out}")
    print("Done!")


if __name__ == "__main__":
    main()
