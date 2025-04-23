"""
Improved RL Comparison: Strategic Coalition Formation for Climate Cooperation

This script trains 3 reinforcement learning agents using different reward models:
1. Raw Reward: encourages full inclusion.
2. Penalized Reward: favors compact, efficient coalitions.
3. Threshold-Based Reward: rewards only coalitions that meet tipping criteria
   with a size cap (≤ 5).

Each policy is trained independently and compared on its final learned coalition.
"""

import torch
from torch import nn, optim
import numpy as np

# -------------------------------
# Define Policy Network
# -------------------------------

class PolicyNetwork(nn.Module):
    """A one-layer sigmoid policy for selecting nations into a coalition."""
    def __init__(self, n_nations):
        super().__init__()
        self.fc = nn.Linear(n_nations, n_nations)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))


# -------------------------------
# Payoff + Reward Definitions
# -------------------------------

def coalition_payoff(strategies, coalition):
    """
    Placeholder payoff: each cooperating nation contributes +1.
    """
    return np.array([1.0 if i in coalition else 0.0 for i in range(len(strategies))])

def reward_raw_size(strategies, coalition):
    """Maximize number of cooperating players."""
    return np.sum(coalition_payoff(strategies, coalition))

def reward_penalized_size(strategies, coalition):
    """
    Encourage smaller but still impactful coalitions.
    Penalize size moderately, and penalize empty coalition.
    """
    if not coalition:
        return -1.0
    reward = np.sum(coalition_payoff(strategies, coalition))
    penalty = 0.1 * (len(coalition) ** 2)
    return reward - penalty
    # return np.sum(coalition_payoff(strategies, coalition)) - 0.5 * len(coalition)
    # cooperation = np.sum(coalition_payoff(strategies, coalition))
    # penalty = 0.2 * (len(coalition) ** 1.5)
    # return cooperation - penalty

def reward_threshold_cooperation(strategies, coalition):
    """
    Reward only if ≥90% cooperation achieved with ≤5 members.
    Larger coalitions are disqualified.
    """
    if len(coalition) > 5:
        return 0.0
    cooperation_rate = np.mean(strategies == 0)
    return 1.0 if cooperation_rate >= 0.9 else 0.0


# -------------------------------
# Training Function
# -------------------------------

def train_policy(reward_fn, n_nations=10, epochs=1000, name=""):
    """
    Train a policy using REINFORCE and return the final selected coalition.

    Args:
        reward_fn: Function defining the reward signal.
        n_nations: Number of nations in the system.
        epochs: Training steps.
        name: Name of reward strategy (for logging).
    """
    model = PolicyNetwork(n_nations)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        # Generate synthetic input vector (placeholder for nation state/features)
        state = torch.rand(n_nations)
        probs = model(state)

        # Sample a binary vector of coalition members
        coalition_mask = torch.bernoulli(probs).bool()
        coalition = torch.where(coalition_mask)[0].tolist()

        if not coalition:
            continue  # Skip training on empty sets

        # Construct strategy vector (0 = cooperate, 1 = defect)
        strategies = np.ones(n_nations)
        for i in coalition:
            strategies[i] = 0

        # Calculate reward and policy gradient loss
        reward = reward_fn(strategies, coalition)

        # Skip training on oversized sets for threshold-based reward
        if reward == 0.0 and name == "Threshold-Based" and len(coalition) > 5:
            continue

        log_probs = torch.log(probs[coalition_mask])
        loss = -log_probs.sum() * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Optional: Print progress every 200 steps
        if epoch % 200 == 0:
            print(f"[{name}] Epoch {epoch}: Reward={reward:.2f}, Size={len(coalition)}")

    # Final evaluation
    with torch.no_grad():
        final_probs = model(torch.ones(n_nations))  # test with all-1s
        final_coalition = torch.where(final_probs > 0.5)[0].tolist()

    return final_coalition


# -------------------------------
# Main Execution
# -------------------------------

if __name__ == "__main__":
    print("\nTraining with Raw Total Reward...")
    coalition_raw = train_policy(reward_raw_size, name="Raw Total")

    print("\nTraining with Penalized Size Reward...")
    coalition_penalized = train_policy(reward_penalized_size, name="Penalized Size")

    print("\nTraining with Threshold-Based Reward...")
    coalition_threshold = train_policy(reward_threshold_cooperation, name="Threshold-Based")

    # Final summary
    print("\n✅ Final Coalitions by Reward Function:")
    print("Raw Total Reward:       ", coalition_raw)
    print("Penalized Size Reward:  ", coalition_penalized)
    print("Threshold-Based Reward: ", coalition_threshold)
