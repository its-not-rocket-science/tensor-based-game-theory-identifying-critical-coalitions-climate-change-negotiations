"""
Visualize RL Training Metrics for Climate Coalition Policy Learning

This script trains a REINFORCE agent under a threshold reward strategy
and plots:
- Reward per epoch
- Coalition size over time
- Average inclusion probabilities per player

Output: ../figure_rl_training_summary.png
"""

import torch
from torch import nn, optim
import numpy as np
from  matplotlib import pyplot as plt

# -------------------------------
# Policy Network Definition
# -------------------------------


class PolicyNetwork(nn.Module):
    """Simple sigmoid-layer policy over n_nations"""

    def __init__(self, n_nations):
        super().__init__()
        self.fc = nn.Linear(n_nations, n_nations)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))


# -------------------------------
# Reward Function: Threshold Based
# -------------------------------

def coalition_payoff(strategies, coalition):
    return np.array([1.0 if i in coalition else 0.0 for i in range(len(strategies))])


def reward_threshold_cooperation(strategies, coalition):
    if len(coalition) > 5:
        return 0.0
    return 1.0 if np.mean(strategies == 0) >= 0.9 else 0.0


# -------------------------------
# Train Policy with Logging
# -------------------------------

def train_with_logging(reward_fn, n_nations=10, epochs=1000):
    model = PolicyNetwork(n_nations)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    training_rewards = []
    training_sizes = []
    probs_log = []

    for _ in range(epochs):
        state = torch.rand(n_nations)
        probs = model(state)
        probs_log.append(probs.detach().numpy())

        coalition_mask = torch.bernoulli(probs).bool()
        coalition = torch.where(coalition_mask)[0].tolist()

        if not coalition:
            continue

        strategies = np.ones(n_nations)
        for i in coalition:
            strategies[i] = 0

        reward = reward_fn(strategies, coalition)
        if reward == 0.0 and len(coalition) > 5:
            continue

        log_probs = torch.log(probs[coalition_mask])
        loss = -log_probs.sum() * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_rewards.append(reward)
        training_sizes.append(len(coalition))

    return training_rewards, training_sizes, np.array(probs_log)

# -------------------------------
# Export Sampled Coalitions
# -------------------------------

def extract_final_coalitions(model, n_nations, num_samples=1000, threshold=0.5):
    coalitions = []
    with torch.no_grad():
        for _ in range(num_samples):
            dummy_input = torch.rand(n_nations)
            probs = model(dummy_input)
            selected = (probs > threshold).nonzero().flatten().tolist()
            if len(selected) >= 2:
                coalitions.append(tuple(sorted(selected)))
    return coalitions

# -------------------------------
# Run + Plot Metrics
# -------------------------------

if __name__ == "__main__":
    rewards, sizes, probs_matrix = train_with_logging(
        reward_threshold_cooperation)

    plt.figure(figsize=(12, 4))

    # Reward vs Epoch
    plt.subplot(1, 3, 1)
    plt.plot(rewards)
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.title("Reward over Training")

    # Coalition Size over Epoch
    plt.subplot(1, 3, 2)
    plt.plot(sizes)
    plt.xlabel("Epoch")
    plt.ylabel("Coalition Size")
    plt.title("Coalition Size per Epoch")

    # Final Policy Probability
    plt.subplot(1, 3, 3)
    avg_probs = probs_matrix.mean(axis=0)
    plt.bar(np.arange(len(avg_probs)), avg_probs)
    plt.xlabel("Nation Index")
    plt.ylabel("Average Inclusion Probability")
    plt.title("Learned Inclusion Probabilities")

    plt.tight_layout()
    plt.savefig("../figure_rl_training_summary.png")
    plt.show()

    # Re-run final extraction
    model_for_csv = PolicyNetwork(10)
    _, _, _ = train_with_logging(reward_threshold_cooperation, epochs=1000)
    final_coalitions = extract_final_coalitions(model_for_csv, n_nations=10)

    # Save to CSV
    import pandas as pd
    df = pd.DataFrame({"Coalition": final_coalitions})
    df.drop_duplicates(subset="Coalition", inplace=True)
    df.to_csv("outputs/rl_learned_coalitions.csv", index=False)
    print(f"✅ Exported {len(df)} RL-learned coalitions to outputs/rl_learned_coalitions.csv")
