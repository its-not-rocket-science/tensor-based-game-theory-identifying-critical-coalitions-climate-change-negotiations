"""
RL Training Script Using Real Climate Data for Coalition Tipping

This script trains a REINFORCE-style agent to identify coalitions of nations that
maximize a real-data-based tipping score (influence × alignment × diversity).

It uses:
- A policy network over 189 minor countries
- Real influence and similarity matrices
- Cluster labels for diversity scoring

Outputs:
- rl_realdata_training_summary.png
- rl_learned_coalitions_realdata.csv
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

g20_members = {
    "Argentina", "Australia", "Brazil", "Canada", "China", "France", "Germany",
    "India", "Indonesia", "Italy", "Japan", "Mexico", "Russia", "Saudi Arabia",
    "South Africa", "South Korea", "Turkey", "United Kingdom", "United States",
    "European Union"
}

# -------------------------------
# Load Real Climate Data
# -------------------------------

# Load clusters for diversity measure
cluster_df = pd.read_csv("outputs/table_countries_by_cluster.csv")
cluster_map = cluster_df.set_index("Country")["ClusterLabel"].to_dict()

# Load minor country index
country_df = cluster_df[~cluster_df["Country"].isin(g20_members)].copy()
countries = country_df["Country"].tolist()
country_to_index = {c: i for i, c in enumerate(countries)}
index_to_country = {i: c for i, c in enumerate(countries)}
N = len(countries)

# Load influence and similarity matrices
cos_sim = np.load("outputs/cosine_similarity.npy")
influence_matrix = np.load("outputs/influence_matrix.npy")


# -------------------------------
# Reward Function (Real-Data-Based)
# -------------------------------

def tipping_score_reward(coalition_indices):
    """Real-data tipping reward: alignment × spread × diversity."""
    if len(coalition_indices) < 2:
        return 0.0

    alignment = np.mean([
        cos_sim[i, j]
        for i in coalition_indices for j in coalition_indices if i < j
    ]) if len(coalition_indices) > 1 else 0.0

    spread = sum(
        influence_matrix[i, j]
        for i in coalition_indices for j in range(N)
        if j not in coalition_indices
    )

    clusters = {
        cluster_map.get(index_to_country[i], "Unknown") for i in coalition_indices
    }
    diversity = len(clusters)

    return alignment * spread * diversity

# -------------------------------
# Policy Network Definition
# -------------------------------


class PolicyNetwork(nn.Module):
    """Simple sigmoid-layer policy model for selecting coalitions."""

    def __init__(self, n):
        super().__init__()
        self.fc = nn.Linear(n, n)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

# -------------------------------
# RL Training Loop
# -------------------------------


def train_rl_policy(n_nations=N, epochs=1000, lr=0.01):
    train_rl_model = PolicyNetwork(n_nations)
    optimizer = optim.Adam(train_rl_model.parameters(), lr=lr)

    train_rl_rewards = []
    train_rl_sizes = []
    train_rl_prob_history = []

    for _ in range(epochs):
        state = torch.rand(n_nations)
        probs = train_rl_model(state)
        train_rl_prob_history.append(probs.detach().numpy())

        # Sample a coalition
        mask = torch.bernoulli(probs).bool()
        coalition = torch.where(mask)[0].tolist()

        if len(coalition) < 2:
            continue

        # Calculate reward
        reward = tipping_score_reward(coalition)
        if reward == 0.0:
            continue

        # Policy gradient update
        log_probs = torch.log(probs[mask])
        loss = -log_probs.sum() * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log training stats
        train_rl_rewards.append(reward)
        train_rl_sizes.append(len(coalition))

    return train_rl_model, train_rl_rewards, train_rl_sizes, np.array(train_rl_prob_history)

# -------------------------------
# Sample Coalitions from Trained Policy
# -------------------------------


def sample_learned_coalitions(sample_model, threshold=0.5, n_samples=1000):
    """Sample coalitions using the trained model and return unique results."""
    coalitions = []
    with torch.no_grad():
        for _ in range(n_samples):
            state = torch.rand(N)
            probs = sample_model(state)
            selected = (probs > threshold).nonzero().flatten().tolist()
            if len(selected) >= 2:
                named_coalition = tuple(
                    sorted(index_to_country[i] for i in selected))
                coalitions.append(named_coalition)
    return coalitions

# -------------------------------
# Main Execution
# -------------------------------


if __name__ == "__main__":
    # Train policy network
    model, rewards, sizes, prob_history = train_rl_policy()

    # Plot training metrics
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(rewards)
    plt.title("Reward Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")

    plt.subplot(1, 3, 2)
    plt.plot(sizes)
    plt.title("Coalition Size")
    plt.xlabel("Epoch")
    plt.ylabel("Size")

    plt.subplot(1, 3, 3)
    avg_probs = prob_history.mean(axis=0)
    plt.bar(np.arange(N), avg_probs)
    plt.title("Average Inclusion Probabilities")
    plt.xlabel("Country Index")
    plt.ylabel("Probability")

    plt.tight_layout()
    plt.savefig("../rl_realdata_training_summary.png", dpi=300)
    plt.show()
    plt.close()
    print("✅ Saved: rl_realdata_training_summary.png")

    # Export learned coalitions to CSV
    final_coalitions = sample_learned_coalitions(model)
    pd.DataFrame({"Coalition": final_coalitions}).drop_duplicates().to_csv(
        "outputs/rl_learned_coalitions_realdata.csv", index=False
    )
    print("✅ Exported: rl_learned_coalitions_realdata.csv")
