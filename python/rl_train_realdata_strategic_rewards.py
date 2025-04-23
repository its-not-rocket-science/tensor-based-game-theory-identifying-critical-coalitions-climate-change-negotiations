"""
RL Training Script with Real-Data Tipping Reward + Policy Constraints

Implements:
- Real tipping reward (alignment × spread × diversity)
- Size penalty (optional)
- Coalition size cap (optional)
- Reward threshold (optional)

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

# -------------------------------
# Parameters (change here)
# -------------------------------
EPOCHS = 1000
SIZE_CAP = 8  # Hard upper limit on coalition size (None for unlimited)
SIZE_PENALTY_ALPHA = 1.0  # Reward /= size^alpha (set 0 for no penalty)
REWARD_THRESHOLD = 500.0  # No reward unless tipping score ≥ threshold (None to disable)

# -------------------------------
# Load data
# -------------------------------
country_df = pd.read_csv("minor_country_index.csv")
countries = country_df["Country"].tolist()
index_to_country = {i: c for i, c in enumerate(countries)}
N = len(countries)

cos_sim = np.load("cosine_similarity.npy")
influence_matrix = np.load("influence_matrix.npy")

cluster_df = pd.read_csv("table_countries_by_cluster.csv")
cluster_map = cluster_df.set_index("Country")["ClusterLabel"].to_dict()

# -------------------------------
# Reward Function
# -------------------------------
def tipping_score_reward(coalition_indices):
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
    clusters = {cluster_map.get(index_to_country[i], "Unknown") for i in coalition_indices}
    diversity = len(clusters)
    score = alignment * spread * diversity

    if SIZE_PENALTY_ALPHA > 0:
        score /= len(coalition_indices) ** SIZE_PENALTY_ALPHA

    if REWARD_THRESHOLD is not None and score < REWARD_THRESHOLD:
        return 0.0

    return score

# -------------------------------
# Policy Network
# -------------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.fc = nn.Linear(n, n)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

# -------------------------------
# Train Policy
# -------------------------------
def train_rl_policy(n_nations=N, epochs=EPOCHS, lr=0.01):
    model = PolicyNetwork(n_nations)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    rewards = []
    sizes = []
    prob_history = []

    for _ in range(epochs):
        state = torch.rand(n_nations)
        probs = model(state)
        prob_history.append(probs.detach().numpy())

        mask = torch.bernoulli(probs).bool()
        coalition = torch.where(mask)[0].tolist()

        if len(coalition) < 2:
            continue
        if SIZE_CAP is not None and len(coalition) > SIZE_CAP:
            continue

        reward = tipping_score_reward(coalition)
        if reward == 0.0:
            continue

        log_probs = torch.log(probs[mask])
        loss = -log_probs.sum() * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rewards.append(reward)
        sizes.append(len(coalition))

    return model, rewards, sizes, np.array(prob_history)

# -------------------------------
# Sample Coalitions from Trained Policy
# -------------------------------
def sample_learned_coalitions(model, threshold=0.5, n_samples=1000):
    coalitions = []
    with torch.no_grad():
        for _ in range(n_samples):
            state = torch.rand(N)
            probs = model(state)
            selected = (probs > threshold).nonzero().flatten().tolist()
            if 2 <= len(selected) <= (SIZE_CAP or N):
                named_coalition = tuple(sorted(index_to_country[i] for i in selected))
                coalitions.append(named_coalition)
    return coalitions

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    model, rewards, sizes, prob_history = train_rl_policy()

    # Plot metrics
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
    plt.savefig("rl_realdata_training_summary.png")
    print("✅ Saved: rl_realdata_training_summary.png")

    # Export learned coalitions
    coalitions = sample_learned_coalitions(model)
    pd.DataFrame({"Coalition": coalitions}).drop_duplicates().to_csv("rl_learned_coalitions_realdata.csv", index=False)
    print("✅ Exported: rl_learned_coalitions_realdata.csv")
