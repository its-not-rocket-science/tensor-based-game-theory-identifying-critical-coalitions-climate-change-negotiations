"""
Generate a Coalition Efficiency Plot

This script simulates all coalitions of minor players in a climate cooperation
game and calculates their efficiency (eigenvalue gain per member).
"""

import itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, ticker as mtick
import seaborn as sns
from numpy.linalg import eigvals

# ----------------------------
# Model Parameters and Setup
# ----------------------------

np.random.seed(42)

NUM_PLAYERS = 10
major_players = [0, 1, 2]
minor_players = [i for i in range(NUM_PLAYERS) if i not in major_players]

# Tensor and player settings
influence_tensor = np.random.rand(NUM_PLAYERS, NUM_PLAYERS, NUM_PLAYERS)
influence_tensor[major_players] *= 2.0  # Amplify major players' influence

intrinsic_biases = np.random.uniform(-1, 1, NUM_PLAYERS)
initial_strategies = np.random.uniform(0.1, 0.3, NUM_PLAYERS)

AMPLIFICATION_FACTOR = 2.5
# DT = 0.05
# TIMESTEPS = 200

# ----------------------------
# Model Dynamics
# ----------------------------


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def strategy_dynamics(strategies, tensor, biases):
    """Calculate rate of change for each player's strategy."""
    n = len(strategies)
    ds_dt = np.zeros(n)
    for i in range(n):
        influence = sum(
            tensor[i, j, k] * strategies[j] * strategies[k]
            for j in range(n) for k in range(n)
        )
        ds_dt[i] = -strategies[i] + sigmoid(influence + biases[i])
    return ds_dt


def run_simulation(n, tensor, biases, initial, steps=200, dt=0.05):
    """Simulate the strategy dynamics forward in time."""
    strategies = np.array(initial)
    for _ in range(steps):
        delta = strategy_dynamics(strategies, tensor, biases)
        strategies += dt * delta
    return strategies


def amplify_tensor(tensor, coalition, eta):
    """Boost influence of coalition members in tensor."""
    t = tensor.copy()
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            for k in range(t.shape[2]):
                if j in coalition or k in coalition:
                    t[i, j, k] *= eta
    return t


def compute_dominant_eigenvalue(tensor, strategies):
    """Construct and compute max real eigenvalue of influence matrix."""
    n = len(strategies)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = sum(tensor[i, j, k] * strategies[k]
                               for k in range(n))
    return max(eigvals(matrix).real)

# ----------------------------
# Baseline (No Coalition)
# ----------------------------


baseline_final = run_simulation(
    NUM_PLAYERS, influence_tensor, intrinsic_biases, initial_strategies)
lambda_baseline = compute_dominant_eigenvalue(influence_tensor, baseline_final)

# ----------------------------
# Coalition Search + Efficiency Computation
# ----------------------------

records = []

for size in range(1, len(minor_players) + 1):
    for coalition_minor in itertools.combinations(minor_players, size):
        # Force full cooperation for the coalition
        coalition_strategies = initial_strategies.copy()
        for idx in coalition_minor:
            coalition_strategies[idx] = 1.0

        # Amplify influence
        tensor_amp = amplify_tensor(
            influence_tensor, coalition_minor, AMPLIFICATION_FACTOR)
        final_state = run_simulation(
            NUM_PLAYERS, tensor_amp, intrinsic_biases, coalition_strategies)

        # Metrics
        lambda_val = compute_dominant_eigenvalue(tensor_amp, final_state)
        efficiency = (lambda_val - lambda_baseline) / size

        records.append({
            "Coalition": coalition_minor,
            "Size": size,
            "Lambda": lambda_val,
            "Efficiency": efficiency
        })

# Convert results to DataFrame
df = pd.DataFrame(records)

# ----------------------------
# Plot: Coalition Efficiency vs. Size
# ----------------------------

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Size", y="Efficiency",
            palette="Blues", showfliers=False)

plt.axhline(0, linestyle='--', color='gray', linewidth=1)
plt.title("Coalition Efficiency vs. Coalition Size")
plt.xlabel("Coalition Size")
plt.ylabel("Efficiency (ΔEigenvalue per Member)")

plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
plt.tight_layout()

# Save the figure
plt.savefig("../coalition_efficiency_plot.png", dpi=300)
plt.show()
