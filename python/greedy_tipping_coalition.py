
"""
Greedy Search for Minimal Tipping Coalitions
--------------------------------------------
This script simulates a climate coalition tipping model using a greedy algorithm.

It selects players one-by-one to maximize marginal increases in a simulated
dominant eigenvalue of the system. The goal is to find the smallest coalition
whose combined influence tips the system past a cooperation threshold.

Output:
    - Plot saved to: figures/figure_greedy_tipping.png
    - Coalition printed to console
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Simulation Configuration
# -----------------------------

np.random.seed(42)
N_PLAYERS = 10
sample_players = list(range(N_PLAYERS))
BASELINE_LAMBDA = 15.0  # Simulated starting eigenvalue
LAMBDA_CRIT = 30.0      # Tipping threshold


# -----------------------------
# Influence Tensor Mockup
# -----------------------------

def mock_lambda(coalition, amplify=1.5):
    """
    Simulate the dominant eigenvalue for a coalition.
    The more influential and coordinated the group, the higher the result.
    """
    size = len(coalition)
    if size == 0:
        return BASELINE_LAMBDA
    influence = sum(np.sqrt(i + 1) for i in coalition) * amplify
    return BASELINE_LAMBDA + influence


# -----------------------------
# Greedy Coalition Builder
# -----------------------------

def greedy_tipping_search(players, lambda_fn, threshold, max_size=6):
    """
    Iteratively adds players that maximize marginal increase in lambda.
    Stops when threshold is crossed or max size reached.
    """
    coalition = []
    lambda_progress = []

    current_lambda = lambda_fn(coalition)
    lambda_progress.append((0, current_lambda))

    for _ in range(max_size):
        best_gain = -np.inf
        best_player = None

        for p in players:
            if p in coalition:
                continue
            test_coalition = coalition + [p]
            test_lambda = lambda_fn(test_coalition)
            gain = test_lambda - current_lambda

            if gain > best_gain:
                best_gain = gain
                best_player = p

        if best_player is None:
            break

        coalition.append(best_player)
        current_lambda = lambda_fn(coalition)
        lambda_progress.append((len(coalition), current_lambda))

        if current_lambda >= threshold:
            break

    return coalition, lambda_progress


# -----------------------------
# Run Greedy Search
# -----------------------------

greedy_coalition, lambda_data = greedy_tipping_search(
    sample_players, mock_lambda, LAMBDA_CRIT)

# -----------------------------
# Plotting Results
# -----------------------------

sizes, values = zip(*lambda_data)

plt.figure(figsize=(8, 5))
plt.plot(sizes, values, marker='o', linestyle='-', color='black')
plt.axhline(LAMBDA_CRIT, color='red', linestyle='--',
            label='Tipping Threshold')
plt.title("Greedy Coalition Tipping Progress")
plt.xlabel("Coalition Size")
plt.ylabel("Dominant Eigenvalue (Simulated)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("../figure_greedy_tipping.png")
plt.show()
plt.close()

print("✅ Greedy Tipping Coalition:", greedy_coalition)
