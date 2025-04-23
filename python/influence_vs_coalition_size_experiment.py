# =============================================================
# Experiment: Influence vs. Coalition Size in a Tensor Game
# =============================================================
# This script simulates how the size of a minor player coalition affects
# systemic cooperation levels and tipping susceptibility in a climate game
# using a tensor-based influence model.
# It evaluates the dominant eigenvalue (as a proxy for tipping) and the
# average final cooperation level for increasing coalition sizes.
# =============================================================

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvals
import matplotlib.ticker as mtick


def sigmoid(x):
    """Sigmoid activation function for bounded rationality."""
    return 1 / (1 + np.exp(-x))


def strategy_dynamics(num_players, strategies, influence_tensor, intrinsic_biases):
    """Compute the rate of change of each player's strategy."""
    ds_dt = np.zeros(num_players)
    for i in range(num_players):
        influence = sum(
            influence_tensor[i, j, k] * strategies[j] * strategies[k]
            for j in range(num_players) for k in range(num_players)
        )
        ds_dt[i] = -strategies[i] + sigmoid(influence + intrinsic_biases[i])
    return ds_dt


def run_simulation(num_players, influence_tensor, biases, initial_strategies, timesteps=100, dt=0.1):
    """Run a simulation of player strategy evolution over time."""
    strategies = np.array(initial_strategies)
    history = [strategies.copy()]
    for _ in range(timesteps):
        ds = strategy_dynamics(num_players, strategies,
                               influence_tensor, biases)
        strategies += dt * ds
        history.append(strategies.copy())
    return np.array(history)


def amplify_tensor(tensor, coalition, eta=2.0):
    """Amplify influence of coalition members within the tensor."""
    perturbed_tensor = tensor.copy()
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            for k in range(tensor.shape[2]):
                if j in coalition or k in coalition:
                    perturbed_tensor[i, j, k] *= eta
    return perturbed_tensor


def compute_tensor_eigenvalue(num_players, tensor, strategy_vector):
    """Compute the dominant eigenvalue of a tensor-based influence matrix."""
    influence_matrix = np.zeros((num_players, num_players))
    for i in range(num_players):
        for j in range(num_players):
            influence_matrix[i, j] = sum(
                tensor[i, j, k] * strategy_vector[k] for k in range(num_players)
            )
    computed_eigenvalues = eigvals(influence_matrix)
    return max(computed_eigenvalues.real)


# -----------------------------
# Experiment Configuration
# -----------------------------
NUM_PLAYERS = 10
np.random.seed(42)

simulation_influence_tensor = np.random.rand(NUM_PLAYERS, NUM_PLAYERS, NUM_PLAYERS)
simulation_influence_tensor[0:3] *= 2.0  # Major players (players 0-2)

simulation_intrinsic_biases = np.random.uniform(-1, 1, NUM_PLAYERS)
simulation_initial_strategies = np.random.uniform(0.1, 0.3, NUM_PLAYERS)

# Sweep over coalition sizes among minor players (3 to 9)
minor_players = list(range(3, NUM_PLAYERS))
max_size = len(minor_players)
eigenvalues = []
avg_strategies = []

for size in range(1, max_size + 1):
    coalition_list = minor_players[:size]
    strategies_copy = simulation_initial_strategies.copy()
    for idx in coalition_list:
        strategies_copy[idx] = 1.0  # Force full cooperation

    amplified_tensor = amplify_tensor(simulation_influence_tensor, coalition_list, eta=2.5)
    simulation_history = run_simulation(NUM_PLAYERS, amplified_tensor,
                             simulation_intrinsic_biases, strategies_copy, timesteps=200, dt=0.05)

    final_strategies = simulation_history[-1]
    lambda_max = compute_tensor_eigenvalue(
        NUM_PLAYERS, amplified_tensor, final_strategies)

    eigenvalues.append(lambda_max)
    avg_strategies.append(np.mean(final_strategies))

# -----------------------------
# Plotting the Results
# -----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.plot(range(1, len(minor_players) + 1), eigenvalues, marker='o')
ax1.set_xlabel("Coalition Size")
ax1.set_ylabel("Dominant Eigenvalue")
ax1.set_title("Influence vs. Coalition Size")
ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

ax2.plot(range(1, len(minor_players) + 1), avg_strategies, marker='s', color='green')
ax2.set_xlabel("Coalition Size")
ax2.set_ylabel("Average Final Cooperation")
ax2.set_title("Cooperation Level vs. Coalition Size")
ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.plot(range(1, max_size + 1), eigenvalues, marker='o')
# plt.xlabel("Coalition Size")
# plt.ylabel("Dominant Eigenvalue")
# plt.title("Influence vs. Coalition Size")

# plt.subplot(1, 2, 2)
# plt.plot(range(1, max_size + 1), avg_strategies, marker='s', color='green')
# plt.xlabel("Coalition Size")
# plt.ylabel("Avg Final Cooperation")
# plt.title("Cooperation Level vs. Coalition Size")

plt.tight_layout()
plt.savefig("../coalition_sweep_plot.png", dpi=300)
plt.show()
