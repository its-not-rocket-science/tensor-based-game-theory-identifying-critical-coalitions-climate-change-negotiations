# ================================================================
# Tensor-Based Climate Strategy Simulation (Fully Commented)
# ================================================================
# This script simulates how players (e.g., countries) in a climate negotiation game
# update their cooperation strategies over time.
# A 3rd-order tensor encodes the nonlinear influence each player has on others.
# We explore how a small coalition of minor players can tip the system
# toward high cooperation through coordinated action and increased influence.
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvals

# ---------------------------
# Mathematical Model Functions
# ---------------------------

def sigmoid(x):
    """Sigmoid activation function to model bounded rationality.
    Players respond gradually to incentives rather than making binary jumps.
    """
    return 1 / (1 + np.exp(-x))

def strategy_dynamics(num_players, strategies, influence_tensor, intrinsic_biases):
    """Compute the time derivative of each player's strategy.
    
    Each player is influenced by all other players according to the 3D influence tensor.
    They also have their own internal bias (e.g., economic costs/benefits).
    """
    ds_dt = np.zeros(num_players)
    for i in range(num_players):
        influence = sum(
            influence_tensor[i, j, k] * strategies[j] * strategies[k]
            for j in range(num_players) for k in range(num_players)
        )
        ds_dt[i] = -strategies[i] + sigmoid(influence + intrinsic_biases[i])
    return ds_dt

def run_simulation(num_players, influence_tensor, biases, initial_strategies, timesteps=100, dt=0.1):
    """Simulate strategy evolution using simple time-stepping (Euler integration)."""
    strategies = np.array(initial_strategies)
    history = [strategies.copy()]
    for _ in range(timesteps):
        ds = strategy_dynamics(num_players, strategies, influence_tensor, biases)
        strategies += dt * ds
        history.append(strategies.copy())
    return np.array(history)

def amplify_tensor(tensor, coalition, eta=2.0):
    """Amplify the influence of coalition members by scaling their interaction terms."""
    perturbed_tensor = tensor.copy()
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            for k in range(tensor.shape[2]):
                if j in coalition or k in coalition:
                    perturbed_tensor[i, j, k] *= eta
    return perturbed_tensor

def compute_tensor_eigenvalue(num_players,tensor, strategy_vector):
    """Approximate the dominant eigenvalue of a tensor-based influence matrix.
    
    This is used as a proxy for the system's susceptibility to tipping.
    A higher eigenvalue implies more sensitivity to player interactions.
    """
    influence_matrix = np.zeros((num_players, num_players))
    for i in range(num_players):
        for j in range(num_players):
            influence_matrix[i, j] = sum(
                tensor[i, j, k] * strategy_vector[k] for k in range(num_players)
            )
    eigenvalues = eigvals(influence_matrix)
    return max(eigenvalues.real)

# ---------------------------
# Setup: Define Players and Conditions
# ---------------------------

NUM_PLAYERS = 5
np.random.seed(0)  # For reproducibility

# Initialize random influence tensor and scale influence of major players (players 0 and 1)
simulation_influence_tensor = np.random.rand(NUM_PLAYERS, NUM_PLAYERS, NUM_PLAYERS)
simulation_influence_tensor[0:2] *= 2.0  # Major players have more influence

# Each player has an intrinsic bias reflecting internal cost-benefit logic
simulation_intrinsic_biases = np.random.uniform(-1, 1, NUM_PLAYERS)

# Initial strategy levels (low cooperation to start)
initial_strategy_levels = np.random.uniform(0.1, 0.3, NUM_PLAYERS)

# Define a coalition of minor players (e.g., small nations)
minor_players = [2, 3, 4]
coalition_players = [3, 4]
strategies_with_coalition = initial_strategy_levels.copy()
for idx in coalition_players:
    strategies_with_coalition[idx] = 1.0  # Force full cooperation in coalition

# ---------------------------
# Simulate System Dynamics
# ---------------------------

# 1. Baseline: No coalition
history_no_coalition = run_simulation(
    NUM_PLAYERS, simulation_influence_tensor, simulation_intrinsic_biases,
    initial_strategy_levels, timesteps=200, dt=0.05
)
lambda_no_coalition = compute_tensor_eigenvalue(
    NUM_PLAYERS, simulation_influence_tensor, history_no_coalition[-1]
)

# 2. With coalition: Amplify influence of cooperative minor players
amplified_tensor = amplify_tensor(simulation_influence_tensor, coalition_players, eta=2.5)
history_with_coalition = run_simulation(
    NUM_PLAYERS, amplified_tensor, simulation_intrinsic_biases,
    strategies_with_coalition, timesteps=200, dt=0.05
)
lambda_with_coalition = compute_tensor_eigenvalue(
    NUM_PLAYERS, amplified_tensor, history_with_coalition[-1]
)

# ---------------------------
# Visualization
# ---------------------------

plt.figure(figsize=(10, 6))

# Plot strategy evolution for each player, both scenarios
for i in range(NUM_PLAYERS):
    plt.plot(history_no_coalition[:, i], '--', label=f'Player {i} (no coalition)')
    plt.plot(history_with_coalition[:, i], label=f'Player {i} (with coalition)')

plt.xlabel("Time Step")
plt.ylabel("Strategy (Cooperation Level)")
plt.title("Effect of Minor Coalition on Strategy Evolution")

# Annotate chart with final eigenvalues
textstr = f"Final Dominant Eigenvalues\n" \
          f"No Coalition: {lambda_no_coalition:.2f}\n" \
          f"With Coalition: {lambda_with_coalition:.2f}"
plt.gcf().text(0.77, 0.45, textstr, fontsize=10, bbox=dict(facecolor='white', edgecolor='gray'))

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

# ---------------------------
# Save Chart to File
# ---------------------------

plt.savefig("../figure_coalition_effect.png", dpi=300, bbox_inches='tight')
plt.show()
