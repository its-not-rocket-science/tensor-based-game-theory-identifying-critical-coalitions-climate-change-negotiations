"""
Generate Power Indices and Network Centrality Plot for Climate Coalition Players

This script estimates:
- Shapley and Banzhaf power indices using Monte Carlo sampling
- Betweenness and eigenvector centralities from a synthetic influence network

It produces a two-panel figure showing:
- Game-theoretic pivotality measures
- Network-based structural leverage

Output:
    - ../figure_power_centrality.png
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ----------------------------
# CONFIGURATION
# ----------------------------

np.random.seed(42)
NUM_PLAYERS = 10
sample_players = list(range(NUM_PLAYERS))


# ----------------------------
# PAYOFF FUNCTION
# ----------------------------

def coalition_payoff(strategies, coalition):
    """
    Returns a binary payoff vector:
    +1 for players in the coalition, 0 otherwise.
    """
    return np.array([1.0 if i in coalition else 0.0 for i in range(len(strategies))])


# ----------------------------
# POWER INDEX ESTIMATION
# ----------------------------

def compute_power_indices(players, payoff_func, trials=500):
    """
    Estimates Shapley and Banzhaf indices for each player.
    Uses random coalition sampling to evaluate marginal impact.
    """
    max_players = len(players)
    shapley = np.zeros(max_players)
    banzhaf = np.zeros(max_players)

    for _ in range(trials):
        coalition = np.random.choice(players, size=np.random.randint(1, len(players)), replace=False)

        for i in players:
            without_i = [p for p in coalition if p != i]
            v_with = np.sum(payoff_func(np.zeros(len(players)), coalition))
            v_without = np.sum(payoff_func(np.zeros(len(players)), without_i))

            shapley[i] += (v_with - v_without) / trials
            if i in coalition:
                banzhaf[i] += int(v_with - v_without > 0)

    banzhaf /= trials

    return pd.DataFrame({
        'Player': players,
        'Shapley Index': shapley,
        'Banzhaf Index': banzhaf
    })


# ----------------------------
# NETWORK CENTRALITY ANALYSIS
# ----------------------------

def generate_centrality_metrics(n_players):
    """
    Creates a synthetic influence network and computes centrality metrics.
    """
    G = nx.erdos_renyi_graph(n=n_players, p=0.4, seed=42)
    nx.set_edge_attributes(G, values=1, name='weight')

    betweenness = nx.betweenness_centrality(G)
    eigenvector = nx.eigenvector_centrality(G, max_iter=1000)

    return pd.DataFrame({
        'Player': list(betweenness.keys()),
        'Betweenness': list(betweenness.values()),
        'Eigenvector': [eigenvector[n] for n in G.nodes]
    })


# ----------------------------
# PLOTTING
# ----------------------------

def plot_power_and_centrality(power_df, centrality_df, filename):
    """
    Creates a two-panel figure showing:
    - Power indices (Shapley, Banzhaf) with fill patterns
    - Network centrality (Betweenness, Eigenvector) with fill patterns
    """
    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar width and position offset
    bar_width = 0.4
    x = np.arange(len(power_df['Player']))

    # Left: Power indices with hatching
    axes[0].bar(x - bar_width/2, power_df['Shapley Index'], width=bar_width,
                label='Shapley', alpha=0.8, edgecolor='black')
    axes[0].bar(x + bar_width/2, power_df['Banzhaf Index'], width=bar_width,
                label='Banzhaf', alpha=0.8, edgecolor='black')

    axes[0].set_title("Power Indices")
    axes[0].set_xlabel("Player")
    axes[0].set_ylabel("Index Value")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(power_df['Player'])
    axes[0].set_ylim(0, 0.6)  # Extended for legend clarity
    axes[0].legend()

    # Right: Network centrality with hatching
    x2 = np.arange(len(centrality_df['Player']))
    axes[1].bar(x2 - bar_width/2, centrality_df['Betweenness'], width=bar_width,
                label='Betweenness', alpha=0.8, edgecolor='black')
    axes[1].bar(x2 + bar_width/2, centrality_df['Eigenvector'], width=bar_width,
                label='Eigenvector', alpha=0.8, edgecolor='black')

    axes[1].set_title("Network Centrality Measures")
    axes[1].set_xlabel("Player")
    axes[1].set_ylabel("Centrality Score")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(centrality_df['Player'])
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close()



# ----------------------------
# MAIN SCRIPT EXECUTION
# ----------------------------

if __name__ == "__main__":
    power_df_main = compute_power_indices(sample_players, coalition_payoff)
    centrality_df_main = generate_centrality_metrics(NUM_PLAYERS)
    plot_power_and_centrality(power_df_main, centrality_df_main, "../figure_power_centrality.png")
