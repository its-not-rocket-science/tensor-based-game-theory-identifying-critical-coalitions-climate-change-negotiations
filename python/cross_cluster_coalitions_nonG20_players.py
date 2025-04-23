"""
Score all cross-cluster coalitions of minor players (non-G20) using:
- Internal similarity (alignment)
- Influence spread to others
- Cluster diversity

The code computes a tipping score for each coalition and saves the top 50.
Adds progress tracking and timing.
"""

import itertools
import time
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Load Data
# -------------------------------

# Load ratification matrix
ratification_df = pd.read_excel("ratification data/Treaty data.xlsx", sheet_name="Data")
ratification_matrix = ratification_df.pivot_table(
    index="Country", columns="Name", values="ratified", aggfunc="max", fill_value=0
)

# Load clusters and define minor players
cluster_df = pd.read_csv("outputs/table_countries_by_cluster.csv")
g20_members = {
    "Argentina", "Australia", "Brazil", "Canada", "China", "France", "Germany",
    "India", "Indonesia", "Italy", "Japan", "Mexico", "Russia", "Saudi Arabia",
    "South Africa", "South Korea", "Turkey", "United Kingdom", "United States",
    "European Union"
}
cluster_df["IsMinor"] = ~cluster_df["Country"].isin(g20_members)

# Prepare inputs
minor_players = cluster_df[cluster_df["IsMinor"]]["Country"].tolist()
countries = ratification_matrix.index.tolist()
similarity_matrix = cosine_similarity(ratification_matrix)
country_idx = {name: i for i, name in enumerate(countries)}
cluster_map = cluster_df.set_index("Country")["ClusterLabel"].to_dict()

# -------------------------------
# Scoring Function
# -------------------------------

def compute_tipping_score(coalition):
    """
    Compute score based on:
    - average internal similarity
    - influence spread to all non-members
    - number of distinct clusters in the coalition
    """
    if len(coalition) < 2:
        return 0.0

    indices = [country_idx[c] for c in coalition]
    others = [i for i in range(len(countries)) if countries[i] not in coalition]

    # Internal alignment
    internal_sim = [
        similarity_matrix[i][j]
        for i, j in itertools.combinations(indices, 2)
    ]
    alignment = np.mean(internal_sim) if internal_sim else 0.0

    # Influence spread
    spread = sum(similarity_matrix[o][j] for o in others for j in indices)

    # Cluster diversity
    clusters = {cluster_map.get(c, "Unknown") for c in coalition}
    diversity = len(clusters)

    return alignment * spread * diversity


# -------------------------------
# Exhaustive Search Over Combinations
# -------------------------------

results = []
start_time = time.time()
checked = 0

print(f"Starting coalition scoring for {len(minor_players)} minor players...")

for size in range(2, 6):  # Coalition sizes 2 to 5
    combos = list(itertools.combinations(minor_players, size))
    print(f"Checking {len(combos):,} combinations of size {size}...")

    for group in combos:
        score = compute_tipping_score(group)
        results.append((score, group))
        checked += 1

        if checked % 10000 == 0:
            elapsed = time.time() - start_time
            print(f"Checked {checked:,} coalitions in {elapsed:.1f} seconds")
            
    results_df = pd.DataFrame(results, columns=["Score", "Coalition"])
    csv_name = f"outputs/minor_tipping_coalitions_2_to_{size}_members.csv"
    results_df.to_csv(csv_name, index=False)
    print(f"\n✅ Interim results saved to {csv_name} at {time.time() - start_time:.1f} seconds.")

# Sort and export
results.sort(reverse=True)
top_df = pd.DataFrame(results[:50], columns=["Score", "Coalition"])
top_df.to_csv("outputs/top_50_minor_tipping_coalitions_2_to_5_members.csv", index=False)

print(f"\n✅ Finished. Scored {checked:,} coalitions in {time.time() - start_time:.1f} seconds.")
