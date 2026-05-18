"""
Greedy forward-selection of minor player coalitions with CLI-configurable size range.

Uses corrected pipeline outputs (cosine_similarity.npy, influence_matrix.npy,
minor_country_index.csv) so scoring is consistent with the exhaustive search
and RL reward function.

TippingScore(C) = Alignment(C) * InfluenceSpread(C) * ClusterDiversity(C)
  Alignment      = mean pairwise cosine similarity within C
  InfluenceSpread = sum of influence_matrix[i,j] for i in C, j not in C
  ClusterDiversity = number of distinct treaty-participation clusters in C

Usage:
    python generate_greedy_coalitions.py --min-size 2 --max-size 6
"""

import argparse
from collections import Counter
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--min-size", type=int, default=2)
parser.add_argument("--max-size", type=int, default=6)
args = parser.parse_args()

MIN_SIZE = args.min_size
MAX_SIZE = args.max_size

# -----------------------------------------------------------------------
# Load corrected pipeline outputs
# -----------------------------------------------------------------------
cos_sim = np.load("outputs/cosine_similarity.npy")
influence_matrix = np.load("outputs/influence_matrix.npy")
row_sums = influence_matrix.sum(axis=1)

minor_df = pd.read_csv("outputs/minor_country_index.csv")
countries = minor_df["Country"].tolist()
N = len(countries)
country_to_idx = {c: i for i, c in enumerate(countries)}

cluster_df = pd.read_csv("outputs/table_countries_by_cluster.csv")
cluster_map = cluster_df.set_index("Country")["ClusterLabel"].to_dict()

print(f"Loaded {N} minor players, max coalition size {MAX_SIZE}")

# -----------------------------------------------------------------------
# Scoring — identical formula to exhaustive search
# -----------------------------------------------------------------------
def score(indices):
    k = len(indices)
    if k < 2:
        return 0.0
    c = list(indices)
    sub_cos = cos_sim[np.ix_(c, c)]
    alignment = (sub_cos.sum() - k) / (k * (k - 1))
    sub_inf = influence_matrix[np.ix_(c, c)]
    spread = row_sums[c].sum() - sub_inf.sum()
    diversity = len({cluster_map.get(countries[i], "") for i in c})
    return alignment * spread * diversity


def is_valid(indices):
    clusters = [cluster_map.get(countries[i], "") for i in indices]
    return max(Counter(clusters).values()) <= len(indices) // 2


# -----------------------------------------------------------------------
# Greedy forward selection — one coalition per starting country
# -----------------------------------------------------------------------
results = []

for start in range(N):
    coalition = [start]
    available = [i for i in range(N) if i != start]

    while len(coalition) < MAX_SIZE:
        best_gain = 0.0
        best_idx = None
        current_score = score(coalition)

        for candidate in available:
            test = coalition + [candidate]
            if len(test) < MIN_SIZE:
                continue
            if not is_valid(test):
                continue
            gain = score(test) - current_score
            if gain > best_gain:
                best_gain = gain
                best_idx = candidate

        if best_idx is not None:
            coalition.append(best_idx)
            available.remove(best_idx)
        else:
            break

    if len(coalition) >= MIN_SIZE:
        names = [countries[i] for i in coalition]
        results.append((score(coalition), str(names)))

# -----------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------
results.sort(reverse=True)
out_df = pd.DataFrame(results, columns=["Score", "Coalition"])
out_path = "outputs/greedy_fast_minor_tipping_coalitions.csv"
out_df.to_csv(out_path, index=False)
print(f"Saved {len(out_df)} coalitions -> {out_path}")
print(f"Top score: {results[0][0]:.1f}  {results[0][1]}")
