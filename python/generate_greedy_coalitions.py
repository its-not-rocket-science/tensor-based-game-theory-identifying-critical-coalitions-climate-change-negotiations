"""
Greedy forward-selection of minor player coalitions with CLI-configurable size range.

Usage:
    python generate_greedy_coalitions.py --min-size 2 --max-size 4
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import zipfile
import os
import argparse


DEFAULT_MIN_COALITION_SIZE = 2
DEFAULT_MAX_COALITION_SIZE = 4
# -------------------------------
# CLI Arguments
# -------------------------------
parser = argparse.ArgumentParser(description="Greedy Coalition Generator")
parser.add_argument("--min-size", type=int, default=DEFAULT_MIN_COALITION_SIZE, help=f"Minimum coalition size (defaults to {DEFAULT_MIN_COALITION_SIZE})")
parser.add_argument("--max-size", type=int, default=DEFAULT_MAX_COALITION_SIZE, help=f"Maximum coalition size (defaults to {DEFAULT_MAX_COALITION_SIZE})")
args = parser.parse_args()

MIN_COALITION_SIZE = args.min_size
MAX_COALITION_SIZE = args.max_size

# -------------------------------
# Load Data
# -------------------------------
zip_path = "ratification data.zip"
extract_path = "iea_ratification_data"

if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

excel_path = os.path.join(extract_path, "ratification data", "Treaty data.xlsx")
ratification_df = pd.ExcelFile(excel_path).parse("Data")
ratification_matrix = ratification_df.pivot_table(
    index="Country", columns="Name", values="ratified", aggfunc="max", fill_value=0
)

cluster_df = pd.read_csv("table_countries_by_cluster.csv")
g20_members = {
    "Argentina", "Australia", "Brazil", "Canada", "China", "France", "Germany",
    "India", "Indonesia", "Italy", "Japan", "Mexico", "Russia", "Saudi Arabia",
    "South Africa", "South Korea", "Turkey", "United Kingdom", "United States",
    "European Union"
}
cluster_df["IsMinor"] = ~cluster_df["Country"].isin(g20_members)
minor_df = cluster_df[cluster_df["IsMinor"]].copy()
minor_players = minor_df["Country"].tolist()
cluster_map = minor_df.set_index("Country")["ClusterLabel"].to_dict()

# -------------------------------
# Precompute Similarity
# -------------------------------
similarity_matrix = cosine_similarity(ratification_matrix)
country_idx = {c: i for i, c in enumerate(ratification_matrix.index)}
pairwise_sim = {
    (a, b): similarity_matrix[country_idx[a], country_idx[b]]
    for a in minor_players for b in minor_players if a != b
}
global_sim = {
    c: np.sum(similarity_matrix[:, country_idx[c]])
    for c in minor_players
}

# -------------------------------
# Scoring and Validation
# -------------------------------
def fast_score(coalition):
    if len(coalition) < 2:
        return 0.0
    internal = [pairwise_sim[a, b] for i, a in enumerate(coalition)
                for b in coalition[i+1:] if (a, b) in pairwise_sim]
    alignment = np.mean(internal) if internal else 0.0
    spread = sum(global_sim[c] for c in coalition)
    diversity = len(set(cluster_map[c] for c in coalition))
    return alignment * spread * diversity

def is_valid(coalition):
    clusters = [cluster_map[c] for c in coalition]
    return max(Counter(clusters).values()) <= len(coalition) // 2

# -------------------------------
# Greedy Coalition Builder
# -------------------------------
results = []

for starter in minor_players:
    candidate_coalition = [starter]
    available = [p for p in minor_players if p != starter]

    while len(candidate_coalition) < MAX_COALITION_SIZE:
        best_gain = 0.0
        best_candidate = None
        current_score = fast_score(candidate_coalition)

        for candidate in available:
            test_coalition = candidate_coalition + [candidate]
            if len(test_coalition) < MIN_COALITION_SIZE:
                continue
            if not is_valid(test_coalition):
                continue
            gain = fast_score(test_coalition) - current_score
            if gain > best_gain:
                best_gain = gain
                best_candidate = candidate

        if best_candidate:
            candidate_coalition.append(best_candidate)
            available.remove(best_candidate)
        else:
            break

    if len(candidate_coalition) >= MIN_COALITION_SIZE:
        results.append((fast_score(candidate_coalition), candidate_coalition))

# -------------------------------
# Output Results
# -------------------------------
results.sort(reverse=True)
df = pd.DataFrame(results, columns=["Score", "Coalition"])
out_file = f"greedy_coalitions_min{MIN_COALITION_SIZE}_max{MAX_COALITION_SIZE}.csv"
df.to_csv(out_file, index=False)
print(f"✅ Saved: {out_file}")
