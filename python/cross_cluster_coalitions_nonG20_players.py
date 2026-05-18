"""
Exhaustive tipping score search over all minor-player (non-G20) coalitions.

Uses corrected pipeline outputs — cosine similarity matrix and influence matrix
from process_ratification_data.py, country list from minor_country_index.csv.
This ensures consistent country resolution and G20 exclusion across all scripts.

Scores each coalition C as:
    TippingScore(C) = Alignment(C) * InfluenceSpread(C) * ClusterDiversity(C)

where:
    Alignment     = mean pairwise cosine similarity within C
    InfluenceSpread = sum of influence_matrix[i,j] for i in C, j not in C
    ClusterDiversity = number of distinct treaty-participation clusters in C

Runs coalition sizes 2, 3, 4. After each size completes, results are written to:
    outputs/minor_tipping_coalitions_2_to_{k}_members.csv

Run in background for k=4 (37.7M combinations).
"""

import itertools
import time
from math import comb
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------
# Load corrected pipeline outputs
# -----------------------------------------------------------------------

cos_sim = np.load("outputs/cosine_similarity.npy")
influence_matrix = np.load("outputs/influence_matrix.npy")

minor_df = pd.read_csv("outputs/minor_country_index.csv")
countries = minor_df["Country"].tolist()
N = len(countries)

cluster_df = pd.read_csv("outputs/table_countries_by_cluster.csv")
cluster_map = cluster_df.set_index("Country")["ClusterLabel"].to_dict()

# Integer cluster IDs for fast diversity computation
unique_clusters = sorted(set(cluster_map.values()))
cluster_id = {c: i for i, c in enumerate(unique_clusters)}
cluster_ids = np.array([cluster_id.get(cluster_map.get(countries[i], ""), 0)
                        for i in range(N)], dtype=np.int32)

# Precompute row sums for fast spread: spread = row_sums[coalition].sum() - sub_inf.sum()
row_sums = influence_matrix.sum(axis=1)

print(f"Pipeline: {N} minor players, {len(unique_clusters)} clusters")

# -----------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------

def score_coalition(c):
    k = len(c)
    sub_cos = cos_sim[np.ix_(c, c)]
    alignment = (sub_cos.sum() - k) / (k * (k - 1))
    sub_inf = influence_matrix[np.ix_(c, c)]
    spread = row_sums[c].sum() - sub_inf.sum()
    diversity = len(set(cluster_ids[i] for i in c))
    return alignment * spread * diversity

# -----------------------------------------------------------------------
# Search
# -----------------------------------------------------------------------

BATCH = 100_000
all_results = []
start = time.time()

for size in range(2, 5):
    n_combos = comb(N, size)
    combos = itertools.combinations(range(N), size)

    print(f"\nSize {size}: {n_combos:,} combinations")

    batch_results = []
    checked = 0

    for combo in combos:
        s = score_coalition(list(combo))
        c_names = tuple(countries[i] for i in combo)
        batch_results.append((s, c_names))
        checked += 1

        if checked % BATCH == 0:
            elapsed = time.time() - start
            rate = checked / elapsed
            remaining = (n_combos - checked) / rate
            print(f"  {checked:,}/{n_combos:,}  "
                  f"{elapsed:.0f}s elapsed  ~{remaining:.0f}s remaining", flush=True)

    all_results.extend(batch_results)

    # Write cumulative results after each size
    df = pd.DataFrame(all_results, columns=["Score", "Coalition"])
    out = f"outputs/minor_tipping_coalitions_2_to_{size}_members.csv"
    df.to_csv(out, index=False)
    elapsed = time.time() - start
    print(f"  Saved {len(all_results):,} rows -> {out}  ({elapsed:.1f}s total)")

print(f"\nDone. {len(all_results):,} coalitions in {time.time() - start:.1f}s")
