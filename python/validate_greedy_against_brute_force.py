"""
Validate scoring consistency: recompute scores for a sample from the exhaustive
reference using the same formula as the pipeline, then compare against stored values.

Approach: sample N coalitions from the reference file across all size groups,
recompute TippingScore independently, check agreement.

Outputs:
  outputs/greedy_vs_reference.csv   — sampled score comparisons
  ../test_data_score_plot.png        — scatter plot (recomputed vs reference score)
"""

import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REFERENCE_FILE = "outputs/minor_tipping_coalitions_2_to_4_members.csv"
SAMPLE_PER_SIZE = 500   # coalitions per size bucket to validate
TOTAL_ROWS = 38_646_125

# -----------------------------------------------------------------------
# Load pipeline matrices
# -----------------------------------------------------------------------
cos_sim = np.load("outputs/cosine_similarity.npy")
influence_matrix = np.load("outputs/influence_matrix.npy")
row_sums = influence_matrix.sum(axis=1)

minor_df = pd.read_csv("outputs/minor_country_index.csv")
countries = minor_df["Country"].tolist()
country_to_idx = {c: i for i, c in enumerate(countries)}

cluster_df = pd.read_csv("outputs/table_countries_by_cluster.csv")
cluster_map = cluster_df.set_index("Country")["ClusterLabel"].to_dict()

print(f"Loaded {len(countries)} minor players")


def recompute_score(names):
    indices = [country_to_idx[n] for n in names]
    k = len(indices)
    if k < 2:
        return 0.0
    c = indices
    sub_cos = cos_sim[np.ix_(c, c)]
    alignment = (sub_cos.sum() - k) / (k * (k - 1))
    sub_inf = influence_matrix[np.ix_(c, c)]
    spread = row_sums[c].sum() - sub_inf.sum()
    diversity = len({cluster_map.get(countries[i], "") for i in c})
    return float(alignment * spread * diversity)


# -----------------------------------------------------------------------
# Sample from reference in chunks — collect SAMPLE_PER_SIZE per size
# -----------------------------------------------------------------------
CHUNKSIZE = 200_000
sampled = {2: [], 3: [], 4: []}
quota = {2: SAMPLE_PER_SIZE, 3: SAMPLE_PER_SIZE, 4: SAMPLE_PER_SIZE}
processed = 0

print("Sampling reference file...")
for chunk in pd.read_csv(REFERENCE_FILE, chunksize=CHUNKSIZE):
    chunk["names"] = chunk["Coalition"].apply(ast.literal_eval)
    chunk["size"] = chunk["names"].apply(len)
    for sz in [2, 3, 4]:
        if quota[sz] <= 0:
            continue
        sub = chunk[chunk["size"] == sz]
        if sub.empty:
            continue
        take = sub.sample(min(quota[sz], len(sub)), random_state=42)
        sampled[sz].extend(zip(take["names"].tolist(), take["Score"].tolist()))
        quota[sz] -= len(take)
    processed += len(chunk)
    if processed % 5_000_000 == 0:
        print(f"  {processed:,} rows scanned...")
    if all(q <= 0 for q in quota.values()):
        print("  All quotas filled — stopping early")
        break

total_sampled = sum(len(v) for v in sampled.values())
print(f"Sampled {total_sampled} coalitions from reference")

# -----------------------------------------------------------------------
# Recompute scores and compare
# -----------------------------------------------------------------------
records = []
skipped = 0
for sz, items in sampled.items():
    for names, ref_score in items:
        if not all(n in country_to_idx for n in names):
            skipped += 1
            continue
        recomp = recompute_score(names)
        records.append({
            "Coalition": str(sorted(names)),
            "Size": sz,
            "RecomputedScore": recomp,
            "ReferenceScore": ref_score,
            "ScoreError": recomp - ref_score,
            "RelError": abs(recomp - ref_score) / (abs(ref_score) + 1e-12),
        })

if skipped:
    print(f"Skipped {skipped} coalitions with unknown country names")

result_df = pd.DataFrame(records)
print(f"\nValidation on {len(result_df)} coalitions:")
print(result_df[["RecomputedScore", "ReferenceScore", "ScoreError", "RelError"]].describe().round(6).to_string())

result_df.to_csv("outputs/greedy_vs_reference.csv", index=False)
print("\nSaved: outputs/greedy_vs_reference.csv")

max_rel_err = result_df["RelError"].max()
if max_rel_err < 1e-6:
    print(f"PASS: max relative error {max_rel_err:.2e} — scores are numerically identical")
else:
    print(f"WARN: max relative error {max_rel_err:.4f} — check for formula mismatch")

# -----------------------------------------------------------------------
# Scatter plot
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
for sz, color in zip([2, 3, 4], ["steelblue", "darkorange", "seagreen"]):
    sub = result_df[result_df["Size"] == sz]
    ax.scatter(sub["ReferenceScore"], sub["RecomputedScore"],
               alpha=0.6, edgecolor="none", s=20, label=f"size {sz}", color=color)
mn = min(result_df["ReferenceScore"].min(), result_df["RecomputedScore"].min())
mx = max(result_df["ReferenceScore"].max(), result_df["RecomputedScore"].max())
ax.plot([mn, mx], [mn, mx], "k--", lw=1, label="Perfect match")
ax.set_xlabel("Reference Score (brute-force)")
ax.set_ylabel("Recomputed Score (pipeline formula)")
ax.set_title("Score Consistency: Pipeline Formula vs. Brute-Force Reference")
ax.legend()
fig.tight_layout()
fig.savefig("../test_data_score_plot.png", dpi=150)
print("Saved: ../test_data_score_plot.png")
