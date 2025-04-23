"""
Generate histogram of tipping scores for 2–3 member minor-player coalitions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load coalition score data
df = pd.read_csv("outputs/minor_tipping_coalitions_2_to_3_members.csv")
df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
df = df.dropna(subset=["Score"])

# Summary stats
scores = df["Score"]
mean_score = scores.mean()
std_score = scores.std()
threshold_95 = np.percentile(scores, 95)
threshold_99 = np.percentile(scores, 99)

# Plot score distribution
plt.figure(figsize=(10, 5))
plt.hist(scores, bins=30, color="skyblue", edgecolor="black")
plt.axvline(threshold_95, color='red', linestyle='--', label='95th percentile')
plt.axvline(threshold_99, color='darkred', linestyle='--', label='99th percentile')
plt.axvline(mean_score, color='black', linestyle='--', label='Mean')
plt.title("Tipping Score Distribution (2–3 Member Coalitions)")
plt.xlabel("Tipping Score")
plt.ylabel("Number of Coalitions")
plt.legend()
plt.tight_layout()

plt.savefig("figures/score_distribution_2to3.png")
