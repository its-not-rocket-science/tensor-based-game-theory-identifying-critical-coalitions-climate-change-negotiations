"""
Analyse greedy-selected minor player coalitions:
- Plot score distribution
- Visualise coalition membership as a network
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from ast import literal_eval

# -------------------------------
# Load Greedy Search Results
# -------------------------------

# File produced by greedy forward-selection algorithm
df = pd.read_csv("outputs/greedy_fast_minor_tipping_coalitions.csv")
df["Score"] = pd.to_numeric(df["Score"], errors="coerce")

# -------------------------------
# Score Distribution Plot
# -------------------------------

scores = df["Score"].dropna()
mean_score = scores.mean()
std_score = scores.std()
threshold_95 = np.percentile(scores, 95)
threshold_99 = np.percentile(scores, 99)

# Plot histogram
plt.figure(figsize=(10, 5))
plt.hist(scores, bins=30, color='cornflowerblue', edgecolor='black')
plt.axvline(threshold_95, color='red', linestyle='--', label='95th percentile')
plt.axvline(threshold_99, color='darkred', linestyle='--', label='99th percentile')
plt.axvline(mean_score, color='black', linestyle='--', label='Mean')
plt.title("Tipping Score Distribution (Greedy Minor Coalitions)")
plt.xlabel("Tipping Score")
plt.ylabel("Number of Coalitions")
plt.legend()
plt.tight_layout()
plt.savefig("../greedy_score_distribution.png")
plt.close()

# -------------------------------
# Build Coalition Membership Network
# -------------------------------

# Initialise empty graph
G = nx.Graph()

# Add edges for co-membership
for _, row in df.iterrows():
    try:
        members = literal_eval(row["Coalition"])  # safely parse the list of members from CSV
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if G.has_edge(a, b):
                    G[a][b]['weight'] += 1
                else:
                    G.add_edge(a, b, weight=1)
    except ValueError as e:
        print(f"Skipping row due to parsing error: {e}")

# Draw network
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=300)
nx.draw_networkx_edges(G, pos, alpha=0.3)
nx.draw_networkx_labels(G, pos, font_size=8)
plt.title("Network of Minor Player Coalitions (Greedy Selected)")
plt.tight_layout()
plt.savefig("../greedy_minor_coalition_network.png")
plt.close()
