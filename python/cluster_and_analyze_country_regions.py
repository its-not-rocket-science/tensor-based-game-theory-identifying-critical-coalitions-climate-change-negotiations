"""
Cluster Countries by Environmental Treaty Participation and Analyze Regional Patterns

This script:
- Loads IEA ratification data
- Builds a country–treaty ratification matrix
- Computes cosine similarity between countries
- Performs hierarchical clustering
- Maps countries to regions using pycountry + pycountry_convert
- Summarizes region distributions by cluster
- Plots a cluster–cluster similarity graph

Outputs:
- figures/figure_cluster_similarity_network.png
- outputs/cluster_region_summary.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
import pycountry
import pycountry_convert as pc


# -------------------------------
# File Paths
# -------------------------------

input_excel_path = "ratification data/Treaty data.xlsx"
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)


# -------------------------------
# Load and Prepare Data
# -------------------------------

excel_file = pd.ExcelFile(input_excel_path)
ratification_df = excel_file.parse("Data")

# Create binary Country × Treaty matrix
ratification_matrix = ratification_df.pivot_table(
    index="Country",
    columns="Name",
    values="ratified",
    aggfunc="max",
    fill_value=0
)


# -------------------------------
# Compute Cosine Similarity Between Countries
# -------------------------------

X = ratification_matrix.values
country_labels = ratification_matrix.index.tolist()
similarity = cosine_similarity(X)


# -------------------------------
# Hierarchical Clustering
# -------------------------------

Z = linkage(X, method="ward")
n_clusters = 8
cluster_labels = fcluster(Z, n_clusters, criterion="maxclust")

cluster_df = pd.DataFrame({
    "Country": country_labels,
    "Cluster": cluster_labels
}).sort_values(by="Cluster")


# -------------------------------
# Map Countries to Continents
# -------------------------------

def country_to_region(country_name):
    """Map country to continent using ISO code lookup."""
    try:
        country = pycountry.countries.lookup(country_name)
        alpha2 = country.alpha_2
        continent_code = pc.country_alpha2_to_continent_code(alpha2)
        continent = pc.convert_continent_code_to_continent_name(continent_code)
        return continent
    except (LookupError, KeyError):
        return "Unknown"


# Apply region mapping
cluster_df["Region"] = cluster_df["Country"].apply(country_to_region)


# -------------------------------
# Region Distribution Summary
# -------------------------------

region_distribution = cluster_df.groupby(
    ["Cluster", "Region"]).size().unstack(fill_value=0)
region_distribution.to_csv("outputs/cluster_region_summary.csv")

print("✅ Region distribution by cluster saved to: outputs/cluster_region_summary.csv")


# -------------------------------
# Compute Cluster–Cluster Similarity Matrix
# -------------------------------

cluster_matrix = np.zeros((n_clusters, n_clusters))
counts = np.zeros((n_clusters, n_clusters))

for i in range(len(cluster_labels)):
    for j in range(len(cluster_labels)):
        ci = cluster_labels[i] - 1
        cj = cluster_labels[j] - 1
        cluster_matrix[ci, cj] += similarity[i, j]
        counts[ci, cj] += 1

with np.errstate(divide="ignore", invalid="ignore"):
    cluster_similarity = np.true_divide(cluster_matrix, counts)
    cluster_similarity[~np.isfinite(cluster_similarity)] = 0


# -------------------------------
# Plot Cluster-to-Cluster Graph
# -------------------------------

G = nx.Graph()
for i in range(n_clusters):
    G.add_node(f"Cluster {i+1}")

# Add edges for high-similarity cluster pairs
for i in range(n_clusters):
    for j in range(i + 1, n_clusters):
        weight = cluster_similarity[i, j]
        if weight >= 0.5:
            G.add_edge(f"Cluster {i+1}", f"Cluster {j+1}", weight=weight)

# Draw network
plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_color="lightgreen", node_size=800)
nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.4)
nx.draw_networkx_labels(G, pos, font_size=10)
plt.title("Cluster-to-Cluster Similarity Network (Cosine ≥ 0.5)")
plt.tight_layout()
plt.savefig("../figure_cluster_similarity_network.png")
plt.show()
plt.close()

print("✅ Cluster similarity figure saved to: ../figure_cluster_similarity_network.png")
