"""
Cluster countries by treaty participation using UNSD subregions.
Outputs:
- figure_cluster_sizes_by_label_wrapped.png
- figure_cluster_similarity_network_named_wrapped_updated.png
- table_countries_by_cluster.csv
- cluster-level PCA, dendrogram, and heatmap
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import textwrap

# -------------------------------
# Load Data
# -------------------------------

excel_path = "ratification data/Treaty data.xlsx"
ratification_df = pd.read_excel(excel_path, sheet_name="Data")

# Use UNSD subregion for each country
country_subregion = (
    ratification_df.drop_duplicates("Country")
    .set_index("Country")["UNSD sub-Region"]
    .to_dict()
)

# -------------------------------
# Build Ratification Matrix
# -------------------------------

ratification_matrix = ratification_df.pivot_table(
    index="Country", columns="Name", values="ratified", aggfunc="max", fill_value=0
)
X = ratification_matrix.values
country_labels = ratification_matrix.index.tolist()

# -------------------------------
# Compute Similarity and Clustering
# -------------------------------

similarity = cosine_similarity(X)
Z = linkage(X, method="ward")
n_clusters = 8
cluster_labels = fcluster(Z, n_clusters, criterion="maxclust")

cluster_df = pd.DataFrame({
    "Country": country_labels,
    "Cluster": cluster_labels
}).sort_values(by="Cluster")

cluster_df["Subregion"] = cluster_df["Country"].map(country_subregion).fillna("Unknown")

# -------------------------------
# Generate Cluster Labels
# -------------------------------

def generate_compact_cluster_names(df):
    cluster_names = {}
    for cid, group in df.groupby("Cluster"):
        subregion_counts = Counter(group["Subregion"])
        top, top_count = subregion_counts.most_common(1)[0]
        total = len(group)
        other = total - top_count
        if other > 0:
            label = f"{top}({top_count}) + {other} more"
        else:
            label = f"{top}({top_count})"
        cluster_names[cid] = label
    return cluster_names

cluster_name_map = generate_compact_cluster_names(cluster_df)
cluster_df["ClusterLabel"] = cluster_df["Cluster"].map(cluster_name_map)

# -------------------------------
# Outputs
# -------------------------------

os.makedirs("outputs", exist_ok=True)
os.makedirs("figures/clusters", exist_ok=True)

# Table for appendix
cluster_df[["Country", "ClusterLabel"]].sort_values(by="ClusterLabel")\
    .to_csv("outputs/table_countries_by_cluster.csv", index=False)

# Bar chart with wrapped cluster names
cluster_counts = cluster_df["ClusterLabel"].value_counts().sort_values(ascending=True)
wrapped_labels = ['\n'.join(textwrap.wrap(label, width=25)) for label in cluster_counts.index]

plt.figure(figsize=(12, 6))
plt.barh(wrapped_labels, cluster_counts.values, color="skyblue", edgecolor="black")
plt.title("Number of Countries per Cluster (Descriptive Labels)")
plt.xlabel("Number of Countries")
plt.ylabel("Cluster Description")
plt.tight_layout()
plt.savefig("../figure_cluster_sizes_by_label_wrapped.png", dpi=300)
plt.close()

# -------------------------------
# Cluster Similarity Network
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

# Build graph
G = nx.Graph()
for i in range(n_clusters):
    G.add_node(cluster_name_map[i + 1])

for i in range(n_clusters):
    for j in range(i + 1, n_clusters):
        if cluster_similarity[i, j] >= 0.5:
            G.add_edge(cluster_name_map[i + 1], cluster_name_map[j + 1], weight=cluster_similarity[i, j])

wrapped_graph_labels = {
    node: '\n'.join(textwrap.wrap(node, width=14)) for node in G.nodes
}

plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_color='lightgreen', node_size=800)
nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.4)
nx.draw_networkx_labels(G, pos, labels=wrapped_graph_labels, font_size=9)
plt.title("Cluster-to-Cluster Similarity Network (Descriptive Labels)")
plt.tight_layout()
plt.savefig("../figure_pca_country_similarity.png", dpi=300)
plt.close()

# -------------------------------
# Cluster-Level Visuals
# -------------------------------

for cluster_id, countries_group in cluster_df.groupby("Cluster"):
    countries = countries_group["Country"]
    cluster_matrix = ratification_matrix.loc[countries]

    # PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(cluster_matrix)
    plt.figure(figsize=(8, 5))
    plt.scatter(coords[:, 0], coords[:, 1], c='cornflowerblue')
    for i, country in enumerate(countries):
        plt.text(coords[i, 0], coords[i, 1], country, fontsize=8)
    plt.title(f"PCA: {cluster_name_map[cluster_id]}")
    plt.tight_layout()
    plt.savefig(f"../cluster_{cluster_id}_pca.png", dpi=300)
    plt.close()

    # Dendrogram
    plt.figure(figsize=(10, 5))
    linked = linkage(cluster_matrix.values, method='ward')
    dendrogram(linked, labels=countries.tolist(), leaf_rotation=90, leaf_font_size=8)
    plt.title(f"Dendrogram: {cluster_name_map[cluster_id]}")
    plt.tight_layout()
    plt.savefig(f"../cluster_{cluster_id}_dendrogram.png", dpi=300)
    plt.close()

    # Cosine similarity heatmap
    sim = cosine_similarity(cluster_matrix)
    plt.figure(figsize=(8, 7))
    sns.heatmap(sim, xticklabels=countries, yticklabels=countries,
                cmap='Blues', square=True, cbar_kws={'label': 'Cosine Similarity'})
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title(f"Cosine Similarity Heatmap: {cluster_name_map[cluster_id]}")
    plt.tight_layout()
    plt.savefig(f"../cluster_{cluster_id}_heatmap.png", dpi=300)
    plt.close()
