"""
Visualize Country Similarity in Environmental Treaty Participation

Generates:
1. Dendrogram of hierarchical clustering
2. Country similarity network (thresholded)
3. Cosine similarity heatmap

Input:
    - ratification_matrix: Country × Treaty binary matrix

Output:
    - figures/figure_country_dendrogram.png
    - figures/figure_country_similarity_network.png
    - figures/figure_country_similarity_heatmap.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx


# -------------------------------
# File Setup
# -------------------------------

# Adjust this path to wherever the Excel file is stored
excel_path = "ratification data/Treaty data.xlsx"

# Load Excel file
excel_file = pd.ExcelFile(excel_path)

# Read the main ratification data sheet
ratification_df = excel_file.parse('Data')

# -------------------------------
# Construct Country–Treaty Matrix
# -------------------------------

# Create a matrix where rows = countries, columns = treaties, values = 0/1 (ratified or not)
ratification_matrix = ratification_df.pivot_table(
    index='Country',
    columns='Name',
    values='ratified',
    aggfunc='max',
    fill_value=0
)

# -----------------------------
# Load Ratification Matrix
# -----------------------------

# Example placeholder: replace with actual loaded DataFrame
# from pandas.read_excel or pickle
# ratification_matrix = pd.read_pickle("data/ratification_matrix.pkl")

# Ensure output directory exists
os.makedirs("figures", exist_ok=True)

# -----------------------------
# Compute Similarity Matrix
# -----------------------------

similarity_matrix = cosine_similarity(ratification_matrix.values)
country_labels = ratification_matrix.index.tolist()

# -----------------------------
# 1. Dendrogram Visualization
# -----------------------------

# Hierarchical clustering
linked = linkage(ratification_matrix.values, method='ward')

plt.figure(figsize=(12, 8))
dendrogram(linked, labels=country_labels, leaf_rotation=90, leaf_font_size=8)
plt.title("Hierarchical Clustering of Countries by Treaty Participation")
plt.tight_layout()
plt.savefig("../figure_country_dendrogram.png", dpi=300)
plt.show()
plt.close()

# -----------------------------
# 2. Similarity Network Graph
# -----------------------------

G = nx.Graph()
threshold = 0.7  # Similarity threshold for edge creation

# Add edges for country pairs with similarity above threshold
for i in range(len(country_labels)):
    for j in range(i + 1, len(country_labels)):
        if similarity_matrix[i, j] >= threshold:
            G.add_edge(country_labels[i], country_labels[j],
                       weight=similarity_matrix[i, j])

# Graph layout and drawing
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=100)
nx.draw_networkx_edges(G, pos, alpha=0.3)
nx.draw_networkx_labels(G, pos, font_size=7)
plt.title("Country Similarity Network (Threshold ≥ 0.7)")
plt.tight_layout()
plt.savefig("../figure_country_similarity_network.png", dpi=300)
plt.show()
plt.close()

# -----------------------------
# 3. Heatmap of Similarity Matrix
# -----------------------------

plt.figure(figsize=(14, 12))
sns.heatmap(similarity_matrix,
            xticklabels=country_labels,
            yticklabels=country_labels,
            cmap='Blues', square=True,
            cbar_kws={'label': 'Cosine Similarity'})
plt.title("Heatmap of Country Similarity via Treaty Participation")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("../figure_country_similarity_heatmap.png", dpi=300)
plt.show()
plt.close()
