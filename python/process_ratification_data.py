import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load ratification matrix
excel_path = "ratification data/Treaty data.xlsx"
rat_df = pd.read_excel(excel_path, sheet_name="Data")
rat_matrix = rat_df.pivot_table(index="Country", columns="Name", values="ratified", aggfunc="max", fill_value=0)

# Filter to only minor countries
cluster_df = pd.read_csv("outputs/table_countries_by_cluster.csv")
g20_members = {
    "Argentina", "Australia", "Brazil", "Canada", "China", "France", "Germany",
    "India", "Indonesia", "Italy", "Japan", "Mexico", "Russia", "Saudi Arabia",
    "South Africa", "South Korea", "Turkey", "United Kingdom", "United States",
    "European Union"
}
minor_countries = cluster_df[~cluster_df["Country"].isin(g20_members)]["Country"].tolist()
rat_matrix = rat_matrix.loc[rat_matrix.index.isin(minor_countries)]

# Compute cosine similarity and influence matrix
cos_sim = cosine_similarity(rat_matrix.values)
influence_matrix = rat_matrix.values @ rat_matrix.values.T

# Save .npy files
np.save("outputs/cosine_similarity.npy", cos_sim)
np.save("outputs/influence_matrix.npy", influence_matrix)

print("✅ Saved: cosine_similarity.npy and influence_matrix.npy")
