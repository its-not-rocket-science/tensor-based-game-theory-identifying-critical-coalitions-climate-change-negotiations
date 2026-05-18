"""
Process treaty ratification data into cosine similarity and influence matrices.

Uses ISO3 codes as canonical join key so historical entities (USSR, Yugoslavia,
East Germany, etc.) are excluded and country name variants don't create
duplicate/mismatched rows.

Outputs:
- outputs/cosine_similarity.npy
- outputs/influence_matrix.npy
- outputs/minor_country_index.csv  (country list in matrix row order, used by RL scripts)
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from country_name_resolver import resolve_to_iso3, G20_ISO3

excel_path = "ratification data/Treaty data.xlsx"
rat_df = pd.read_excel(excel_path, sheet_name="Data")

# Resolve all country names to ISO3; rows that resolve to None are historical/invalid
rat_df["ISO3"] = rat_df["Country"].apply(resolve_to_iso3)
excluded = rat_df[rat_df["ISO3"].isna()]["Country"].unique().tolist()
if excluded:
    print(f"Excluding {len(excluded)} historical/unresolvable entities: {excluded}")
rat_df = rat_df.dropna(subset=["ISO3"])

# Pivot on ISO3 to avoid name-variant duplicates
rat_matrix = rat_df.pivot_table(
    index="ISO3", columns="Name", values="ratified", aggfunc="max", fill_value=0
)

# --- Filter to minor countries (non-G20, non-historical) ---
cluster_df = pd.read_csv("outputs/table_countries_by_cluster.csv")
cluster_df["ISO3"] = cluster_df["Country"].apply(resolve_to_iso3)
minor_df = cluster_df[
    cluster_df["ISO3"].notna() & ~cluster_df["ISO3"].isin(G20_ISO3)
].copy()
minor_iso3 = set(minor_df["ISO3"])

rat_matrix = rat_matrix.loc[rat_matrix.index.isin(minor_iso3)]

# --- Compute matrices ---
cos_sim = cosine_similarity(rat_matrix.values)
influence_matrix = rat_matrix.values @ rat_matrix.values.T

np.save("outputs/cosine_similarity.npy", cos_sim)
np.save("outputs/influence_matrix.npy", influence_matrix)

# --- Save country index in matrix row order for downstream RL scripts ---
iso3_to_treaty = minor_df.set_index("ISO3")["Country"].to_dict()
country_order = [iso3_to_treaty.get(iso3, iso3) for iso3 in rat_matrix.index]
pd.DataFrame({
    "ISO3": rat_matrix.index.tolist(),
    "Country": country_order
}).to_csv("outputs/minor_country_index.csv", index=False)

print(f"Saved cosine_similarity.npy, influence_matrix.npy, minor_country_index.csv ({len(rat_matrix)} countries)")
