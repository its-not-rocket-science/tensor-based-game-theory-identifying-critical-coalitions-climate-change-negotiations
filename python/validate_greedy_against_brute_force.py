"""
Validate greedy-selected coalitions against a brute-force reference set.
Exports matched score comparisons to CSV and generates scatter plots.
"""

import ast
import pandas as pd
import matplotlib.pyplot as plt


# Canonical name mappings for normalising country names across datasets
canonical_names = {
    "brunei darussalam": "brunei",
    "czech republic": "czechia",
    "cape verde": "cabo verde",
    "congo, democratic republic of": "democratic republic of the congo",
    "congo, republic of the": "republic of the congo",
    "cote d'ivoire": "côte d’ivoire",
    "cote d’ivoire": "côte d’ivoire",
    "syrian arab republic": "syria",
    "syrian arab rep.": "syria",
    "iran, islamic republic of": "iran",
    "iran (islamic republic of)": "iran",
    "lao people's democratic republic": "laos",
    "korea, democratic people's republic": "north korea",
    "democratic people's republic of korea": "north korea",
    "korea, republic of": "south korea",
    "republic of korea": "south korea",
    "viet nam": "vietnam",
    "micronesia, federated states of": "micronesia",
    "moldova, republic of": "moldova",
    "palestinian territory, occupied": "palestine",
    "russian federation": "russia",
    "swaziland": "eswatini",
    "united states of america": "united states",
    "united kingdom of great britain and northern ireland": "united kingdom",
    "the bahamas": "bahamas",
    "the gambia": "gambia",
    "bahamas, the": "bahamas",
    "gambia, the": "gambia",
    "bolivia, plurinational state of": "bolivia",
    "venezuela, bolivarian republic of": "venezuela",
    "venezuela": "venezuela",
    "myanmar": "burma",
    "holy see (vatican city state)": "vatican city",
    "timor-leste": "east timor",
    "north macedonia": "macedonia",
    "slovak republic": "slovakia",
    "tanzania, united republic of": "tanzania",
    "são tomé and príncipe": "sao tome and principe"
}

# Enhanced normalisation function
def normalise_name(name):
    """
    Normalise a country name for standardised coalition matching.
    Converts to lowercase, strips punctuation and articles, applies canonical mapping.
    """
    name = name.lower().strip().replace("the ", "").replace(",", "").replace(".", "")
    return canonical_names.get(name, name)



# -------------------------------
# File paths
# -------------------------------

REFERENCE_FILE = "outputs/minor_tipping_coalitions_2_to_4_members.csv"
greedy_files = [
    ("outputs/TEST_greedy_coalitions_max_size_4.csv", "Test_Data"),
    # ("outputs/greedy_fast_minor_tipping_coalitions.csv", "Greedy_Current"),
    # ("outputs/first-run greedy_fast_minor_tipping_coalitions.csv", "Greedy_FirstRun")
]

# -------------------------------
# Load and standardise greedy datasets
# -------------------------------

greedy_dfs = {}
for path, label in greedy_files:
    df = pd.read_csv(path)
    # df["Coalition"] = df["Coalition"].apply(lambda x: tuple(sorted(ast.literal_eval(x))))
    df["Coalition"] = df["Coalition"].apply(
        lambda x: tuple(sorted(normalise_name(c) for c in ast.literal_eval(x)))
    )
    greedy_dfs[label] = df.set_index("Coalition")

# -------------------------------
# Load reference in chunks
# -------------------------------

reference_dict = {}
CHUNKSIZE = 100_000
processed = 0
print("⏳ Indexing reference file in chunks...")

for chunk in pd.read_csv(REFERENCE_FILE, chunksize=CHUNKSIZE):
    # chunk["Coalition"] = chunk["Coalition"].apply(lambda x: tuple(sorted(ast.literal_eval(x))))
    chunk["Coalition"] = chunk["Coalition"].apply(
        lambda x: tuple(sorted(normalise_name(c) for c in ast.literal_eval(x)))
    )

    for row in chunk.itertuples(index=False):
        reference_dict[row.Coalition] = row.Score

    processed += len(chunk)
    if processed % 500_000 == 0:
        print(f"Indexed {processed:,} rows so far...")

print(f"✅ Reference index built: {len(reference_dict)} coalitions")

for path, label in greedy_files:
    print("🔎 Greedy example:", list(greedy_dfs[label].index[:10]))
print("🔎 Reference example:", list(reference_dict.keys())[:10])

# -------------------------------
# Validate and Export Matches
# -------------------------------

for label, df in greedy_dfs.items():
    print(f"\n🔍 Validating: {label}")
    match_scores = []

    for coalition, row in df.iterrows():
        if coalition in reference_dict:
            match_scores.append({
                "Coalition": coalition,
                "GreedyScore": row.Score,
                "ReferenceScore": reference_dict[coalition],
                "ScoreError": row.Score - reference_dict[coalition]
            })

    match_df = pd.DataFrame(match_scores)

    if match_df.empty:
        print("⚠️ No matching coalitions found.")
        continue

    # Summary
    print(f"  Matches: {len(match_df)} / {len(df)}")
    print(match_df[["GreedyScore", "ReferenceScore", "ScoreError"]].describe())

    # Save CSV
    OUT_CSV = f"outputs/greedy_vs_reference_{label.lower()}.csv"
    match_df.to_csv(OUT_CSV, index=False)
    print(f"  📄 Comparison CSV saved to: {OUT_CSV}")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(match_df["ReferenceScore"], match_df["GreedyScore"],
                alpha=0.7, edgecolor='black', label=label)
    plt.plot([match_df["ReferenceScore"].min(), match_df["ReferenceScore"].max()],
             [match_df["ReferenceScore"].min(), match_df["ReferenceScore"].max()],
             color='black', linestyle='--', label='Perfect Match')
    plt.xlabel("Reference Score")
    plt.ylabel("Greedy Score")
    plt.title(f"{label.replace('_', ' ')} vs Reference Score")
    plt.legend()
    plt.tight_layout()
    PLOT_FILE = f"../{label.lower()}_score_plot.png"
    plt.savefig(PLOT_FILE)
    print(f"  📈 Score plot saved to: {PLOT_FILE}")
