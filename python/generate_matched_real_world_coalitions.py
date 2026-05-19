"""
Regenerate matched_real_world_coalitions.csv and the paper LaTeX summary table
using the corrected pipeline (n=175, influence_matrix spread).

Source coalitions: top greedy coalitions (corrected pipeline).
Membership lists: AOSIS, HAC, FF-NPT (key non-G20 members, 2023).

A coalition "matches" an initiative if at least one member appears in the
initiative's member list AND that member is in our 175-country minor set.
"""

import ast
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------
# Load corrected pipeline matrices
# -----------------------------------------------------------------------
cos_sim     = np.load("outputs/cosine_similarity.npy")
inf_mat     = np.load("outputs/influence_matrix.npy")
row_sums    = inf_mat.sum(axis=1)

minor_df    = pd.read_csv("outputs/minor_country_index.csv")
countries   = minor_df["Country"].tolist()
c2i         = {c: i for i, c in enumerate(countries)}

cluster_df  = pd.read_csv("outputs/table_countries_by_cluster.csv")
cluster_map = cluster_df.set_index("Country")["ClusterLabel"].to_dict()


def tipping_score(names):
    idx = [c2i[n] for n in names]
    k   = len(idx)
    sub_cos   = cos_sim[np.ix_(idx, idx)]
    alignment = (sub_cos.sum() - k) / (k * (k - 1))
    sub_inf   = inf_mat[np.ix_(idx, idx)]
    spread    = row_sums[idx].sum() - sub_inf.sum()
    diversity = len({cluster_map.get(countries[i], "") for i in idx})
    return float(alignment * spread * diversity)


# -----------------------------------------------------------------------
# Real-world initiative membership (non-G20 members only)
# AOSIS: Alliance of Small Island States (36 full members + observers)
# HAC:   High Ambition Coalition (key non-G20 members at COP21/COP26)
# FFNPT: Fossil Fuel Non-Proliferation Treaty initiative signatories
# -----------------------------------------------------------------------
AOSIS = {
    "Antigua and Barbuda", "Bahamas", "Barbados", "Belize", "Cape Verde",
    "Comoros", "Cook Islands", "Cuba", "Dominica", "Fiji", "Grenada",
    "Guinea-Bissau", "Guyana", "Haiti", "Jamaica", "Kiribati", "Maldives",
    "Marshall Islands", "Mauritius", "Nauru", "Niue", "Palau",
    "Papua New Guinea", "Samoa", "Sao Tome and Principe", "Seychelles",
    "Singapore", "Solomon Islands", "Saint Kitts and Nevis", "Saint Lucia",
    "Saint Vincent and the Grenadines", "Suriname", "Timor-Leste", "Tonga",
    "Trinidad and Tobago", "Tuvalu", "Vanuatu",
}

HAC = {
    # EU non-G20 member states
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czechia",
    "Denmark", "Estonia", "Finland", "Greece", "Hungary", "Ireland",
    "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland",
    "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden",
    # Pacific / SIDS co-leaders
    "Marshall Islands", "Tuvalu", "Palau", "Saint Lucia",
    # African / LAC members
    "Ethiopia", "Morocco", "Colombia", "Chile", "Costa Rica",
    "Peru", "Ecuador", "Panama", "Paraguay", "Uruguay",
    # Other notable members
    "Norway", "New Zealand", "Iceland", "Switzerland",
    "Bangladesh", "Nepal", "Afghanistan",
}

FFNPT = {
    "Tuvalu", "Vanuatu", "Fiji", "Timor-Leste", "Samoa", "Palau", "Tonga",
    "Niue", "Belize", "Solomon Islands", "Kiribati", "Marshall Islands",
    "Nauru", "Cook Islands", "Papua New Guinea", "Barbados",
}

# -----------------------------------------------------------------------
# Load corrected greedy coalitions
# -----------------------------------------------------------------------
greedy_df = pd.read_csv("outputs/greedy_fast_minor_tipping_coalitions.csv")
greedy_df["names"] = greedy_df["Coalition"].apply(ast.literal_eval)

# verify all countries are in minor index
def valid(names):
    return all(n in c2i for n in names)

greedy_df = greedy_df[greedy_df["names"].apply(valid)].copy()
print(f"Valid greedy coalitions: {len(greedy_df)}")

# -----------------------------------------------------------------------
# Match to initiatives
# -----------------------------------------------------------------------
def has_overlap(names, initiative_set):
    return bool(set(names) & initiative_set)

seen_keys = set()
records = []
for _, row in greedy_df.iterrows():
    names = row["names"]
    key   = tuple(sorted(names))
    if key in seen_keys:
        continue
    seen_keys.add(key)
    hac   = has_overlap(names, HAC)
    aosis = has_overlap(names, AOSIS)
    ffnpt = has_overlap(names, FFNPT)
    if hac or aosis or ffnpt:
        records.append({
            "Score":     tipping_score(names),
            "Coalition": str(sorted(names)),
            "HAC":       hac,
            "AOSIS":     aosis,
            "FFTreaty":  ffnpt,
        })

matched_df = pd.DataFrame(records).sort_values("Score", ascending=False)
matched_df.to_csv("outputs/matched_real_world_coalitions.csv", index=False)
print(f"Matched coalitions: {len(matched_df)}")
print(matched_df[["Score", "HAC", "AOSIS", "FFTreaty"]].head(10).to_string())

# -----------------------------------------------------------------------
# Build LaTeX summary table (top 14, one row per matched coalition)
# -----------------------------------------------------------------------
def initiative_label(row):
    labels = []
    if row["HAC"]:    labels.append("HAC")
    if row["AOSIS"]:  labels.append("AOSIS")
    if row["FFTreaty"]: labels.append("FF-NPT")
    return ", ".join(labels)

top = matched_df.head(14).copy()
top["Initiative"] = top.apply(initiative_label, axis=1)
top["Members"]    = top["Coalition"].apply(
    lambda x: ", ".join(ast.literal_eval(x))
)
top["Score_fmt"]  = top["Score"].apply(lambda x: f"{int(round(x)):,}")

tex_rows = []
for _, row in top.iterrows():
    members = row["Members"]
    if len(members) > 52:
        members = members[:52] + "\\ldots"
    tex_rows.append(
        f"{row['Initiative']} & {members} & {row['Score_fmt']} \\\\"
    )

tex = (
    "\\begin{tabular}{lll}\n"
    "\\toprule\n"
    "Initiative & Coalition Members & Tipping Score \\\\\n"
    "\\midrule\n"
    + "\n".join(tex_rows) + "\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
)

out_path = "../preprint paper/matched_real_world_coalitions_summary.tex"
with open(out_path, "w") as f:
    f.write(tex)
print(f"Saved: {out_path}")
