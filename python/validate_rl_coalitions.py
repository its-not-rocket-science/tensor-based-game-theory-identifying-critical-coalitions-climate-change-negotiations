"""
Validator for RL-learned coalitions with vulnerability weighting.

Checks:
- Only non-G20 countries appear
- Every coalition has >= 2 members
- Every country has a vulnerability score
"""

import pandas as pd

# -------------------------------
# Configuration
# -------------------------------

coalitions_path = "outputs/rl_learned_coalitions_realdata_with_sim_vulnerability.csv"
vulnerability_path = "outputs/simulated_vulnerability_scores.csv"

# Define G20 members
g20_members = {
    "Argentina", "Australia", "Brazil", "Canada", "China", "France", "Germany",
    "India", "Indonesia", "Italy", "Japan", "Mexico", "Russia", "Saudi Arabia",
    "South Africa", "South Korea", "Turkey", "United Kingdom", "United States",
    "European Union"
}

# -------------------------------
# Load data
# -------------------------------

coalitions_df = pd.read_csv(coalitions_path)
vul_df = pd.read_csv(vulnerability_path)

# Convert Coalition strings back into Python lists
coalitions_df["Coalition"] = coalitions_df["Coalition"].apply(eval)

# Map vulnerability scores
vul_map = dict(zip(vul_df["Country"], vul_df["VulnerabilityScore"]))

# -------------------------------
# Validation Checks
# -------------------------------

issues = []

for idx, row in coalitions_df.iterrows():
    coalition = row["Coalition"]

    # Check minimum size
    if len(coalition) < 2:
        issues.append(f"Row {idx}: Coalition too small: {coalition}")

    for country in coalition:
        # Check not G20
        if country in g20_members:
            issues.append(f"Row {idx}: G20 country included: {country}")
        # Check vulnerability score
        if country not in vul_map:
            issues.append(f"Row {idx}: Missing vulnerability score for {country}")

# -------------------------------
# Results
# -------------------------------

if issues:
    print("⚠️ Issues found:")
    for issue in issues:
        print("-", issue)
else:
    print("✅ All coalitions validated successfully!")
