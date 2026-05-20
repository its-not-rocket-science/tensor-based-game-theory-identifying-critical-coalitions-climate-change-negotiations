# Tipping the Climate Equilibrium: Tensor-Based Game Theory for Identifying Critical Coalitions in Climate Policy Negotiations

**Paul Schleifer** — Independent Researcher, London, United Kingdom  
Preprint: arXiv (econ.GN / cs.GT)  
Submitted to: *Environmental and Resource Economics*

---

## Overview

This repository contains the paper, all analysis code, input data, and pre-computed output data for a study of small-coalition tipping dynamics in multilateral climate negotiations. The central contribution is a third-order tensor model of strategic interaction in which each country's cooperative strategy is updated by the joint influence of pairs of other actors. A tractable coalition scoring rule (TippingScore) is derived from the dominant Z-eigenvalue of this tensor, and a REINFORCE policy gradient agent with Gumbel top-K sampling discovers high-scoring coalitions from a space of 175 non-G20 nations and 263 multilateral environmental agreements (1950–2017). The agent achieves a 14.7% improvement over random search. Keystone actors identified include Uruguay, Tunisia, Greece, and Algeria.

---

## Repository structure

```
.
├── preprint paper/
│   ├── main.tex                           # primary LaTeX source
│   ├── main.pdf                           # compiled paper PDF
│   ├── main_anon.tex                      # double-anonymous review version (auto-generated)
│   ├── main_anon.pdf                      # compiled anonymous PDF
│   ├── title_page.tex                     # Statements and Declarations title page
│   ├── references.bib                     # BibTeX bibliography (27 entries)
│   ├── make_anon.py                       # generates main_anon.tex from main.tex
│   ├── matched_real_world_coalitions_summary.tex   # \input{} fragment: Table 5
│   ├── rl_experiment_summary_table.tex             # \input{} fragment: Table 4
│   ├── figures/                           # all 14 paper figures (PNG)
│   ├── arxiv_submission.zip               # arXiv-ready source bundle
│   └── cover_letter.txt                   # ERE submission cover letter
│
├── python/
│   ├── country_name_resolver.py           # ISO 3166-1 resolver module (shared utility)
│   ├── process_ratification_data.py       # Step 1: treaty matrix → cosine similarity + influence matrix
│   ├── build_real_vulnerability_data.py   # Step 2: merge ND-GAIN + Aqueduct + SIDS → vulnerability scores
│   ├── cluster_and_analyze_with_descriptive_labels.py  # Step 3: cluster countries → table_countries_by_cluster.csv
│   ├── cluster_and_analyze_country_regions.py          # Step 3 (alt): cluster + UNSD region mapping
│   ├── cross_cluster_coalitions_nonG20_players.py      # Step 4: exhaustive search k=2,3,4
│   ├── generate_greedy_coalitions.py      # Step 5: greedy forward selection
│   ├── validate_greedy_against_brute_force.py          # Step 6: scoring consistency validation
│   ├── generate_matched_real_world_coalitions.py       # Step 7: match to AOSIS/HAC/FF-NPT
│   ├── run_rl_sensitivity_sweep.py        # Step 8: RL parameter sweep (3×2×3 grid)
│   ├── rl_train_realdata_strategic_rewards.py          # Step 9: RL with composite vulnerability reward
│   ├── rl_train_realdata_strategic_rewards_cli.py      # Step 9 (CLI): same, with command-line arguments
│   ├── rl_train_realdata.py               # Step 9 (alt): RL without vulnerability weighting
│   ├── validate_rl_coalitions.py          # Step 10: validate RL coalitions (non-G20, coverage, scoring)
│   ├── generate_simulated_vulnerability_scores.py      # (optional) generate random vulnerability scores
│   ├── tensor_game_simulation.py          # Figure 1: small-system dynamics
│   ├── influence_vs_coalition_size_experiment.py       # Figure 2: coalition size sweep
│   ├── coalition_efficiency_plot.py       # Figure 3: efficiency (eigenvalue gain per member)
│   ├── figure_power_centrality.py         # Figure 4: Shapley/Banzhaf power indices + centrality
│   ├── greedy_tipping_coalition.py        # Figure 5: greedy tipping illustration
│   ├── generate_rl_training_summary.py    # Figure 6: RL training curves
│   ├── country_treaty_similarity_pca.py   # Figure 7: PCA of treaty participation
│   ├── histogram_tipping_scores_small_coalitions.py    # Figure 8: score distribution (k=2,3)
│   ├── analyse_greedy_coalitions.py       # Figures 9–10: greedy score distribution + network
│   ├── reinforcement_learning_based_optimisation.py    # Figures 11–12: RL sweep summary plots
│   ├── inputs/
│   │   ├── all.csv                        # country list with ISO3 codes
│   │   ├── nd_gain_country_index.csv      # ND-GAIN Country Index (2022 scores)
│   │   ├── real_vulnerability_scores.csv  # pre-merged vulnerability scores (committed)
│   │   └── small_island_developing_states.csv  # SIDS membership list
│   ├── outputs/                           # pipeline outputs (all committed; large files via Git LFS)
│   │   ├── cosine_similarity.npy          # 175×175 cosine similarity matrix
│   │   ├── influence_matrix.npy           # 175×175 co-ratification influence matrix
│   │   ├── minor_country_index.csv        # country row-order index for matrices
│   │   ├── table_countries_by_cluster.csv # country → cluster label mapping
│   │   ├── cluster_region_summary.csv     # cluster × UNSD region distribution
│   │   ├── greedy_fast_minor_tipping_coalitions.csv    # greedy-discovered coalitions
│   │   ├── greedy_vs_reference.csv        # brute-force validation comparison
│   │   ├── minor_tipping_coalitions_2_to_2_members.csv.gz  # all 2-member coalitions
│   │   ├── minor_tipping_coalitions_2_to_3_members.csv.gz  # all 3-member coalitions
│   │   ├── minor_tipping_coalitions_2_to_4_members.csv.gz  # all 4-member coalitions (~37.7M rows)
│   │   ├── matched_real_world_coalitions.csv               # AOSIS/HAC/FF-NPT overlap
│   │   ├── rl_learned_coalitions_realdata.csv              # RL coalitions (no vulnerability weighting)
│   │   ├── rl_learned_coalitions_realdata_with_sim_vulnerability.csv
│   │   ├── rl_experiment_summary.csv      # RL sensitivity sweep results
│   │   ├── simulated_vulnerability_scores.csv
│   │   └── table_countries_by_cluster.csv
│   ├── ratification data/
│   │   ├── Treaty data.xlsx               # IEA treaty ratification matrix (Country × Treaty)
│   │   └── README.pdf                     # data source documentation
│   └── figures/                           # figures generated by Python scripts
│
└── outputs/                               # root-level output copies (legacy; mirrors python/outputs/)
```

---

## Environment setup

### Python

Tested with **Python 3.12.5**. All dependencies are available via pip.

```bash
pip install numpy==1.26.4 pandas scipy scikit-learn matplotlib torch networkx openpyxl
```

Or pin exact versions used:

```
numpy==1.26.4
pandas>=2.0
scipy>=1.13
scikit-learn>=1.6
matplotlib>=3.10
torch>=2.6
networkx>=3.6
openpyxl>=3.1
```

No GPU is required. PyTorch CPU build (`torch>=2.6+cpu`) is sufficient.

All scripts are run from the `python/` directory unless otherwise noted.

### LaTeX

Tested with MiKTeX on Windows; TeX Live on Linux/macOS should work identically. Required packages: `amsmath`, `amssymb`, `graphicx`, `hyperref`, `geometry`, `authblk`, `natbib`, `setspace`, `float`, `booktabs`. All are included in standard MiKTeX/TeX Live distributions.

---

## Data sources

| Dataset | File in repo | Licence / access |
|---------|-------------|-----------------|
| IEA International Environmental Agreements Dataset (Bellelli & Bernauer 2021) | `python/ratification data/Treaty data.xlsx` | Publicly available at https://iea.ucl.ac.uk/data |
| ND-GAIN Country Index (2022) | `python/inputs/nd_gain_country_index.csv` | CC BY 4.0 · https://gain.nd.edu/our-work/country-index/download/ |
| WRI Aqueduct 4.0 (Kuzma et al. 2023) | **must download** — see below | CC BY 4.0 · https://www.wri.org/research/aqueduct-40-updated-decision-relevant-global-water-risk-indicators |
| UN SIDS membership list | `python/inputs/small_island_developing_states.csv` | UN public domain |

### Downloading WRI Aqueduct 4.0

The Aqueduct rankings file is excluded from the repository due to its size. Download it manually:

1. Visit https://www.wri.org/research/aqueduct-40-updated-decision-relevant-global-water-risk-indicators
2. Download the **Country Rankings** Excel file (filename: `Aqueduct40_rankings_download_Y2023M07D05.xlsx`)
3. Place it at `python/inputs/Aqueduct40_rankings_download_Y2023M07D05.xlsx`

The pre-merged vulnerability scores (`python/inputs/real_vulnerability_scores.csv`) are committed to the repository; if you only want to reproduce coalition search results without re-merging vulnerability data, you can skip this download.

---

## Reproducing all results

All scripts are run from the `python/` directory:

```bash
cd python/
```

### Step 1 — Build the ratification matrices

```bash
python process_ratification_data.py
```

**Inputs:** `ratification data/Treaty data.xlsx`, `inputs/all.csv`  
**Outputs:** `outputs/cosine_similarity.npy`, `outputs/influence_matrix.npy`, `outputs/minor_country_index.csv`  
**Note:** Applies ISO 3166-1 deduplication and G20 exclusion, resulting in N=175 minor countries.

### Step 2 — Build vulnerability scores

```bash
python build_real_vulnerability_data.py
```

**Inputs:** `inputs/nd_gain_country_index.csv`, `inputs/Aqueduct40_rankings_download_Y2023M07D05.xlsx`, `inputs/small_island_developing_states.csv`  
**Output:** `inputs/real_vulnerability_scores.csv`  
**Requires:** Aqueduct file downloaded (see above). Skip if using the committed pre-merged file.

### Step 3 — Cluster countries by treaty participation

```bash
python cluster_and_analyze_with_descriptive_labels.py
```

**Inputs:** `ratification data/Treaty data.xlsx`, `outputs/cosine_similarity.npy`  
**Outputs:** `outputs/table_countries_by_cluster.csv`, `outputs/cluster_region_summary.csv`, `figures/figure_cluster_sizes_by_label_wrapped.png`, `figures/figure_cluster_similarity_network_named_wrapped_updated.png`, `figures/clusters/cluster_{1–8}_{dendrogram,heatmap,pca}.png`  
**Expected clusters:** 8 treaty-participation communities.

### Step 4 — Exhaustive coalition search (k = 2, 3, 4)

```bash
python cross_cluster_coalitions_nonG20_players.py
```

**Inputs:** `outputs/cosine_similarity.npy`, `outputs/influence_matrix.npy`, `outputs/minor_country_index.csv`, `outputs/table_countries_by_cluster.csv`  
**Outputs:** `outputs/minor_tipping_coalitions_2_to_2_members.csv.gz`, `..._3_members.csv.gz`, `..._4_members.csv.gz`  
**Runtime:** k=2 and k=3 complete in minutes; k=4 (~37.7 million coalitions) may take several hours on a single CPU. Pre-computed results are committed to the repository.

### Step 5 — Greedy coalition search

```bash
python generate_greedy_coalitions.py --min-size 2 --max-size 6
```

**Inputs:** `outputs/cosine_similarity.npy`, `outputs/influence_matrix.npy`, `outputs/minor_country_index.csv`, `outputs/table_countries_by_cluster.csv`  
**Output:** `outputs/greedy_fast_minor_tipping_coalitions.csv`  
**Runtime:** Under 1 minute.

### Step 6 — Validate greedy scoring consistency

```bash
python validate_greedy_against_brute_force.py
```

**Inputs:** `outputs/minor_tipping_coalitions_2_to_4_members.csv.gz`, `outputs/cosine_similarity.npy`, `outputs/influence_matrix.npy`, `outputs/minor_country_index.csv`, `outputs/table_countries_by_cluster.csv`  
**Outputs:** `outputs/greedy_vs_reference.csv`, `figures/test_data_score_plot.png`

### Step 7 — Match to real-world coalitions

```bash
python generate_matched_real_world_coalitions.py
```

**Inputs:** `outputs/greedy_fast_minor_tipping_coalitions.csv`, `outputs/minor_country_index.csv`  
**Outputs:** `outputs/matched_real_world_coalitions.csv`, `../preprint paper/matched_real_world_coalitions_summary.tex`  
**Coalitions matched against:** AOSIS, High Ambition Coalition (HAC), Fossil Fuel Non-Proliferation Treaty Initiative.

### Step 8 — RL sensitivity sweep

```bash
python run_rl_sensitivity_sweep.py
```

**Inputs:** `outputs/cosine_similarity.npy`, `outputs/influence_matrix.npy`, `outputs/minor_country_index.csv`, `outputs/table_countries_by_cluster.csv`, `inputs/real_vulnerability_scores.csv`  
**Outputs:** `outputs/rl_experiment_summary.csv`, `../preprint paper/rl_experiment_summary_table.tex`, `figures/rl_experiment_summary_plot.png`, `figures/rl_experiment_summary_plot_sizes.png`  
**Parameters swept:** size_cap ∈ {4, 6, 8} × penalty_alpha ∈ {0.0, 1.0} × reward_threshold ∈ {0, 2000, 5000} (18 configurations × 500 epochs each).  
**Runtime:** ~20–40 minutes on CPU.

### Step 9 — RL training with composite vulnerability reward (primary result)

```bash
python rl_train_realdata_strategic_rewards.py
```

Or using the CLI with explicit arguments:

```bash
python rl_train_realdata_strategic_rewards_cli.py \
    --epochs 1000 \
    --size-cap 8 \
    --size-penalty 1.0 \
    --vulnerability-csv inputs/real_vulnerability_scores.csv
```

**Inputs:** `outputs/cosine_similarity.npy`, `outputs/influence_matrix.npy`, `outputs/minor_country_index.csv`, `outputs/table_countries_by_cluster.csv`, `inputs/real_vulnerability_scores.csv`  
**Outputs:** `outputs/rl_learned_coalitions_realdata.csv`, `figures/rl_realdata_training_summary.png`  
**Primary result:** peak tipping score 22,229 achieved after 1,000 epochs; 14.7% improvement over random search.  
**Runtime:** ~5–10 minutes on CPU.

### Step 10 — Validate RL coalitions

```bash
python validate_rl_coalitions.py
```

**Inputs:** `outputs/rl_learned_coalitions_realdata_with_sim_vulnerability.csv`, `outputs/simulated_vulnerability_scores.csv`  
**Checks:** non-G20 membership, coalition size ≥ 2, vulnerability score presence for all members.

---

## Reproducing all paper figures

Pre-computed figures are committed to `preprint paper/figures/`. To regenerate from scratch, run the following scripts from `python/`. Copy outputs to `preprint paper/figures/` before recompiling LaTeX.

| Figure in paper | Generating script | Output path |
|----------------|------------------|-------------|
| Fig. 1 — small-system coalition effect | `tensor_game_simulation.py` | `figures/figure_coalition_effect.png` |
| Fig. 2 — coalition size sweep | `influence_vs_coalition_size_experiment.py` | `figures/coalition_sweep_plot.png` |
| Fig. 3 — coalition efficiency | `coalition_efficiency_plot.py` | `figures/coalition_efficiency_plot.png` |
| Fig. 4 — power indices and centrality | `figure_power_centrality.py` | `figures/figure_power_centrality.png` |
| Fig. 5 — greedy tipping illustration | `greedy_tipping_coalition.py` | `figures/figure_greedy_tipping.png` |
| Fig. 6 — RL training curves | `generate_rl_training_summary.py` | `figures/figure_rl_training_summary.png` |
| Fig. 7 — PCA of treaty participation | `country_treaty_similarity_pca.py` | `figures/figure_pca_country_similarity.png` |
| Fig. 8 — tipping score distribution (k=2,3) | `histogram_tipping_scores_small_coalitions.py` | `figures/score_distribution_2to3.png` |
| Fig. 9 — greedy score distribution | `analyse_greedy_coalitions.py` | `figures/greedy_score_distribution.png` |
| Fig. 10 — greedy coalition network | `analyse_greedy_coalitions.py` | `figures/greedy_minor_coalition_network.png` |
| Fig. 11 — RL sweep by reward strategy | `run_rl_sensitivity_sweep.py` | `figures/rl_experiment_summary_plot.png` |
| Fig. 12 — RL sweep by coalition size | `run_rl_sensitivity_sweep.py` | `figures/rl_experiment_summary_plot_sizes.png` |
| Fig. 13 — RL real-data training | `rl_train_realdata_strategic_rewards.py` | `figures/rl_realdata_training_summary.png` |
| Fig. A1 — greedy validation (appendix) | `validate_greedy_against_brute_force.py` | `figures/test_data_score_plot.png` |

---

## Compiling the paper

```bash
cd "preprint paper"
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

**Expected output:** `main.pdf` (approximately 37 pages).

To regenerate the anonymous version:

```bash
python make_anon.py
pdflatex main_anon.tex
bibtex main_anon
pdflatex main_anon.tex
pdflatex main_anon.tex
```

---

## Reproducibility checklist

### 175-country pipeline

- [ ] `python/ratification data/Treaty data.xlsx` present (IEA dataset)
- [ ] `process_ratification_data.py` run; `minor_country_index.csv` contains exactly 175 rows
- [ ] G20 countries excluded: verify `country_name_resolver.py::G20_ISO3` list covers all 20 G20 members
- [ ] Historical entities excluded: verify `HISTORICAL_ENTITIES` in `country_name_resolver.py` covers dissolved states in treaty data
- [ ] `cosine_similarity.npy` and `influence_matrix.npy` are both 175×175

### Vulnerability data joins

- [ ] `build_real_vulnerability_data.py` run (or pre-merged `real_vulnerability_scores.csv` used)
- [ ] `real_vulnerability_scores.csv` contains a row for each of the 175 countries
- [ ] Six micro-states lacking ND-GAIN coverage assigned composite score 0.0 (documented in paper §7)
- [ ] SIDS receive Sea Level Exposure = 1.0 (maximum); verify via `small_island_developing_states.csv` join

### Coalition scoring (TippingScore)

- [ ] Alignment computed as mean pairwise cosine similarity across coalition members
- [ ] InfluenceSpread computed as sum of off-diagonal influence matrix entries (coalition × non-coalition)
- [ ] ClusterDiversity computed as number of distinct cluster labels in coalition
- [ ] TippingScore = Alignment × InfluenceSpread × ClusterDiversity
- [ ] Formula consistent between `generate_greedy_coalitions.py`, `cross_cluster_coalitions_nonG20_players.py`, and `validate_greedy_against_brute_force.py` (validated in Step 6 above)

### Exhaustive coalition search

- [ ] k=2: all C(175,2) = 15,225 coalitions enumerated; result in `minor_tipping_coalitions_2_to_2_members.csv.gz`
- [ ] k=3: all C(175,3) = 881,275 coalitions; result in `minor_tipping_coalitions_2_to_3_members.csv.gz`
- [ ] k=4: all C(175,4) ≈ 37.7 million coalitions; result in `minor_tipping_coalitions_2_to_4_members.csv.gz`

### Greedy validation

- [ ] `validate_greedy_against_brute_force.py` produces `greedy_vs_reference.csv`
- [ ] All sampled coalitions show identical TippingScore under both implementations (zero discrepancy)

### RL experiments

- [ ] REINFORCE agent architecture: single linear layer, sigmoid activations, input dimension N=175
- [ ] Bias initialised to `logit(8/175) ≈ −3.0` to match size-cap prior
- [ ] Gumbel top-K sampling used (not Bernoulli); k drawn from Uniform{2,…,8} each step
- [ ] EMA baseline subtracted; mixing coefficient α = 0.05
- [ ] Training: 1,000 epochs; 5,000 post-training samples
- [ ] Peak score of 22,229 achieved; 4,964 unique coalitions from 5,000 samples
- [ ] RL improvement over random search: 22,229 / 19,382 − 1 ≈ 14.7%

### Bernoulli sampling ablation

Demonstrates why Gumbel top-K is necessary (paper §6.2).

- [ ] Run `rl_train_realdata_strategic_rewards.py` with Bernoulli sampling enabled (replace Gumbel top-K block with `torch.bernoulli(probs)` and filter to size-cap)
- [ ] Confirm 0 valid coalitions sampled from 5,000 post-training draws — expected coalition size ~N/2 = 87 always exceeds size cap of 8
- [ ] Result committed: `python/outputs/rl_learned_coalitions_realdata_with_sim_vulnerability.csv` contains header only (0 data rows)

---

## arXiv source checklist

The `preprint paper/arxiv_submission.zip` contains the full arXiv submission bundle. Contents:

| File | Role |
|------|------|
| `main.tex` | Primary LaTeX source |
| `main.bbl` | Compiled bibliography (included for arXiv compatibility) |
| `references.bib` | BibTeX source |
| `matched_real_world_coalitions_summary.tex` | `\input{}` fragment (Table 5) |
| `rl_experiment_summary_table.tex` | `\input{}` fragment (Table 4) |
| `figures/*.png` | All 14 figures referenced in main.tex |

Files **excluded** from the arXiv bundle (kept in the repository only):

| File | Reason |
|------|--------|
| `main_anon.tex` / `main_anon.pdf` | Double-anonymous review version; not for public posting |
| `make_anon.py` | Internal tooling |
| `title_page.tex` / `title_page.pdf` | Separate journal submission document |
| `cover_letter.txt` | Separate journal submission document |
| `*.aux`, `*.log`, `*.out`, `*.blg`, `*.bbl` (build artifacts) | Generated by compilation |

---

## Licence

Code and data in this repository are released under the MIT Licence — see [LICENSE](LICENSE).

The IEA International Environmental Agreements Dataset, ND-GAIN Country Index, and WRI Aqueduct 4.0 are subject to their respective licences (CC BY 4.0). Attribution is included in the paper's data availability statement.

---

## Citation

If you use this code or data, please cite the paper:

```
Schleifer, P. (2025). Tipping the Climate Equilibrium: Tensor-Based Game Theory for
Identifying Critical Coalitions in Climate Policy Negotiations.
arXiv preprint arXiv:[ID]. https://arxiv.org/abs/[ID]
```

BibTeX:

```bibtex
@article{schleifer2025tipping,
  author  = {Schleifer, Paul},
  title   = {Tipping the Climate Equilibrium: Tensor-Based Game Theory for
             Identifying Critical Coalitions in Climate Policy Negotiations},
  year    = {2025},
  journal = {arXiv preprint},
  note    = {arXiv:[ID] [econ.GN]}
}
```

Replace `[ID]` with the arXiv identifier once assigned.
