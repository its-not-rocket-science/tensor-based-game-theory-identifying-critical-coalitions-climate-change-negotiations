"""
Re-run RL reward strategy sensitivity sweep on corrected pipeline (n=175,
influence_matrix.npy as spread metric).

Sweep grid:
  size_cap:   4, 6, 8
  penalty_alpha: 0.0, 1.0   (score /= |C|^alpha)
  threshold:  0, 2000, 5000  (min reward; 0 = none)

Thresholds chosen relative to corrected pipeline:
  mean per-member score (3-member, alpha=1) ≈ 14190/3 ≈ 4730
  2000 ≈ 42nd percentile; 5000 ≈ median → both filter meaningfully

Outputs:
  outputs/rl_experiment_summary.csv
  outputs/rl_experiment_summary_table.tex
  ../preprint paper/rl_experiment_summary_plot.png
  ../preprint paper/rl_experiment_summary_plot_sizes.png
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn, optim

# -----------------------------------------------------------------------
# Load pipeline data
# -----------------------------------------------------------------------
cos_sim      = np.load("outputs/cosine_similarity.npy")
inf_mat      = np.load("outputs/influence_matrix.npy")
row_sums     = inf_mat.sum(axis=1)

minor_df     = pd.read_csv("outputs/minor_country_index.csv")
countries    = minor_df["Country"].tolist()
N            = len(countries)
idx_to_ctry  = {i: c for i, c in enumerate(countries)}

cluster_df   = pd.read_csv("outputs/table_countries_by_cluster.csv")
cluster_map  = cluster_df.set_index("Country")["ClusterLabel"].to_dict()

print(f"Pipeline loaded: N={N}")

# -----------------------------------------------------------------------
# Scoring (same formula as exhaustive search and greedy)
# -----------------------------------------------------------------------
def tipping_score(indices):
    k = len(indices)
    if k < 2:
        return 0.0
    c = list(indices)
    sub_cos = cos_sim[np.ix_(c, c)]
    alignment = (sub_cos.sum() - k) / (k * (k - 1))
    sub_inf   = inf_mat[np.ix_(c, c)]
    spread    = row_sums[c].sum() - sub_inf.sum()
    diversity = len({cluster_map.get(idx_to_ctry[i], "") for i in c})
    return float(alignment * spread * diversity)


def reward_fn(indices, penalty_alpha, threshold):
    score = tipping_score(indices)
    if penalty_alpha > 0:
        score /= len(indices) ** penalty_alpha
    if threshold > 0 and score < threshold:
        return 0.0
    return score


# -----------------------------------------------------------------------
# Policy network
# -----------------------------------------------------------------------
class Policy(nn.Module):
    def __init__(self, n, target_p):
        super().__init__()
        self.fc = nn.Linear(n, n)
        import math
        logit_p = math.log(target_p / (1.0 - target_p))
        nn.init.constant_(self.fc.bias, logit_p)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))


# -----------------------------------------------------------------------
# Single training run
# -----------------------------------------------------------------------
EPOCHS  = 500
LR      = 0.01
BASELINE_ALPHA = 0.05


def run_one(size_cap, penalty_alpha, threshold):
    target_p = size_cap / N
    model    = Policy(N, target_p)
    opt      = optim.Adam(model.parameters(), lr=LR)
    rewards, sizes = [], []
    baseline = 0.0

    for _ in range(EPOCHS):
        state = torch.rand(N)
        probs = model(state)
        k     = torch.randint(2, size_cap + 1, (1,)).item()
        gumbel    = -torch.log(-torch.log(torch.rand(N) + 1e-10) + 1e-10)
        perturbed = torch.log(probs + 1e-10) + gumbel
        coalition = torch.topk(perturbed, k).indices.tolist()

        r = reward_fn(coalition, penalty_alpha, threshold)
        if r == 0.0:
            continue

        baseline = (1 - BASELINE_ALPHA) * baseline + BASELINE_ALPHA * r
        advantage = r - baseline

        log_p = torch.log(probs[torch.tensor(coalition)])
        loss  = -log_p.sum() * advantage
        opt.zero_grad()
        loss.backward()
        opt.step()

        rewards.append(r)
        sizes.append(k)

    mean_r  = float(np.mean(rewards)) if rewards else 0.0
    avg_sz  = float(np.mean(sizes))   if sizes   else 0.0
    n_coal  = len(rewards)
    return mean_r, avg_sz, n_coal


# -----------------------------------------------------------------------
# Sweep
# -----------------------------------------------------------------------
size_caps     = [4, 6, 8]
penalty_alphas = [0.0, 1.0]
thresholds    = [0, 2000, 5000]

records = []
total   = len(size_caps) * len(penalty_alphas) * len(thresholds)
done    = 0

for cap, alpha, thresh in itertools.product(size_caps, penalty_alphas, thresholds):
    label = f"cap={cap}_penalty={alpha}_thresh={'none' if thresh == 0 else thresh}"
    print(f"[{done+1}/{total}] {label} ...", end=" ", flush=True)
    mean_r, avg_sz, n_coal = run_one(cap, alpha, thresh)
    print(f"mean_reward={mean_r:.1f}  avg_size={avg_sz:.2f}  n={n_coal}")
    records.append({
        "Label":         label,
        "SizeCap":       cap,
        "PenaltyAlpha":  alpha,
        "Threshold":     thresh,
        "MeanReward":    round(mean_r, 2),
        "AvgSize":       round(avg_sz, 2),
        "NumCoalitions": n_coal,
    })
    done += 1

df = pd.DataFrame(records)
df.to_csv("outputs/rl_experiment_summary.csv", index=False)
print("\nSaved: outputs/rl_experiment_summary.csv")

# -----------------------------------------------------------------------
# LaTeX table
# -----------------------------------------------------------------------
tex = df.to_latex(index=False, float_format="%.2f")
tex_path = "../preprint paper/rl_experiment_summary_table.tex"
with open(tex_path, "w") as f:
    f.write(tex)
print(f"Saved: {tex_path}")

# -----------------------------------------------------------------------
# Figure 1: mean reward by size cap and penalty
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
for ax, cap in zip(axes, size_caps):
    sub = df[df["SizeCap"] == cap]
    x     = np.arange(len(thresholds))
    width = 0.35
    for j, alpha in enumerate(penalty_alphas):
        row = sub[sub["PenaltyAlpha"] == alpha].sort_values("Threshold")
        ax.bar(x + j * width - width / 2, row["MeanReward"],
               width, label=f"α={alpha}")
    ax.set_xticks(x)
    ax.set_xticklabels(["none", "2,000", "5,000"])
    ax.set_xlabel("Reward Threshold")
    ax.set_ylabel("Mean Reward")
    ax.set_title(f"Size Cap = {cap}")
    ax.legend()
fig.suptitle("Mean Tipping Reward by Configuration")
fig.tight_layout()
fig.savefig("../preprint paper/figures/rl_experiment_summary_plot.png", dpi=150)
print("Saved: rl_experiment_summary_plot.png")

# -----------------------------------------------------------------------
# Figure 2: avg coalition size
# -----------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, cap in zip(axes2, size_caps):
    sub = df[df["SizeCap"] == cap]
    x     = np.arange(len(thresholds))
    width = 0.35
    for j, alpha in enumerate(penalty_alphas):
        row = sub[sub["PenaltyAlpha"] == alpha].sort_values("Threshold")
        ax.bar(x + j * width - width / 2, row["AvgSize"],
               width, label=f"α={alpha}")
    ax.set_xticks(x)
    ax.set_xticklabels(["none", "2,000", "5,000"])
    ax.set_xlabel("Reward Threshold")
    ax.set_ylabel("Avg Coalition Size")
    ax.set_title(f"Size Cap = {cap}")
    ax.set_ylim(0, 9)
    ax.legend()
fig2.suptitle("Average Coalition Size by Configuration")
fig2.tight_layout()
fig2.savefig("../preprint paper/figures/rl_experiment_summary_plot_sizes.png", dpi=150)
print("Saved: rl_experiment_summary_plot_sizes.png")
