"""One-figure calibration story for the project page (web/assets/fig):
reliability curves + confidence histograms per model, standard benchmark
(MMLU-Pro) vs failure-reasoning target set.

Run from anywhere: python analysis/scripts/make_web_calibration_figure.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STD = ROOT / "outputs/analysis/llm_calibration_standard_entropy/results_all_models.csv"
TGT = ROOT / "outputs/analysis/llm_calibration_target_distribution/results_all_models.csv"
OUT = ROOT / "web/assets/fig/calibration_std_vs_target.png"

# palette (validated reference instance, light mode)
BLUE = "#2a78d6"      # target series (the story)
GRAY = "#898781"      # standard series (de-emphasis)
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
SURF = "#ffffff"      # page figure background is white

MODELS = ["gpt-5.4", "qwen3.5:27b", "qwen3.5:9b"]
# published ECE (std -> tgt), from metrics_by_model.csv of each experiment
ECE = {"gpt-5.4": (0.165, 0.281), "qwen3.5:27b": (0.062, 0.053), "qwen3.5:9b": (0.088, 0.051)}

std = pd.read_csv(STD).dropna(subset=["confidence", "correct"])
tgt = pd.read_csv(TGT).dropna(subset=["confidence", "correct"])

def reliability(df, model, n_bins=10, min_count=5):
    d = df[df["model"] == model]
    conf = d["confidence"].to_numpy()
    corr = d["correct"].to_numpy().astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() >= min_count:
            xs.append(conf[m].mean())
            ys.append(corr[m].mean())
            ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.edgecolor": BASE,
    "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
})

fig = plt.figure(figsize=(13.2, 5.9), dpi=170)
gs = fig.add_gridspec(2, 3, height_ratios=[3.1, 1.0], hspace=0.10, wspace=0.16,
                      left=0.055, right=0.985, top=0.76, bottom=0.10)

hist_edges = np.linspace(0, 1, 21)

for j, model in enumerate(MODELS):
    ax = fig.add_subplot(gs[0, j])
    hx = fig.add_subplot(gs[1, j], sharex=ax)

    # --- reliability panel ---
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.plot([0, 1], [0, 1], color=BASE, linewidth=1.0, zorder=1)

    for df, color, z in ((std, GRAY, 2), (tgt, BLUE, 3)):
        xs, ys, ns = reliability(df, model)
        ax.plot(xs, ys, color=color, linewidth=2, zorder=z,
                marker="o", markersize=7.5, markerfacecolor=color,
                markeredgecolor=SURF, markeredgewidth=1.6, solid_capstyle="round")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.tick_params(length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    e_std, e_tgt = ECE[model]
    ax.set_title(model, fontsize=13, fontweight="bold", color=INK, pad=26, loc="left")
    ax.text(0, 1.06, f"ECE  {e_std:.3f} standard  →  {e_tgt:.3f} target",
            transform=ax.transAxes, fontsize=10, color=INK2)

    if j == 0:
        ax.set_ylabel("empirical accuracy", fontsize=10, color=INK2)

    # --- histogram panel ---
    for df, color, filled in ((std, GRAY, False), (tgt, BLUE, True)):
        d = df[df["model"] == model]
        w = np.ones(len(d)) / len(d)
        if filled:
            hx.hist(d["confidence"], bins=hist_edges, weights=w,
                    color=BLUE, alpha=0.55, zorder=3)
        else:
            hx.hist(d["confidence"], bins=hist_edges, weights=w,
                    histtype="step", color=GRAY, linewidth=1.6, zorder=2)
    hx.set_xlim(0, 1)
    hx.set_ylim(0, 0.85)
    hx.set_yticks([0, 0.4, 0.8])
    hx.grid(True, axis="y", color=GRID, linewidth=0.8)
    hx.set_axisbelow(True)
    hx.tick_params(length=0)
    for s in ("top", "right"):
        hx.spines[s].set_visible(False)
    hx.set_xlabel("model confidence  (1 − normalized entropy)", fontsize=10, color=INK2)
    if j == 0:
        hx.set_ylabel("share", fontsize=10, color=INK2)

    # single annotation on the story panel
    if model == "gpt-5.4":
        d = tgt[tgt["model"] == model]
        top_share = ((d["confidence"] >= 0.95)).mean()
        hx.annotate(f"{top_share:.0%} of answers above\n0.95 confidence",
                    xy=(0.945, min(0.80, top_share)), xytext=(0.55, 0.50),
                    ha="right", va="center", fontsize=9.5, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=BASE, linewidth=1.0))
        ax.annotate("high confidence,\ncoin-flip accuracy",
                    xy=(0.77, 0.295), xytext=(0.97, 0.08),
                    ha="right", va="bottom", fontsize=9.5, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=BASE, linewidth=1.0))

# headline + legend
fig.text(0.055, 0.960, "Accuracy transfers, calibration does not",
         fontsize=16, fontweight="bold", color=INK)
fig.text(0.055, 0.912, "Reliability curves (top) and confidence distributions (bottom) on the standard benchmark",
         fontsize=10.5, color=INK2)
fig.text(0.055, 0.876, "versus the robot failure-reasoning target set. Perfect calibration lies on the diagonal.",
         fontsize=10.5, color=INK2)

legend_handles = [
    Line2D([], [], color=GRAY, linewidth=2, marker="o", markersize=7.5,
           markerfacecolor=GRAY, markeredgecolor=SURF, markeredgewidth=1.6,
           label="standard (MMLU-Pro)"),
    Line2D([], [], color=BLUE, linewidth=2, marker="o", markersize=7.5,
           markerfacecolor=BLUE, markeredgecolor=SURF, markeredgewidth=1.6,
           label="target (failure reasoning)"),
]
fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.985, 0.985),
           frameon=False, fontsize=10.5, handlelength=2.2, labelcolor=INK2)

fig.savefig(OUT, dpi=170, facecolor=SURF)
print("saved", OUT)
