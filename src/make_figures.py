"""Figures for REPORT.md, built from results/analysis/*. Run after analyze.py."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

TAGS = {"qwen25_7b": "Qwen2.5-7B-Instruct", "llama31_8b": "Llama-3.1-8B-Instruct"}
DOMAINS = ["biography", "geography", "creative_works", "science", "trivia"]
DLAB = ["biography", "geography", "creative works", "science", "trivia"]
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
A = "results/analysis"

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
                     "legend.frameon": False, "savefig.dpi": 160, "savefig.bbox": "tight"})
R = {t: json.load(open(f"{A}/{t}.json")) for t in TAGS}
PL = {t: pd.read_csv(f"{A}/{t}_per_layer.csv") for t in TAGS}


def fig_layer_profile():
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.4), sharex="col")
    for j, (t, name) in enumerate(TAGS.items()):
        p = PL[t][PL[t].feature == "ent_ans"]
        L = len(p)
        a = ax[0, j]
        a.plot(p.layer, p.mean_factual, color=BLUE, lw=2, label="factual (correct)")
        a.plot(p.layer, p.mean_halluc, color=ORANGE, lw=2, label="hallucinated (wrong)")
        a.axvspan(L - 3.5, L - 0.5, color=MUTED, alpha=0.08, lw=0)
        a.text(L - 2, a.get_ylim()[1] * 0.97, "final 3", ha="center", va="top", color=MUTED, fontsize=8)
        a.set_title(name)
        a.set_ylabel("mean attention entropy (nats)")
        a.legend(loc="lower left")
        b = ax[1, j]
        b.axhline(0, color=MUTED, lw=1)
        b.plot(p.layer, p.cohens_d, color=INK, lw=2, label="pooled over domains")
        b.plot(p.layer, p.d_within_domain, color=INK, lw=2, ls=(0, (2, 2)), label="mean of within-domain d")
        b.axvspan(L - 3.5, L - 0.5, color=MUTED, alpha=0.08, lw=0)
        b.set_ylabel("Cohen's d (hallucinated - factual)")
        b.set_xlabel("layer")
        b.legend(loc="upper right" if j == 0 else "lower left", fontsize=8)
    fig.suptitle("Attention entropy by layer during answer generation", color=INK, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig("figures/fig1_layer_profile.png")
    plt.close(fig)


def fig_last3_effect():
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True, sharex=True)
    keys = ["pooled"] + DOMAINS
    labs = ["POOLED"] + DLAB
    for j, (t, name) in enumerate(TAGS.items()):
        a = ax[j]
        a.axvspan(15, 30, color=BLUE, alpha=0.10, lw=0)
        a.axvline(0, color=MUTED, lw=1)
        for off, feat, col, lab in [(0.15, "ent_ans", INK, "mean over answer tokens"), (-0.15, "ent_first", MUTED, "first answer token")]:
            e = R[t]["E1_last3_effect"][feat]
            x = [e[k]["rel_diff_pct"] for k in keys]
            ci = np.array([e[k]["rel_diff_ci95"] for k in keys])
            yy = np.arange(len(keys))[::-1] + off
            a.errorbar(x, yy, xerr=[x - ci[:, 0], ci[:, 1] - x], fmt="o", color=col, ms=6, lw=1.5, label=lab)
        a.set_yticks(np.arange(len(keys))[::-1], labs)
        a.set_title(name)
        a.set_xlabel("relative difference in final-3-layer entropy,\nhallucinated vs factual (%)")
        a.grid(axis="y", visible=False)
    ax[0].text(22.5, 5.45, "hypothesised\n+15 to +30%", ha="center", va="top", color=BLUE, fontsize=8)
    ax[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/fig2_last3_effect.png")
    plt.close(fig)


def fig_auroc_sets():
    names = ["domain_only", "length_only", "entropy_last3_mean (1 feat)", "entropy_layers (first token)",
             "norm_entropy_layers (ans mean)", "entropy_layers (ans mean)", "entropy_layers + length",
             "entropy_heads (L*H)", "logit_baseline", "entropy_layers + logit"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, sharex=True)
    for j, (t, name) in enumerate(TAGS.items()):
        a = ax[j]
        e = R[t]["E2_pooled_classifiers"]
        yy = np.arange(len(names))[::-1]
        x = np.array([e[n]["auroc"] for n in names])
        ci = np.array([e[n]["ci95"] for n in names])
        w = np.array([e[n]["within_domain_macro_auroc"] for n in names])
        a.axvline(0.75, color=BLUE, lw=1.5, ls=(0, (4, 2)))
        a.axvline(0.5, color=MUTED, lw=1)
        a.errorbar(x, yy, xerr=[x - ci[:, 0], ci[:, 1] - x], fmt="o", color=INK, ms=6, lw=1.5, label="pooled AUROC (95% CI)")
        a.plot(w, yy, "D", color=ORANGE, ms=6, mec="#fcfcfb", mew=1, label="same scores, mean within-domain AUROC")
        a.set_yticks(yy, names)
        a.set_title(name)
        a.set_xlabel("AUROC (5x5-fold CV)")
        a.grid(axis="y", visible=False)
    h, l = ax[1].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=BLUE, lw=1.5, ls=(0, (4, 2)))); l.append("H2 threshold (0.75)")
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.6, -0.06))
    fig.tight_layout()
    fig.savefig("figures/fig3_auroc_feature_sets.png")
    plt.close(fig)


def fig_auroc_domain():
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for j, (t, name) in enumerate(TAGS.items()):
        a = ax[j]
        e = R[t]["E3_within_domain"]
        xx = np.arange(len(DOMAINS))
        for off, key, col, lab in [(-0.12, "entropy_layers (ans mean)", BLUE, "layer-wise attention entropy"),
                                   (0.12, "logit_baseline", ORANGE, "logit baseline")]:
            v = np.array([e[d][key]["auroc"] for d in DOMAINS])
            ci = np.array([e[d][key]["ci95"] for d in DOMAINS])
            a.errorbar(xx + off, v, yerr=[v - ci[:, 0], ci[:, 1] - v], fmt="o", color=col, ms=7, lw=1.5, label=lab)
            for x_, v_ in zip(xx + off, v):
                a.text(x_ + (0.07 if off > 0 else -0.07), v_ + 0.012, f"{v_:.2f}", ha="left" if off > 0 else "right", va="center", color=INK, fontsize=8)
        a.axhline(0.75, color=MUTED, lw=1, ls=(0, (4, 2)))
        a.axhline(0.5, color=MUTED, lw=1)
        a.set_xticks(xx, DLAB)
        a.set_title(name)
        a.grid(axis="x", visible=False)
    ax[0].set_ylabel("within-domain AUROC (95% CI)")
    ax[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/fig4_auroc_by_domain.png")
    plt.close(fig)


def fig_transfer():
    cmap = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#2a78d6", "#0f3a73"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for j, (t, name) in enumerate(TAGS.items()):
        T = pd.read_csv(f"{A}/{t}_transfer.csv", index_col=0).values
        a = ax[j]
        im = a.imshow(T, cmap=cmap, vmin=0.5, vmax=0.9)
        for i in range(5):
            for k in range(5):
                a.text(k, i, f"{T[i, k]:.2f}", ha="center", va="center", color="white" if T[i, k] > 0.72 else INK, fontsize=9)
        a.set_xticks(range(5), DLAB, rotation=25, ha="right")
        a.set_yticks(range(5), DLAB)
        a.set_xlabel("test domain")
        a.set_ylabel("train domain")
        a.set_title(name)
        a.grid(False)
        fig.colorbar(im, ax=a, fraction=0.046, label="AUROC")
    fig.suptitle("Entropy-only classifier: cross-domain transfer (diagonal = within-domain CV)", color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig("figures/fig5_transfer.png")
    plt.close(fig)


if __name__ == "__main__":
    for f in [fig_layer_profile, fig_last3_effect, fig_auroc_sets, fig_auroc_domain, fig_transfer]:
        f()
        print("wrote", f.__name__)
