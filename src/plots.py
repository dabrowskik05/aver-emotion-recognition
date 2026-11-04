from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from src.data import EMOTIONS

PLOTS_DIR = Path("plots")
COLORS = {"audio": "#2a78d6", "video": "#eb6834", "fusion": "#1baf7a"}


def _grouped_bars(ax, labels, values, fmt, rotate_values=False):
    """Audio, video and fusion bars side by side for every label, value on top of each bar."""
    x = np.arange(len(labels))
    width = 0.27
    for i, (name, vals) in enumerate(values.items()):
        pos = x + (i - 1) * width
        ax.bar(pos, vals, width - 0.03, color=COLORS[name], label=name)
        for px, v in zip(pos, vals):
            if v >= 0.005:  # labels of empty bars only add clutter
                ax.text(px, v + 0.01, fmt(v), ha="center", va="bottom", fontsize=8,
                        rotation=90 if rotate_values else 0)
    ax.set_xticks(x, labels)
    ax.yaxis.grid(True, linestyle="--", color="lightgray")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig, ax, path):
    # legend above the plot, so it never covers a bar
    ncols = len(ax.get_legend_handles_labels()[1])
    ax.legend(ncols=ncols, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_proba(probas, path):
    """Probability of every emotion from audio, video and fusion for one recording."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    _grouped_bars(ax, EMOTIONS, probas, lambda v: f"{v:.0%}")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("probability")
    _save(fig, ax, path)


def plot_metrics(scores, path):
    """Accuracy and macro F1 of each model, with the random-guess level."""
    metrics = list(next(iter(scores.values())))
    values = {name: [s[k] for k in metrics] for name, s in scores.items()}
    top = max(max(v) for v in values.values())

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    _grouped_bars(ax, metrics, values, lambda v: f"{v:.1%}")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    chance = 1 / len(EMOTIONS)
    ax.axhline(chance, color="gray", linestyle=":", linewidth=1.5, label=f"random ({chance:.1%})")
    ax.set_ylim(0, top + 0.1)
    _save(fig, ax, path)


def plot_per_emotion(f1, path):
    """F1 of every emotion for each model."""
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    _grouped_bars(ax, EMOTIONS, f1, lambda v: f"{v:.2f}", rotate_values=True)
    ax.set_ylim(0, 1.15)  # room for the rotated labels above 1
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("F1")
    _save(fig, ax, path)


def plot_confusion(cm, path):
    """Confusion matrix of the fusion (rows: true emotion, columns: predicted), shaded by row share."""
    share = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.imshow(share, cmap="Purples", vmin=0, vmax=1)
    for i, j in np.ndindex(cm.shape):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                    color="white" if share[i, j] > 0.45 else "black",
                    fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(len(EMOTIONS)), EMOTIONS, rotation=30, ha="right")
    ax.set_yticks(range(len(EMOTIONS)), EMOTIONS)
    ax.set_xlabel("predicted emotion")
    ax.set_ylabel("true emotion")
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
