import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.classifiers import late_fusion, save_classifier
from src.data import EMOTIONS, TEST_ACTORS, split_by_actor

FEATURES_DIR = Path("features")
CHECKPOINTS_DIR = Path("checkpoints")
PLOTS_DIR = Path("plots")
MODALITIES = ["audio", "video"]
COLORS = {"audio": "#2a78d6", "video": "#eb6834", "fusion": "#1baf7a"}  # same as plot_proba in predict.py


def make_classifier():
    """Scaler + MLP with one hidden ReLU layer; predict_proba columns follow EMOTIONS order."""
    mlp = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2, max_iter=1000, random_state=0)
    return make_pipeline(StandardScaler(), mlp)


def main():
    """Train audio and video classifiers and evaluate them on the test actors."""
    y = np.load(FEATURES_DIR / "y.npy")
    actors = np.load(FEATURES_DIR / "actors.npy")
    train, test = split_by_actor(actors, TEST_ACTORS)
    print(f"Train: {len(train)} files (actors {sorted(set(actors[train].tolist()))}), "
          f"test: {len(test)} files (actors {sorted(set(actors[test].tolist()))})")

    probas, preds = {}, {}
    for m in MODALITIES:
        X = np.load(FEATURES_DIR / f"X_{m}.npy")
        assert len(X) == len(y), f"X_{m}.npy has {len(X)} rows, y.npy has {len(y)}"
        clf = make_classifier().fit(X[train], y[train])
        save_classifier(clf, CHECKPOINTS_DIR / f"{m}.joblib")
        probas[m] = clf.predict_proba(X[test])
        preds[m] = probas[m].argmax(1)
    preds["fusion"] = late_fusion(probas["audio"], probas["video"]).argmax(1)
    # baseline: always the most frequent emotion in train
    preds["majority"] = np.full(len(test), np.bincount(y[train]).argmax())

    scores = {}
    header = f"{'Model':<8} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>6} | {'F1':>6}"
    line = "-" * len(header)
    print(f"\n{header}\n{line}")
    for name, pred in preds.items():
        if name == "majority":
            print(line)
        p, r, f1, _ = precision_recall_fscore_support(y[test], pred, average="macro", zero_division=0)
        scores[name] = {"Accuracy": accuracy_score(y[test], pred), "Precision": p, "Recall": r, "F1": f1}
        print(f"{name:<8} | {scores[name]['Accuracy']:>8.1%} | {p:>9.1%} | {r:>6.1%} | {f1:>6.1%}")
    print(f"{'random':<8} | {1 / len(EMOTIONS):>8.1%} |")
    print("Precision, recall and F1 are macro averages over the 8 emotions.")

    # per emotion: % of its test recordings each model got right, hardest for fusion first
    cms = {m: confusion_matrix(y[test], preds[m], labels=range(len(EMOTIONS))) for m in ["audio", "video", "fusion"]}
    correct = {m: cm.diagonal() / cm.sum(1) for m, cm in cms.items()}
    header = f"{'Emotion':<9} | {'audio':>5} | {'video':>5} | {'fusion':>6} | Fusion most often mistakes it for"
    print(f"\n{header}\n{'-' * len(header)}")
    for e in np.argsort(correct["fusion"]):
        row = cms["fusion"][e].copy()
        row[e] = 0
        wrong = row.argmax()
        mistake = f"{EMOTIONS[wrong]} ({row[wrong] / cms['fusion'][e].sum():.0%})" if row.any() else "-"
        print(f"{EMOTIONS[e]:<9} | {correct['audio'][e]:>5.0%} | {correct['video'][e]:>5.0%} | "
              f"{correct['fusion'][e]:>6.0%} | {mistake}")
    print(f"\nSaved models to {CHECKPOINTS_DIR}/")

    if "--plot" in sys.argv:
        PLOTS_DIR.mkdir(exist_ok=True)
        plot_metrics({m: scores[m] for m in ["audio", "video", "fusion"]}, PLOTS_DIR / "metrics.png")
        plot_per_emotion(correct, PLOTS_DIR / "per_emotion.png")
        plot_confusion(cms["fusion"], PLOTS_DIR / "confusion_fusion.png")
        print(f"Saved plots to {PLOTS_DIR}/")


def _style(ax):
    """Shared look of the bar charts, matching plot_proba in predict.py."""
    ax.yaxis.grid(True, linestyle="--", color="lightgray")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ncols = len(ax.get_legend_handles_labels()[1])
    ax.legend(ncols=ncols, edgecolor="none", framealpha=1, loc="upper center", bbox_to_anchor=(0.5, 1.12))


def _grouped_bars(ax, labels, values_by_model, fmt, vertical_values=False):
    """Bars of audio/video/fusion side by side for every label, with value on top."""
    x = np.arange(len(labels))
    width = 0.27
    for i, (name, values) in enumerate(values_by_model.items()):
        pos = x + (i - 1) * width
        ax.bar(pos, values, width - 0.03, color=COLORS[name], label=name)
        for px, v in zip(pos, values):
            ax.text(px, v + 0.01, fmt(v), ha="center", va="bottom", fontsize=9,
                    rotation=90 if vertical_values else 0)
    ax.set_xticks(x, labels)


def plot_metrics(scores, out_path):
    """Accuracy and macro precision/recall/F1 of every model on the test set, saved as PNG."""
    import matplotlib.pyplot as plt

    names = list(next(iter(scores.values())))
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    _grouped_bars(ax, names, {m: [s[k] for k in names] for m, s in scores.items()}, lambda v: f"{v:.1%}")
    ax.axhline(1 / len(EMOTIONS), color="gray", linestyle=":", linewidth=1.5, label=f"random ({1 / len(EMOTIONS):.1%})")
    ax.set_ylim(0, 0.7)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_per_emotion(correct, out_path):
    """Share of test recordings of every emotion recognised correctly by each model, saved as PNG."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    _grouped_bars(ax, EMOTIONS, correct, lambda v: f"{v:.0%}", vertical_values=True)
    ax.set_ylim(0, 1.2)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("recordings recognised correctly")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion(cm, out_path):
    """Confusion matrix (rows: true emotion, columns: predicted), shaded by row share, saved as PNG."""
    import matplotlib.pyplot as plt

    share = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.imshow(share, cmap="Purples", vmin=0, vmax=1)
    for i, j in np.ndindex(cm.shape):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                    color="white" if share[i, j] > 0.45 else "black", fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(len(EMOTIONS)), EMOTIONS, rotation=30, ha="right")
    ax.set_yticks(range(len(EMOTIONS)), EMOTIONS)
    ax.set_xlabel("predicted emotion")
    ax.set_ylabel("true emotion")
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
