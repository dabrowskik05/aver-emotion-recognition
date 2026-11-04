import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.classifiers import late_fusion, save_classifier
from src.data import EMOTIONS, TEST_ACTORS, split_by_actor

FEATURES_DIR = Path("features")
CHECKPOINTS_DIR = Path("checkpoints")
MODALITIES = ["audio", "video"]
MODELS = ["audio", "video", "fusion"]
LABELS = range(len(EMOTIONS))


def make_classifier():
    """Scaler + MLP with one hidden ReLU layer; predict_proba columns follow EMOTIONS order."""
    mlp = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2, max_iter=1000, random_state=0)
    return make_pipeline(StandardScaler(), mlp)


def print_scores(y_true, preds):
    """Table of accuracy and macro F1; returns the scores of every model."""
    # accuracy: share of all test recordings guessed right;
    # macro F1: F1 of every emotion averaged with equal weight, so the smaller neutral class counts as much as the rest
    scores = {}
    header = f"{'Model':<8} | {'Accuracy':>8} | {'F1':>6}"
    line = "-" * len(header)
    print(f"\n{header}\n{line}")
    for name, pred in preds.items():
        if name == "majority":
            print(line)
        scores[name] = {"Accuracy": accuracy_score(y_true, pred),
                        "F1": f1_score(y_true, pred, labels=LABELS, average="macro", zero_division=0)}
        print(f"{name:<8} | {scores[name]['Accuracy']:>8.1%} | {scores[name]['F1']:>6.1%}")
    print(f"{'random':<8} | {1 / len(EMOTIONS):>8.1%} |")
    print("F1 is the macro average over the 8 emotions.")
    return scores


def print_per_emotion(y_true, preds):
    """F1 of every emotion for each model and the most common fusion mistake, hardest emotion first.

    Returns the per-emotion F1 and the confusion matrix of the fusion.
    """
    f1 = {m: f1_score(y_true, preds[m], labels=LABELS, average=None, zero_division=0) for m in MODELS}
    cm = confusion_matrix(y_true, preds["fusion"], labels=LABELS)

    header = f"{'Emotion':<9} | {'audio':>5} | {'video':>5} | {'fusion':>6} | Fusion most often mistakes it for"
    print(f"\nF1 per emotion\n{header}\n{'-' * len(header)}")
    for e in np.argsort(f1["fusion"]):
        mistakes = cm[e].copy()
        mistakes[e] = 0
        top = mistakes.argmax()
        mistake = f"{EMOTIONS[top]} ({mistakes[top] / cm[e].sum():.0%})" if mistakes.any() else "-"
        print(f"{EMOTIONS[e]:<9} | {f1['audio'][e]:>5.2f} | {f1['video'][e]:>5.2f} | "
              f"{f1['fusion'][e]:>6.2f} | {mistake}")
    return f1, cm


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

    scores = print_scores(y[test], preds)
    f1, cm = print_per_emotion(y[test], preds)
    print(f"\nSaved models to {CHECKPOINTS_DIR}/")

    if "--plot" in sys.argv:
        from src.plots import PLOTS_DIR, plot_confusion, plot_metrics, plot_per_emotion

        PLOTS_DIR.mkdir(exist_ok=True)
        plot_metrics({m: scores[m] for m in MODELS}, PLOTS_DIR / "metrics.png")
        plot_per_emotion(f1, PLOTS_DIR / "per_emotion.png")
        plot_confusion(cm, PLOTS_DIR / "confusion_fusion.png")
        print(f"Saved plots to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
