from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.classifiers import late_fusion, save_classifier
from src.data import EMOTIONS, TEST_ACTORS, split_by_actor

FEATURES_DIR = Path("features")
CHECKPOINTS_DIR = Path("checkpoints")
MODALITIES = ["audio", "video"]


def make_classifier():
    """Scaler + logistic regression; predict_proba columns follow EMOTIONS order."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


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

    header = f"{'Model':<8} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>6} | {'F1':>6}"
    line = "-" * len(header)
    print(f"\n{header}\n{line}")
    for name, pred in preds.items():
        if name == "majority":
            print(line)
        p, r, f1, _ = precision_recall_fscore_support(y[test], pred, average="macro", zero_division=0)
        print(f"{name:<8} | {accuracy_score(y[test], pred):>8.1%} | {p:>9.1%} | {r:>6.1%} | {f1:>6.1%}")
    print(f"{'random':<8} | {1 / len(EMOTIONS):>8.1%} |")
    print("Precision, recall and F1 are macro averages over the 8 emotions.")
    print(f"\nSaved models to {CHECKPOINTS_DIR}/")


if __name__ == "__main__":
    main()
