from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.classifiers import late_fusion, save_classifier
from src.data import TEST_ACTORS, split_by_actor

FEATURES_DIR = Path("features")
CHECKPOINTS_DIR = Path("checkpoints")
MODALITIES = ["audio", "video"]


def make_classifier():
    """Model agreed in the plan: predict_proba columns are 0-7 in EMOTIONS order."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def main():
    """Trenuje klasyfikatory audio i wideo i ocenia wyniki na aktorach testowych."""
    y = np.load(FEATURES_DIR / "y.npy")
    actors = np.load(FEATURES_DIR / "actors.npy")
    train, test = split_by_actor(actors, TEST_ACTORS)
    print(f"Train: {len(train)} files (actors {sorted(set(actors[train].tolist()))}), "
          f"test: {len(test)} files (actors {sorted(set(actors[test].tolist()))})")

    probas, results = {}, {}
    for m in MODALITIES:
        X = np.load(FEATURES_DIR / f"X_{m}.npy")
        assert len(X) == len(y), f"X_{m}.npy has {len(X)} rows, y.npy has {len(y)}"
        clf = make_classifier().fit(X[train], y[train])
        save_classifier(clf, CHECKPOINTS_DIR / f"{m}.joblib")
        probas[m] = clf.predict_proba(X[test])
        results[m] = accuracy_score(y[test], probas[m].argmax(1))

    fused = late_fusion(probas["audio"], probas["video"]).argmax(1)
    results["fusion"] = accuracy_score(y[test], fused)

    print(f"\n{'Model':<8} | Test accuracy")
    print(f"{'-' * 8}-+-{'-' * 13}")
    for name, acc in results.items():
        print(f"{name:<8} | {acc:6.1%}")
    print("(chance level: 12.5%)")
    print(f"\nSaved models to {CHECKPOINTS_DIR}/")


if __name__ == "__main__":
    main()
