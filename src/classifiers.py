from pathlib import Path

import joblib


def save_classifier(clf, path):
    """Zapisuje klasyfikator na dysk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, path)


def load_classifier(path):
    """Wczytuje klasyfikator zapisany przez save_classifier."""
    if not Path(path).exists():
        raise FileNotFoundError(f"No classifier at {path}; run train.py first")
    return joblib.load(path)


def late_fusion(proba_audio, proba_video):
    """Łączy prawdopodobieństwa z audio i wideo."""
    return (proba_audio + proba_video) / 2
