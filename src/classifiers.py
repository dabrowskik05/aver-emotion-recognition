from pathlib import Path

import joblib


def save_classifier(clf, path):
    """Save a classifier to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, path)


def load_classifier(path):
    """Load a classifier saved with save_classifier."""
    if not Path(path).exists():
        raise FileNotFoundError(f"No classifier at {path}; run train.py first")
    return joblib.load(path)


def late_fusion(proba_audio, proba_video):
    """Average the audio and video class probabilities."""
    return (proba_audio + proba_video) / 2
