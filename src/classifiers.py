def save_classifier(clf, path):
    """Zapisuje klasyfikator na dysk."""
    raise NotImplementedError


def load_classifier(path):
    """Wczytuje klasyfikator zapisany przez save_classifier."""
    raise NotImplementedError


def late_fusion(proba_audio, proba_video):
    """Łączy prawdopodobieństwa z audio i wideo."""
    return (proba_audio + proba_video) / 2
