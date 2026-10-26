from pathlib import Path

EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


def list_videos(data_dir):
    """Lista plików .mp4 z audio i wideo."""
    raise NotImplementedError


def parse_emotion(path):
    """Etykieta 0-7 z 3. liczby w nazwie pliku."""
    return int(Path(path).stem.split("-")[2]) - 1


def parse_actor(path):
    """Numer aktora z ostatniej liczby w nazwie pliku."""
    return int(Path(path).stem.split("-")[-1])


def split_by_actor(actors, test_actors):
    """Indeksy (train, test) dla tablicy aktorów."""
    raise NotImplementedError
