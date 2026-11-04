from pathlib import Path

import numpy as np

EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

# actor 5 is male, actor 6 is female
TEST_ACTORS = [5, 6]

# fields of a RAVDESS filename, e.g. 01-01-06-01-02-01-12.mp4
FIELDS = ["modality", "vocal_channel", "emotion", "intensity", "statement", "repetition", "actor"]


def parse_filename(path):
    """All 7 filename fields as a dict of ints (values as in the name, starting at 1)."""
    parts = Path(path).stem.split("-")
    if len(parts) != len(FIELDS):
        raise ValueError(f"Unknown RAVDESS filename format: {Path(path).name}")
    return dict(zip(FIELDS, map(int, parts)))


def list_videos(data_dir):
    """Sorted list of .mp4 files that have both audio and video."""
    # 01- is audio+video, 02- is video only (no sound for the audio model)
    paths = sorted(Path(data_dir).rglob("01-*.mp4"))
    if not paths:
        raise FileNotFoundError(f"No 01-*.mp4 files in {Path(data_dir).resolve()}")
    return paths


def parse_emotion(path):
    """Label 0-7 from the 3rd number in the filename."""
    return parse_filename(path)["emotion"] - 1


def parse_actor(path):
    """Actor number from the last number in the filename."""
    return parse_filename(path)["actor"]


def split_by_actor(actors, test_actors):
    """(train, test) indices for an array of actors."""
    # whole actors go to one side, so test faces and voices are never seen in training
    is_test = np.isin(np.asarray(actors), test_actors)
    return np.where(~is_test)[0], np.where(is_test)[0]


# quick check: python -m src.data (counts per actor/emotion and the train/test split)
if __name__ == "__main__":
    from collections import Counter

    paths = list_videos("data/ravdess")
    print(f"Audio+video files: {len(paths)}")
    print(f"First: {paths[0].name} -> emotion {parse_emotion(paths[0])} "
          f"({EMOTIONS[parse_emotion(paths[0])]}), actor {parse_actor(paths[0])}")
    print("Per actor:", dict(sorted(Counter(parse_actor(p) for p in paths).items())))
    print("Per emotion:", {EMOTIONS[e]: n for e, n in sorted(Counter(parse_emotion(p) for p in paths).items())})

    actors = np.array([parse_actor(p) for p in paths])
    train, test = split_by_actor(actors, TEST_ACTORS)
    print(f"Train: {len(train)} files, actors {sorted(set(actors[train].tolist()))}")
    print(f"Test:  {len(test)} files, actors {sorted(set(actors[test].tolist()))}")
    assert not set(actors[train]) & set(actors[test]), "Same actors in train and test!"
