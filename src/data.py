from pathlib import Path

import numpy as np

EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

# Test actors: 5 (male) and 6 (female), one person of each sex.
TEST_ACTORS = [5, 6]

# Fields of a RAVDESS filename, in order, e.g. 01-01-06-01-02-01-12.mp4
FIELDS = ["modality", "vocal_channel", "emotion", "intensity", "statement", "repetition", "actor"]


def parse_filename(path):
    """All 7 filename fields as a dict of ints (values as in the name, starting at 1)."""
    parts = Path(path).stem.split("-")
    if len(parts) != len(FIELDS):
        raise ValueError(f"Unknown RAVDESS filename format: {Path(path).name}")
    return dict(zip(FIELDS, map(int, parts)))


def list_videos(data_dir):
    """Sorted list of .mp4 files that have both audio and video."""
    # 01- = audio+video; 02- (video-only) is skipped because it has no sound for the audio branch.
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
    # Each actor goes entirely to train or to test, so the model never sees a known face or voice in testing.
    is_test = np.isin(np.asarray(actors), test_actors)
    return np.where(~is_test)[0], np.where(is_test)[0]


# Dataset sanity check.
#
# Usage:
#     python -m src.data
#
# Reports the number of audio-visual samples, the per-actor and per-emotion
# distributions, and the actor-disjoint train/test split. Raises AssertionError
# if any actor appears in both splits.
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
