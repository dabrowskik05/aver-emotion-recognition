from pathlib import Path

import numpy as np

from src.data import list_videos, parse_actor, parse_emotion

DATA_DIR = Path("data/ravdess")
FEATURES_DIR = Path("features")


def extract_audio(paths):
    """Compute a wav2vec2 embedding for every file and save features/X_audio.npy."""
    from src.audio import audio_embedding, load_audio  # imported here: loading the model takes a while

    embeddings = []
    for i, path in enumerate(paths, 1):
        embeddings.append(audio_embedding(load_audio(path)))
        print(f"\r[audio] {i}/{len(paths)}", end="", flush=True)
    print()

    X_audio = np.stack(embeddings)
    np.save(FEATURES_DIR / "X_audio.npy", X_audio)
    print(f"Saved X_audio.npy {X_audio.shape}")

def extract_video(paths):
    """Compute video embeddings and save features/X_video.npy."""
    embs = []
    for i, path in enumerate(paths, 1):
        frames = sample_frames(path, NUM_FRAMES)
        faces = [crop_face(f) for f in frames]
        embs.append(video_embedding(faces))
        if i % 50 == 0 or i == len(paths):
            print(f"Video: {i}/{len(paths)}")
    FEATURES_DIR.mkdir(exist_ok=True)
    X = np.stack(embs)
    np.save(FEATURES_DIR / "X_video.npy", X)
    return X
from src.video import NUM_FRAMES, crop_face, sample_frames, video_embedding

FEATURES_DIR = Path("features")

def main():
    """Compute labels and audio/video embeddings for all files and save them to features/."""
    FEATURES_DIR.mkdir(exist_ok=True)
    paths = list_videos(DATA_DIR)  # sorted, so row i is the same recording in every .npy file

    np.save(FEATURES_DIR / "y.npy", np.array([parse_emotion(p) for p in paths]))
    np.save(FEATURES_DIR / "actors.npy", np.array([parse_actor(p) for p in paths]))
    print(f"Saved y.npy and actors.npy ({len(paths)} files)")

    extract_audio(paths)
    extract_video(paths)


if __name__ == "__main__":
    main()
