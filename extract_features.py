from pathlib import Path

import numpy as np

from src.video import NUM_FRAMES, crop_face, sample_frames, video_embedding

FEATURES_DIR = Path("features")


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


def main():
    """Liczy embeddingi audio i wideo i zapisuje je."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
