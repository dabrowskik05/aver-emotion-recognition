import sys
from pathlib import Path

from src.audio import audio_embedding, load_audio
from src.classifiers import late_fusion, load_classifier
from src.data import EMOTIONS
from src.video import NUM_FRAMES, crop_face, sample_frames, video_embedding

AUDIO_MODEL = "checkpoints/audio.joblib"
VIDEO_MODEL = "checkpoints/video.joblib"


def predict_proba(video_path):
    """Probabilities of the 8 emotions from audio, video and their fusion."""
    # must match extract_features.py, otherwise the features differ from training
    x_audio = audio_embedding(load_audio(video_path)).reshape(1, -1)
    faces = [crop_face(f) for f in sample_frames(video_path, NUM_FRAMES)]
    x_video = video_embedding(faces).reshape(1, -1)

    p_audio = load_classifier(AUDIO_MODEL).predict_proba(x_audio)[0]
    p_video = load_classifier(VIDEO_MODEL).predict_proba(x_video)[0]
    return {"audio": p_audio, "video": p_video, "fusion": late_fusion(p_audio, p_video)}


def predict(video_path):
    """Predicted emotion for a single .mp4 file."""
    return EMOTIONS[predict_proba(video_path)["fusion"].argmax()]


def plot_proba(probas, out_path, figsize=(11, 4.5)):
    """Grouped bar chart of emotion probabilities (audio, video, fusion), saved as PNG."""
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(EMOTIONS))
    width = 0.27
    fig, ax = plt.subplots(figsize=figsize)
    for i, ((name, p), color) in enumerate(zip(probas.items(), ["#2a78d6", "#eb6834", "#1baf7a"])):
        pos = x + (i - 1) * width
        ax.bar(pos, p, width - 0.03, color=color, label=name)
        for px, v in zip(pos, p):
            if v >= 0.005:  # skip bars that would read 0%
                ax.text(px, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, EMOTIONS)
    ax.yaxis.grid(True, linestyle="--", color="lightgray")
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("probability")
    ax.spines[["top", "right"]].set_visible(False)
    # above the axes, so it never covers a tall bar
    ax.legend(ncols=3, edgecolor="none", framealpha=1, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python predict.py <video.mp4> [--plot]")
    probas = predict_proba(sys.argv[1])
    print(EMOTIONS[probas["fusion"].argmax()])
    if "--plot" in sys.argv:
        out_path = Path(sys.argv[1]).stem + ".png"
        plot_proba(probas, out_path)
        print(f"Saved {out_path}")