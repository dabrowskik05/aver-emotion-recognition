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


def plot_proba(probas, out_path):
    """Three bar charts (audio, video, fusion) of emotion probabilities, saved as PNG."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (name, p), color in zip(axes, probas.items(), ["#2a78d6", "#eb6834", "#1baf7a"]):
        ax.bar(EMOTIONS, p, color=color)
        ax.set_xlabel(name, fontsize=13)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)
        ax.spines[["top", "right"]].set_visible(False)
        top = p.argmax()
        ax.text(top, p[top] + 0.02, f"{p[top]:.0%}", ha="center")
    axes[0].set_ylabel("probability")
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
