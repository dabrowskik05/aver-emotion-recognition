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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python predict.py <video.mp4> [--plot]")
    probas = predict_proba(sys.argv[1])
    print(EMOTIONS[probas["fusion"].argmax()])

    if "--plot" in sys.argv:
        from src.plots import PLOTS_DIR, plot_proba

        PLOTS_DIR.mkdir(exist_ok=True)
        path = PLOTS_DIR / f"{Path(sys.argv[1]).stem}.png"
        plot_proba(probas, path)
        print(f"Saved {path}")
