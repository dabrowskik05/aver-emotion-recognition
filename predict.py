import sys

from src.audio import audio_embedding, load_audio
from src.classifiers import late_fusion, load_classifier
from src.data import EMOTIONS
from src.video import NUM_FRAMES, crop_face, sample_frames, video_embedding

AUDIO_MODEL = "checkpoints/audio.joblib"
VIDEO_MODEL = "checkpoints/video.joblib"


def predict(video_path):
    """Predicted emotion for a single .mp4 file."""
    # must match extract_features.py, otherwise the features differ from training
    x_audio = audio_embedding(load_audio(video_path)).reshape(1, -1)
    faces = [crop_face(f) for f in sample_frames(video_path, NUM_FRAMES)]
    x_video = video_embedding(faces).reshape(1, -1)

    clf_audio = load_classifier(AUDIO_MODEL)
    clf_video = load_classifier(VIDEO_MODEL)
    proba = late_fusion(clf_audio.predict_proba(x_audio), clf_video.predict_proba(x_video))
    return EMOTIONS[proba.argmax(1)[0]]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python predict.py <video.mp4>")
    print(predict(sys.argv[1]))
