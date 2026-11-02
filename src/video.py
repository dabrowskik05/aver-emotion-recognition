import cv2
import numpy as np
import torch
from torchvision.models import ResNet18_Weights, resnet18

NUM_FRAMES = 16

# Haar face detector (bundled with OpenCV) and ResNet-18 backbone loaded once.
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_device = "cuda" if torch.cuda.is_available() else "cpu"
_weights = ResNet18_Weights.DEFAULT
_transform = _weights.transforms()
_model = resnet18(weights=_weights)
_model.fc = torch.nn.Identity()  # features instead of ImageNet class scores
_model = _model.eval().to(_device)


def sample_frames(video_path, num_frames):
    """Evenly spaced RGB frames from a clip."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    # Read all frames: CAP_PROP_FRAME_COUNT can be too high for .mp4.
    all_frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        all_frames.append(frame)
    cap.release()
    if not all_frames:
        raise ValueError(f"No frames in video: {video_path}")
    indices = np.linspace(0, len(all_frames) - 1, num_frames).astype(int)
    return [cv2.cvtColor(all_frames[i], cv2.COLOR_BGR2RGB) for i in indices]


def crop_face(frame):
    """Largest face crop with ~15% margin; whole frame if none found."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return frame  # keep pipeline running when detection fails
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    mx, my = int(0.15 * w), int(0.15 * h)
    H, W = frame.shape[:2]
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(W, x + w + mx), min(H, y + h + my)
    return frame[y1:y2, x1:x2]


def video_embedding(faces):
    """Mean ResNet-18 embedding over face crops, shape (512,)."""
    batch = torch.stack([
        _transform(torch.from_numpy(face).permute(2, 0, 1)) for face in faces
    ]).to(_device)
    with torch.no_grad():
        emb = _model(batch).mean(dim=0)
    return emb.cpu().numpy()


# Video-branch sanity check.
#
# Usage:
#     python -m src.video
#
# Saves a frame and a face crop to data/ and checks the embedding shape.
if __name__ == "__main__":
    from src.data import list_videos

    path = list_videos("data/ravdess")[0]
    frames = sample_frames(path, NUM_FRAMES)
    faces = [crop_face(f) for f in frames]
    emb = video_embedding(faces)

    cv2.imwrite("data/frame_check.png", cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite("data/face_check.png", cv2.cvtColor(faces[0], cv2.COLOR_RGB2BGR))
    print(f"File: {path.name}")
    print(f"Frames: {len(frames)} x {frames[0].shape}")
    print(f"Face crop: {faces[0].shape} (saved to data/face_check.png)")
    print(f"Embedding: {emb.shape}")
    assert emb.shape == (512,), "Unexpected embedding shape!"
