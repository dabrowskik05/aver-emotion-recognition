import subprocess

import imageio_ffmpeg
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from transformers.utils import logging as hf_logging

SAMPLE_RATE = 16000  # wav2vec2 expects 16 kHz
MODEL_NAME = "facebook/wav2vec2-base"

# hides the harmless "unexpected keys" warning when loading wav2vec2
hf_logging.set_verbosity_error()
_device = "cuda" if torch.cuda.is_available() else "cpu"
_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
_model = Wav2Vec2Model.from_pretrained(MODEL_NAME).eval().to(_device)


def load_audio(video_path):
    """Audio track of an .mp4 as a mono 16 kHz float32 array."""
    # ffmpeg writes raw float32 mono samples to stdout
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-i", str(video_path),
        "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-loglevel", "quiet",
        "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True)
    waveform = np.frombuffer(out.stdout, np.float32)
    if waveform.size == 0:
        raise RuntimeError(f"No audio decoded from {video_path}")
    return waveform


def audio_embedding(waveform):
    """wav2vec2 embedding averaged over time, shape (768,)."""
    inputs = _feature_extractor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        hidden = _model(inputs.input_values.to(_device)).last_hidden_state
    return hidden.mean(dim=1).squeeze(0).cpu().numpy()
