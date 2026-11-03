import subprocess

import imageio_ffmpeg
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from transformers.utils import logging as hf_logging

SAMPLE_RATE = 16000  # wav2vec2 expects 16 kHz mono input
MODEL_NAME = "facebook/wav2vec2-base"

# Loaded once per process; downloaded to ~/.cache/huggingface on first use.
# The checkpoint contains pretraining-only weights (quantizer, project_*) that the
# base model does not use; the resulting "unexpected keys" report is harmless.
hf_logging.set_verbosity_error()
device = "cuda" if torch.cuda.is_available() else "cpu"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2Model.from_pretrained(MODEL_NAME).eval().to(device)


def load_audio(video_path):
    """Audio track of an .mp4 as a 1-D float32 array (mono, 16 kHz)."""
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-i", str(video_path),
        "-f", "f32le",            # raw 32-bit float samples
        "-ac", "1",               # mono
        "-ar", str(SAMPLE_RATE),  # resample to 16 kHz
        "-loglevel", "quiet",
        "-",                      # write to stdout
    ]
    out = subprocess.run(cmd, capture_output=True, check=True)
    waveform = np.frombuffer(out.stdout, np.float32)
    if waveform.size == 0:
        raise RuntimeError(f"No audio decoded from {video_path}")
    return waveform


@torch.no_grad()
def audio_embedding(waveform):
    """Embedding of a recording from wav2vec2, averaged over time; shape (768,)."""
    inputs = feature_extractor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    hidden = model(inputs.input_values.to(device)).last_hidden_state  # (1, time, 768)
    return hidden.mean(dim=1).squeeze(0).cpu().numpy()
