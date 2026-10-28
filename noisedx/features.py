"""The 38 hand-crafted features from the CSCI 6364 proposal, plus log-mel spectrograms."""
from __future__ import annotations

import librosa
import numpy as np

from .config import HOP, N_MELS, SR

N_MFCC = 13


def feature_names() -> list[str]:
    names = [f"mfcc{i+1}_mean" for i in range(N_MFCC)] + [f"mfcc{i+1}_var" for i in range(N_MFCC)]
    names += [f"contrast{i+1}" for i in range(7)]
    names += ["centroid", "rolloff", "flatness", "zcr", "crest_factor"]
    return names


# Plain-English descriptions shown in the app.
FEATURE_HELP = {
    "centroid": "Spectral centroid: the 'centre of mass' of the spectrum (Hz). Low = dark/boomy.",
    "rolloff": "Spectral roll-off: frequency below which 85% of the energy sits (Hz).",
    "flatness": "Spectral flatness: 0 = tonal (like a hum), 1 = noise-like (like a fan).",
    "zcr": "Zero-crossing rate: how often the waveform crosses zero; high for hiss.",
    "crest_factor": "Crest factor: peak level / average (RMS) level. Clipping squashes it.",
}


def extract_features(y: np.ndarray, sr: int = SR) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    crest = np.max(np.abs(y)) / (np.sqrt(np.mean(y ** 2)) + 1e-12)
    return np.concatenate([
        mfcc.mean(axis=1), mfcc.var(axis=1), contrast.mean(axis=1),
        [centroid.mean(), rolloff.mean(), flatness.mean(), zcr.mean(), crest],
    ]).astype(np.float32)


def log_mel(y: np.ndarray, sr: int = SR) -> np.ndarray:
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP)
    return librosa.power_to_db(m, ref=1.0, top_db=None).astype(np.float32)


def frame_envelope(y: np.ndarray, hop: int = HOP) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame RMS (dB) and peak, for the time-series tab."""
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    return librosa.amplitude_to_db(rms, ref=1.0), np.arange(len(rms)) * hop / SR
