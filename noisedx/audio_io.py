"""Loading the raw source audio (LibriSpeech speech, ESC-50 backgrounds) and saved clips."""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from .config import CLIP_SAMPLES, RAW, SR

LIBRI_DIR = RAW / "LibriSpeech" / "dev-clean-2"
ESC_DIR = RAW / "ESC-50-master"


def librispeech_index() -> pd.DataFrame:
    """One row per utterance: path, speaker, chapter."""
    rows = []
    for f in sorted(LIBRI_DIR.glob("*/*/*.flac")):
        speaker, chapter = f.parts[-3], f.parts[-2]
        rows.append({"path": str(f), "speaker": speaker, "chapter": chapter, "utt": f.stem})
    if not rows:
        raise FileNotFoundError(f"No LibriSpeech audio in {LIBRI_DIR}. Run scripts/download_data.py first.")
    return pd.DataFrame(rows)


def esc50_index() -> pd.DataFrame:
    meta = pd.read_csv(ESC_DIR / "meta" / "esc50.csv")
    meta["path"] = meta["filename"].map(lambda f: str(ESC_DIR / "audio" / f))
    return meta


def load(path: str | Path, sr: int = SR) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def loudest_window(y: np.ndarray, n: int = CLIP_SAMPLES) -> np.ndarray:
    """Return the n-sample window with the most energy (pads short audio with zeros)."""
    if len(y) <= n:
        return np.pad(y, (0, n - len(y)))
    hop = SR // 4
    starts = np.arange(0, len(y) - n + 1, hop)
    energy = [np.sum(y[s:s + n] ** 2) for s in starts]
    s = starts[int(np.argmax(energy))]
    return y[s:s + n]


def save_wav(path: Path, y: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y, sr, subtype="PCM_16")
