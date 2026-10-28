"""Synthetic defect generators. Each takes clean speech (float32, mono, SR) and returns a new array.

`narrow` = fixed strength/conditions, `diverse` = randomised (domain randomisation),
matching hypothesis H2 of the CSCI 6364 proposal.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from .config import SR


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def peak_normalize(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    return (x / (np.max(np.abs(x)) + 1e-12) * peak).astype(np.float32)


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale `noise` so speech-to-noise ratio equals snr_db, then add."""
    noise = noise[: len(speech)]
    if len(noise) < len(speech):
        noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))[: len(speech)]
    gain = _rms(speech) / (_rms(noise) * 10 ** (snr_db / 20))
    return speech + gain * noise


def add_hum(speech: np.ndarray, rng: np.random.Generator, strategy: str) -> tuple[np.ndarray, dict]:
    t = np.arange(len(speech)) / SR
    if strategy == "narrow":
        f0, n_harm, snr = 60.0, 3, 10.0
    else:
        f0 = float(rng.choice([50.0, 60.0]))
        n_harm = int(rng.integers(1, 7))
        snr = float(rng.uniform(0, 20))
    hum = sum((0.6 ** k) * np.sin(2 * np.pi * f0 * (k + 1) * t + rng.uniform(0, 2 * np.pi))
              for k in range(n_harm))
    return mix_at_snr(speech, hum, snr), {"f0": f0, "harmonics": n_harm, "snr_db": snr}


def add_clipping(speech: np.ndarray, rng: np.random.Generator, strategy: str) -> tuple[np.ndarray, dict]:
    gain_db = 18.0 if strategy == "narrow" else float(rng.uniform(8, 30))
    ceiling = 0.5 if strategy == "narrow" else float(rng.uniform(0.3, 0.8))
    x = peak_normalize(speech, ceiling) * 10 ** (gain_db / 20)
    return np.clip(x, -ceiling, ceiling), {"gain_db": gain_db, "ceiling": ceiling}


def add_recording(speech: np.ndarray, background: np.ndarray, rng: np.random.Generator,
                  strategy: str) -> tuple[np.ndarray, dict]:
    snr = 5.0 if strategy == "narrow" else float(rng.uniform(-5, 15))
    return mix_at_snr(speech, background, snr), {"snr_db": snr}


def synthetic_room_ir(rng: np.random.Generator, rt60: float) -> np.ndarray:
    """Exponentially decaying noise impulse response (a cheap stand-in for OpenSLR SLR28 RIRs)."""
    n = int(SR * rt60)
    ir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
    ir[0] = 1.0
    return ir / np.max(np.abs(ir))


def add_reverb(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    rt60 = float(rng.uniform(0.15, 0.7))
    y = fftconvolve(x, synthetic_room_ir(rng, rt60))[: len(x)]
    return y * (_rms(x) / _rms(y)), {"rt60": rt60}
