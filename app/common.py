"""Cached data loaders and small plotting helpers shared by every tab."""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from noisedx.config import (CLASS_COLORS, CLASS_LABELS, CLASSES, CLIPS, N_MELS,  # noqa: E402
                            PROCESSED, SR)
from noisedx.features import feature_names  # noqa: E402

FONT = dict(family="Inter, Helvetica, Arial, sans-serif", size=14, color="#0b0b0b")
GRID = "#e4e3df"


def _stamp() -> float:
    """Modification time of the processed data, so caches refresh after build_dataset.py is re-run."""
    return (PROCESSED / "features.csv").stat().st_mtime


_seen_stamp: float | None = None


def refresh_cache_if_data_changed() -> None:
    """Clear every st.cache_data result once the dataset on disk changes (imported modules persist between reruns)."""
    global _seen_stamp
    stamp = _stamp()
    if _seen_stamp is not None and stamp != _seen_stamp:
        st.cache_data.clear()
    _seen_stamp = stamp


@st.cache_data
def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(PROCESSED / "metadata.csv")
    feats = pd.read_csv(PROCESSED / "features.csv")
    meta["label"] = pd.Categorical(meta["label"], CLASSES, ordered=True)
    return meta, feats


@st.cache_data
def load_mels() -> tuple[np.ndarray, np.ndarray]:
    """Log-mel spectrograms, each shifted so its loudest cell is 0 dB and floored at -80 dB (standard display range)."""
    z = np.load(PROCESSED / "mels.npz")
    m = z["mels"].astype(np.float32)
    m -= m.max(axis=(1, 2), keepdims=True)
    return z["ids"], np.maximum(m, -80.0)


@st.cache_data
def mel_hz() -> np.ndarray:
    return librosa.mel_frequencies(n_mels=N_MELS, fmax=SR / 2)


@st.cache_data
def load_wav(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def showcase_paths(strategy: str, cls: str) -> list[Path]:
    return sorted((CLIPS / "showcase" / strategy / cls).glob("*.wav"))


def source_paths(prefix: str) -> list[Path]:
    return sorted((CLIPS / "sources").glob(f"{prefix}_*.wav"))


def data_ready() -> bool:
    return (PROCESSED / "features.csv").exists()


def style(fig: go.Figure, height: int = 420, title: str | None = None) -> go.Figure:
    """Shared look: white surface, recessive grid, readable 14px text, legend on top."""
    fig.update_layout(
        template="plotly_white", font=FONT, height=height,
        margin=dict(l=60, r=20, t=60 if title else 30, b=50),
        title=dict(text=title, x=0, font=dict(size=16)) if title else None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb",
        hoverlabel=dict(font_size=13),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor="#8a8984")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="#8a8984")
    return fig


def alt_text(text: str) -> None:
    """Visible text description for screen-reader and low-vision users (Streamlit charts have no alt attribute)."""
    st.caption(f"**Chart description:** {text}")


def maybe_table(df: pd.DataFrame, label: str = "Show the data behind this chart") -> None:
    if st.session_state.get("show_tables", False):
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        with st.expander(label):
            st.dataframe(df, width="stretch", hide_index=True)


def class_name(c: str) -> str:
    return CLASS_LABELS[c]


__all__ = ["CLASSES", "CLASS_COLORS", "CLASS_LABELS", "feature_names", "load_tables", "load_mels",
           "mel_hz", "load_wav", "showcase_paths", "source_paths", "style", "alt_text", "maybe_table",
           "class_name", "data_ready", "SR"]
