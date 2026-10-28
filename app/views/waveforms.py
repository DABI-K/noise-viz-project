"""Tab 2 (time series): waveforms, loudness envelopes, amplitude histograms and a live defect lab."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st
from plotly.subplots import make_subplots

from common import (CLASS_COLORS, CLASS_LABELS, CLASSES, SR, alt_text, load_wav, maybe_table,
                    showcase_paths, source_paths, style)
from noisedx import defects
from noisedx.features import frame_envelope


def _wav_bytes(y: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _crest(y: np.ndarray) -> float:
    return float(np.max(np.abs(y)) / (np.sqrt(np.mean(y ** 2)) + 1e-12))


def render() -> None:
    st.markdown("**Question:** what does each defect do to the shape of the signal over time?")

    c1, c2, c3 = st.columns([1, 1, 2])
    strategy = c1.radio("Strategy", ["narrow", "diverse"], horizontal=True, key="wf_strategy",
                        format_func=str.title)
    n_examples = len(showcase_paths(strategy, "clean"))
    example = c2.selectbox("Example clip", list(range(1, n_examples + 1)), key="wf_example")
    clips = {c: load_wav(str(showcase_paths(strategy, c)[example - 1])) for c in CLASSES}
    # Default zoom: the loudest 50 ms of the clean clip, where clipping is easiest to see.
    peak = int(np.argmax(np.convolve(clips["clean"] ** 2, np.ones(SR // 20), "valid"))) / SR
    start = round(min(peak, 2.95), 2)
    window = c3.slider("Zoom window (seconds)", 0.0, 3.0, (start, round(start + 0.05, 2)), step=0.01,
                       help="Keep it about 50 ms wide. Move it onto a voiced part to see flat-topped clipping, "
                            "or into a pause to see hum cycles and noise.")

    # --- Small multiples: full 3 s waveform (left) and zoomed detail (right) --------------------
    fig = make_subplots(rows=5, cols=2, shared_xaxes="columns", shared_yaxes=True,
                        column_widths=[0.62, 0.38], horizontal_spacing=0.05, vertical_spacing=0.03,
                        column_titles=["Whole clip (3 s)", f"Zoom: {window[0]:.2f}–{window[1]:.2f} s"])
    t = np.arange(len(clips["clean"])) / SR
    step = 15  # plot every 15th sample on the overview; full resolution in the zoom
    lo, hi = int(window[0] * SR), max(int(window[1] * SR), int(window[0] * SR) + 2)
    for r, c in enumerate(CLASSES, start=1):
        y = clips[c]
        fig.add_scatter(x=t[::step], y=y[::step], row=r, col=1, mode="lines", name=CLASS_LABELS[c],
                        line=dict(color=CLASS_COLORS[c], width=1), showlegend=False,
                        hovertemplate="%{x:.3f} s<br>amplitude %{y:.2f}<extra>" + CLASS_LABELS[c] + "</extra>")
        fig.add_scatter(x=t[lo:hi], y=y[lo:hi], row=r, col=2, mode="lines", showlegend=False,
                        line=dict(color=CLASS_COLORS[c], width=2),
                        hovertemplate="%{x:.4f} s<br>amplitude %{y:.3f}<extra>" + CLASS_LABELS[c] + "</extra>")
        fig.add_vrect(x0=window[0], x1=window[1], row=r, col=1, fillcolor="#8a8984", opacity=0.15, line_width=0)
        fig.update_yaxes(title_text=CLASS_LABELS[c].replace(" ", "<br>"), row=r, col=1, range=[-1, 1],
                         title_font=dict(size=12))
    fig.update_xaxes(title_text="Time (s)", row=5)
    st.plotly_chart(style(fig, 760), width="stretch")
    alt_text("Five stacked waveforms of the same sentence, one per class, each with a zoomed view on the right. "
             "Clean speech has sharp peaks with quiet gaps between words. Hum fills the gaps with a steady wave "
             "(one cycle every 17–20 ms in the zoom). Clipping looks like a solid block whose tops are cut flat. "
             "Wind and background noise fill the gaps with irregular energy.")

    cols = st.columns(5)
    for col, c in zip(cols, CLASSES):
        col.markdown(f"**{CLASS_LABELS[c]}**")
        col.audio(_wav_bytes(clips[c]), format="audio/wav")

    st.markdown(
        "**What we see:** clipping is the only defect you can spot from the waveform alone, because the peaks hit a "
        "flat ceiling. Hum, wind and noise mostly show up in the *pauses between words*, where clean speech is silent."
    )
    st.divider()

    # --- Loudness envelope over time --------------------------------------------------------------
    st.markdown("**Question:** how does loudness change through the clip? Do the pauses stay quiet?")
    rows = []
    fig = go.Figure()
    for c in CLASSES:
        env, times = frame_envelope(clips[c])
        rows.append(pd.DataFrame({"class": CLASS_LABELS[c], "time_s": times.round(3), "rms_db": env.round(1)}))
        fig.add_scatter(x=times, y=env, mode="lines", name=CLASS_LABELS[c],
                        line=dict(color=CLASS_COLORS[c], width=2, dash=_dash(c)),
                        hovertemplate="%{x:.2f} s: %{y:.1f} dB<extra>" + CLASS_LABELS[c] + "</extra>")
    fig.update_layout(xaxis_title="Time (s)", yaxis_title="Loudness (RMS, dB)", hovermode="x unified")
    st.plotly_chart(style(fig, 380), width="stretch")
    alt_text("Line chart of loudness in decibels over the 3-second clip, one line per class with different dash "
             "styles. In pauses the clean line drops lowest. Every defect raises that floor, and clipping keeps the "
             "whole line highest because the speech was amplified before being cut.")
    maybe_table(pd.concat(rows))
    st.divider()

    # --- Amplitude histogram ---------------------------------------------------------------------
    st.markdown("**Question:** which sample values are most common? (A histogram of every sample in the clip.)")
    fig = make_subplots(rows=1, cols=5, shared_yaxes=True, subplot_titles=[CLASS_LABELS[c] for c in CLASSES],
                        horizontal_spacing=0.02)
    hist_rows = []
    bins = np.linspace(-1, 1, 81)
    for i, c in enumerate(CLASSES, start=1):
        counts, _ = np.histogram(clips[c], bins=bins)
        centers = (bins[:-1] + bins[1:]) / 2
        fig.add_bar(x=centers, y=counts + 1, row=1, col=i, marker_color=CLASS_COLORS[c], showlegend=False,
                    hovertemplate="value %{x:.2f}<br>%{y} samples<extra>" + CLASS_LABELS[c] + "</extra>")
        hist_rows.append(pd.DataFrame({"class": CLASS_LABELS[c], "bin_center": centers.round(3), "samples": counts}))
    fig.update_yaxes(type="log", title_text="Samples (log scale)", col=1)
    fig.update_xaxes(title_text="Amplitude", range=[-1, 1])
    fig.update_layout(bargap=0.05)
    st.plotly_chart(style(fig, 320), width="stretch")
    alt_text("Five histograms of sample amplitude on a log scale. Most show a narrow peak at zero that tapers "
             "smoothly. The clipping histogram instead has two tall spikes at its ceiling on both sides, where "
             "thousands of samples were cut to the same value.")
    maybe_table(pd.concat(hist_rows))
    st.markdown("**What we see:** the twin spikes are why *crest factor* (peak ÷ average level) is in the feature set. "
                "Clipping drags it down (see the number in the lab below).")
    st.divider()

    _defect_lab()


def _dash(c: str) -> str:
    from noisedx.config import CLASS_DASH
    return CLASS_DASH[c]


def _defect_lab() -> None:
    st.subheader("Defect lab: make your own defect")
    st.markdown("Pick a held-out speech clip, choose a defect and drag its strength. Everything is recomputed live.")
    speech_files = source_paths("speech")
    bg_files = {p.stem.removeprefix("bg_").replace("_", " "): p for p in source_paths("bg")}

    c1, c2, c3 = st.columns(3)
    s_idx = c1.selectbox("Speech clip", range(len(speech_files)), format_func=lambda i: f"Speaker {i + 1}")
    kind = c2.selectbox("Defect", ["hum", "clipping", "wind / noise recording"])
    speech = load_wav(str(speech_files[s_idx]))

    if kind == "hum":
        f0 = c3.radio("Mains frequency", [50, 60], horizontal=True, format_func=lambda f: f"{f} Hz")
        snr = st.slider("Speech-to-hum ratio (dB). Lower = louder hum", -5, 30, 10)
        n_h = st.slider("Number of harmonics", 1, 8, 3)
        tt = np.arange(len(speech)) / SR
        hum = sum((0.6 ** k) * np.sin(2 * np.pi * f0 * (k + 1) * tt) for k in range(n_h))
        y = defects.mix_at_snr(speech, hum, snr)
    elif kind == "clipping":
        gain = st.slider("Gain before the ceiling (dB). Higher = more clipping", 0, 30, 12)
        x = defects.peak_normalize(speech, 0.5) * 10 ** (gain / 20)
        y = np.clip(x, -0.5, 0.5)
    else:
        name = c3.selectbox("Background recording", list(bg_files))
        snr = st.slider("Speech-to-background ratio (dB). Lower = louder background", -10, 30, 5)
        y = defects.mix_at_snr(speech, load_wav(str(bg_files[name])), snr)
    y = defects.peak_normalize(y)

    t = np.arange(len(y)) / SR
    fig = go.Figure()
    fig.add_scatter(x=t[::10], y=speech[::10], name="Original speech", mode="lines",
                    line=dict(color="#8a8984", width=1))
    fig.add_scatter(x=t[::10], y=y[::10], name="With defect", mode="lines",
                    line=dict(color=CLASS_COLORS["clipping" if kind == "clipping" else "hum" if kind == "hum" else "noise"],
                              width=1), opacity=0.85)
    fig.update_layout(xaxis_title="Time (s)", yaxis_title="Amplitude", yaxis_range=[-1, 1])
    st.plotly_chart(style(fig, 300), width="stretch")
    alt_text("The original speech waveform in gray with the defected version drawn on top in colour, "
             "updated as you move the sliders.")

    m1, m2, a1, a2 = st.columns([1, 1, 2, 2])
    m1.metric("Crest factor, original", f"{_crest(speech):.2f}")
    m2.metric("Crest factor, with defect", f"{_crest(y):.2f}", delta=f"{_crest(y) - _crest(speech):+.2f}",
              delta_color="off")
    a1.markdown("Original")
    a1.audio(_wav_bytes(speech), format="audio/wav")
    a2.markdown("With defect")
    a2.audio(_wav_bytes(y), format="audio/wav")
