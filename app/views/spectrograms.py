"""Tab 3 (images): spectrogram montages, class-average images, difference images and frequency profiles."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from common import (CLASS_COLORS, CLASS_LABELS, CLASSES, SR, alt_text, load_mels, load_tables, maybe_table,
                    mel_hz, style)
from noisedx.config import CLASS_DASH, HOP, STRATEGY_COLORS

CMAPS = {"Cividis (colour-blind safe)": "cividis", "Viridis": "viridis", "Magma": "magma", "Grayscale": "gray_r"}


def _mel_axis(fig: go.Figure, **kw) -> None:
    """Label mel bins with real frequencies."""
    hz = mel_hz()
    ticks = [0, 8, 16, 24, 32, 40, 48, 56, 63]
    fig.update_yaxes(tickvals=ticks, ticktext=[f"{hz[i]/1000:.1f}k" if hz[i] >= 1000 else f"{hz[i]:.0f}"
                                               for i in ticks], **kw)


@st.cache_data
def _class_means(strategy: str) -> dict[str, np.ndarray]:
    meta, _ = load_tables()
    ids, mels = load_mels()
    pos = {k: i for i, k in enumerate(ids)}
    out = {}
    for c in CLASSES:
        sel = meta[(meta.strategy == strategy) & (meta.label == c)].id
        out[c] = mels[[pos[i] for i in sel]].mean(axis=0)
    return out


@st.cache_data
def _profiles() -> pd.DataFrame:
    """Time-averaged spectrum per clip, summarised as mean and SD per strategy x class x mel band."""
    meta, _ = load_tables()
    ids, mels = load_mels()
    prof = mels.mean(axis=2)  # (clips, n_mels)
    df = pd.DataFrame(prof, columns=range(prof.shape[1]))
    df["id"] = ids
    df = df.merge(meta[["id", "label", "strategy"]], on="id").melt(
        id_vars=["id", "label", "strategy"], var_name="band", value_name="db")
    g = df.groupby(["strategy", "label", "band"], observed=True).db.agg(["mean", "std"]).reset_index()
    g["hz"] = mel_hz()[g.band.astype(int)]
    return g[g.hz > 0]


def render() -> None:
    st.markdown(
        "**Question:** if we draw each clip as a picture (time across, pitch up, brightness = energy), "
        "does each defect leave a visual signature? This is exactly the input a CNN would see."
    )
    meta, _ = load_tables()
    ids, mels = load_mels()
    pos = {k: i for i, k in enumerate(ids)}

    c1, c2, c3 = st.columns(3)
    strategy = c1.radio("Strategy", ["narrow", "diverse"], horizontal=True, key="sp_strategy", format_func=str.title)
    cmap = CMAPS[c2.selectbox("Colour map", list(CMAPS), key="sp_cmap")]
    n = c3.slider("Examples per class", 2, 6, 4, key="sp_n")

    # --- Montage --------------------------------------------------------------------------------
    fig = make_subplots(rows=5, cols=n, shared_xaxes=True, shared_yaxes=True,
                        horizontal_spacing=0.01, vertical_spacing=0.02)
    times = np.arange(mels.shape[2]) * HOP / SR
    for r, c in enumerate(CLASSES, start=1):
        sel = meta[(meta.strategy == strategy) & (meta.label == c) & (meta.split == "test")].id.head(n).tolist()
        for k, cid in enumerate(sel, start=1):
            fig.add_heatmap(z=mels[pos[cid]], x=times, coloraxis="coloraxis", row=r, col=k,
                            hovertemplate=f"{CLASS_LABELS[c]} · {cid}<br>%{{x:.2f}} s, band %{{y}}<br>"
                                          "%{z:.0f} dB<extra></extra>")
        fig.update_yaxes(title_text=CLASS_LABELS[c].replace(" ", "<br>"), title_font=dict(size=12), row=r, col=1)
    fig.update_layout(coloraxis=dict(colorscale=cmap, colorbar=dict(title="dB", thickness=12), cmin=-80, cmax=0))
    fig.update_xaxes(title_text="Time (s)", row=5)
    _mel_axis(fig)
    st.plotly_chart(style(fig, 820), width="stretch")
    alt_text(f"A grid of {5 * n} spectrograms, one row per class. Clean speech shows bright vertical bursts "
             "(syllables) with dark gaps. Hum adds a bright, perfectly straight band along the bottom. Wind adds a "
             "smeared glow in the lowest frequencies. Background noise brightens the whole image, filling the gaps. "
             "Clipping makes the image brighter and adds energy in the high bands above each syllable.")
    st.divider()

    # --- Class-average images and difference from clean -------------------------------------------
    st.markdown("**Question:** averaging all clips of one class cancels out the words. What is left is the defect itself.")
    means = _class_means(strategy)
    view = st.radio("Show", ["Average spectrogram", "Difference from clean (dB)"], horizontal=True, key="sp_view")
    fig = make_subplots(rows=1, cols=5, shared_yaxes=True, subplot_titles=[CLASS_LABELS[c] for c in CLASSES],
                        horizontal_spacing=0.015)
    for k, c in enumerate(CLASSES, start=1):
        z = means[c] if view.startswith("Average") else means[c] - means["clean"]
        fig.add_heatmap(z=z, x=times, coloraxis="coloraxis", row=1, col=k,
                        hovertemplate=f"{CLASS_LABELS[c]}<br>%{{x:.2f}} s, band %{{y}}<br>%{{z:.1f}} dB<extra></extra>")
    if view.startswith("Average"):
        fig.update_layout(coloraxis=dict(colorscale=cmap, colorbar=dict(title="dB", thickness=12), cmin=-80, cmax=0))
    else:  # diverging: two hues, neutral midpoint at 0 dB
        lim = 30
        fig.update_layout(coloraxis=dict(colorscale="RdBu_r", cmid=0, cmin=-lim, cmax=lim,
                                         colorbar=dict(title="dB vs<br>clean", thickness=12)))
    fig.update_xaxes(title_text="Time (s)")
    _mel_axis(fig, title_text="Frequency (Hz)", col=1)
    st.plotly_chart(style(fig, 360), width="stretch")
    alt_text("Five average spectrograms side by side. With the difference view, red means more energy than clean "
             "speech and blue means less. Hum is red only in a thin strip at the very bottom; wind is red across the "
             "lowest quarter; noise is red almost everywhere; clipping is red in the upper bands.")
    st.divider()

    # --- Frequency profile with spread, narrow vs diverse -------------------------------------------
    st.markdown("**Question:** where in the frequency range does each defect add energy, and how much does that vary "
                "between the narrow and diverse strategies?")
    prof = _profiles()
    mode = st.radio("Compare", ["All classes (one strategy)", "Narrow vs diverse (one class)"], horizontal=True,
                    key="sp_prof_mode")
    fig = go.Figure()
    if mode.startswith("All"):
        for c in CLASSES:
            d = prof[(prof.strategy == strategy) & (prof.label == c)]
            fig.add_scatter(x=d.hz, y=d["mean"], name=CLASS_LABELS[c], mode="lines",
                            line=dict(color=CLASS_COLORS[c], width=2, dash=CLASS_DASH[c]),
                            hovertemplate="%{x:.0f} Hz: %{y:.1f} dB<extra>" + CLASS_LABELS[c] + "</extra>")
        shown = prof[prof.strategy == strategy]
        desc = ("Line chart of average energy per frequency band. Wind and hum lines rise above clean speech below "
                "about 300 Hz; the noise line sits above clean across the whole range; clipping rises in the high range.")
    else:
        c = st.selectbox("Class", CLASSES, index=1, format_func=CLASS_LABELS.get, key="sp_prof_class")
        for s in ["narrow", "diverse"]:
            d = prof[(prof.strategy == s) & (prof.label == c)]
            col = STRATEGY_COLORS[s]
            fig.add_scatter(x=np.r_[d.hz, d.hz[::-1]], y=np.r_[d["mean"] + d["std"], (d["mean"] - d["std"])[::-1]],
                            fill="toself", fillcolor=col, opacity=0.18, line=dict(width=0), hoverinfo="skip",
                            showlegend=False)
            fig.add_scatter(x=d.hz, y=d["mean"], name=f"{s.title()} (mean ± 1 SD)", mode="lines",
                            line=dict(color=col, width=2, dash="solid" if s == "narrow" else "dash"),
                            hovertemplate="%{x:.0f} Hz: %{y:.1f} dB<extra>" + s.title() + "</extra>")
        shown = prof[prof.label == c]
        desc = (f"Two lines for {CLASS_LABELS[c]}, narrow (solid) and diverse (dashed), each with a shaded band of "
                "plus or minus one standard deviation. The diverse band is wider at most frequencies, showing it covers "
                "more recording conditions.")
    fig.update_layout(xaxis_title="Frequency (Hz, log scale)", yaxis_title="Average energy (dB)", xaxis_type="log",
                      xaxis_range=[np.log10(40), np.log10(10_000)], hovermode="x unified")
    st.plotly_chart(style(fig, 400), width="stretch")
    alt_text(desc)
    maybe_table(shown.assign(hz=shown.hz.round(0), mean=shown["mean"].round(2), std=shown["std"].round(2)))
    st.markdown(
        "**What we see:** each defect lives in its own part of the picture. Hum is a line at the bottom, wind is a "
        "low-frequency haze, noise is everywhere and clipping is in the highs. That is why spectrogram images are a "
        "strong input for a CNN. The diverse bands are wider, so a model trained on them sees more of the variety that "
        "real recordings will have."
    )
    st.warning(
        "**Look at the top of the images.** Clean, hum and (partly) wind spectrograms are dark above about 8 kHz, "
        "because LibriSpeech was recorded at 16 kHz. Noise and clipping fill that band. This is a data artefact, not "
        "part of the defect, so it is a shortcut a model could cheat with. Tab 5 shows how it shows up in the features."
    )
