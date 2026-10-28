"""Tab 1: the narrative and the dataset at a glance."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from common import CLASS_COLORS, CLASS_LABELS, CLASSES, alt_text, load_tables, maybe_table, style
from noisedx.config import CLASS_HELP


def render() -> None:
    meta, _ = load_tables()

    st.markdown(
        """
Anyone who records voice audio, like podcasters, video editors or students recording a class presentation,
runs into the same four problems. Each one needs a **different fix**, so it is not enough to know a clip
"sounds bad". You need to know *which* problem it has.

This app is the visual side of my CSCI 6364 machine-learning project, **Something Sounds Off**. That project
trains a classifier on clean speech with *synthetic* defects added, then asks whether it still works on
*real* recordings. Before training anything, this app looks at the same audio four different ways:
"""
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("**⏱ Time series**  \nThe raw waveform and its loudness over 3 seconds.")
    c2.markdown("**🖼 Images**  \nSpectrograms: the sound drawn as a picture of frequency over time.")
    c3.markdown("**▦ Tabular**  \n38 numeric features per clip, compared across classes.")
    c4.markdown("**⛓ Network**  \nWhich features move together, and which classes look alike.")

    st.subheader("The five classes")
    cols = st.columns(5)
    for col, c in zip(cols, CLASSES):
        col.markdown(
            f"<div style='border-left:6px solid {CLASS_COLORS[c]};padding-left:10px'>"
            f"<b>{CLASS_LABELS[c]}</b><br><span style='font-size:0.9em'>{CLASS_HELP[c]}</span></div>",
            unsafe_allow_html=True,
        )

    st.subheader("Two ways to fake a defect")
    st.markdown(
        """
The ML project's second hypothesis (H2) is that **varied** fake training data transfers to real audio better
than **narrow** fake data. So every clip here was made under one of two strategies with the same class balance:

* **Narrow:** 4 speakers, one wind recording, one vacuum-cleaner recording, fixed hum (60 Hz) and clipping strength.
* **Diverse:** 18 speakers, all 40 wind recordings, nine kinds of background noise, random strength, 50 or 60 Hz hum, and
  simulated room echo on half of the clips.

Eight more speakers are held out as a speaker-disjoint *test* split, as the proposal requires.
Look for the narrow/diverse switch on the next tabs: the diverse clouds are wider, which is the whole point.
"""
    )

    counts = meta.groupby(["strategy", "label"], observed=True).size().reset_index(name="clips")
    fig = go.Figure()
    for c in CLASSES:
        d = counts[counts.label == c].sort_values("strategy", key=lambda s: s.map({"narrow": 0, "diverse": 1}))
        fig.add_bar(x=d.strategy.str.title(), y=d.clips, name=CLASS_LABELS[c],
                    marker_color=CLASS_COLORS[c], marker_line=dict(color="#fcfcfb", width=2),
                    hovertemplate=f"{CLASS_LABELS[c]}<br>%{{x}}: %{{y}} clips<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.25, yaxis_title="Clips", xaxis_title="Augmentation strategy")
    st.plotly_chart(style(fig, 320, "Clips per class and strategy (balanced by design)"), width="stretch")
    alt_text(f"Grouped bar chart. Both strategies have exactly {counts.clips.iloc[0]} clips for each of the "
             "five classes, so no class can win just by being common.")
    maybe_table(counts.rename(columns={"label": "class"}))

    st.info(
        "**How to read this app.** Each tab opens with the question it answers and closes with what the chart shows. "
        "Hover over any mark for exact values, and use the widgets above each chart to change the view."
    )
