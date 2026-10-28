"""Something Sounds Off: a multi-modality look at audio defects.

Run from the project root:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import streamlit as st

from common import data_ready, refresh_cache_if_data_changed
from views import about, feature_space, network, spectrograms, story, waveforms

st.set_page_config(page_title="Something Sounds Off", page_icon="🎙️", layout="wide")

st.title("Something Sounds Off")
st.markdown(
    "#### What do wind, hum, clipping and background noise *look* like, and can fake defects stand in for real ones?"
)

if not data_ready():
    st.error(
        "The processed dataset is missing. From the project folder run:\n\n"
        "```\npython scripts/download_data.py\npython scripts/build_dataset.py\n```"
    )
    st.stop()
refresh_cache_if_data_changed()

with st.sidebar:
    st.header("Display options")
    st.toggle("Always show data tables under charts", key="show_tables",
              help="Every chart has a table view. Turn this on to show them all without clicking.")
    st.markdown(
        "**Accessibility**\n\n"
        "- Class colours are checked for colour-blind readers, and every class also has its own "
        "marker shape or line dash.\n"
        "- Each chart has a written description and a table view.\n"
        "- Spectrogram colour maps are perceptually uniform (default: *cividis*, built for colour-vision deficiency)."
    )

tabs = st.tabs([
    "1 · The story",
    "2 · Time series: waveforms",
    "3 · Images: spectrograms",
    "4 · Tabular: feature space",
    "5 · Network: feature links",
    "6 · Data & method",
])
for tab, view in zip(tabs, [story, waveforms, spectrograms, feature_space, network, about]):
    with tab:
        view.render()
