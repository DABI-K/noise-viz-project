"""Tab 6: data sources, method, accessibility choices, reproducibility and AI-use note."""
from __future__ import annotations

import streamlit as st

from common import load_tables


def render() -> None:
    meta, feats = load_tables()
    st.subheader("Data sources")
    st.markdown(
        """
| Source | What it gives | Licence |
|---|---|---|
| [Mini LibriSpeech, dev-clean-2](https://www.openslr.org/31/) (OpenSLR 31) | Clean read English speech, 26 speakers | CC BY 4.0 |
| [ESC-50](https://github.com/karolpiczak/ESC-50) (Piczak, 2015) | Real wind, appliance, engine, rain and other recordings (originally from Freesound) | CC BY-NC 3.0 |
| Our own code (`noisedx/defects.py`) | Mains hum (50/60 Hz + harmonics), digital clipping, simulated room echo | - |

The CSCI 6364 proposal names VCTK for speech. I used Mini LibriSpeech here because it is 126 MB instead of about 11 GB,
so anyone can reproduce the app in a few minutes. The pipeline is the same, so swapping in VCTK later only changes the loader.
"""
    )
    st.subheader("Method")
    st.markdown(
        """
1. Cut the loudest 3-second window from each utterance, resample to 22,050 Hz mono and peak-normalise it.
2. Add one defect per clip (none for *clean*) under the **narrow** or **diverse** strategy, then peak-normalise again so
   loudness alone gives nothing away.
3. Hold out 8 speakers as a speaker-disjoint test split (20% of clips).
4. Extract the proposal's 38 features with librosa (13 MFCC means + 13 variances, 7 spectral-contrast bands, centroid,
   roll-off, flatness, zero-crossing rate, crest factor) and a 64-band log-mel spectrogram.
"""
    )
    st.subheader("Accessibility choices")
    st.markdown(
        """
* **Colour.** The five class colours were run through a colour-vision-deficiency check (worst adjacent ΔE 9.2 under
  deuteranopia). Colour is never the only cue: scatter plots also use marker shape, line charts use dash style, and
  small multiples carry text labels.
* **Colour maps.** Spectrograms default to *cividis*, which was designed for colour-blind readers and is perceptually
  uniform. Difference images use a diverging red–blue map with a neutral midpoint at 0 dB.
* **Text.** Every chart has a written description (shown as alt text under it) and a table view. A sidebar switch
  shows all tables at once for screen-reader users.
* **Labels.** Every chart has axis titles with units, 14 px or larger text, and plain-English definitions of the jargon.
"""
    )
    st.subheader("Reproduce it")
    st.code(
        "python -m venv .venv && source .venv/bin/activate\n"
        "pip install -r requirements.txt\n"
        "python scripts/download_data.py      # ~770 MB download\n"
        "python scripts/build_dataset.py      # ~2 minutes, 1,500 clips\n"
        "streamlit run app/streamlit_app.py",
        language="bash",
    )
    st.caption(f"Loaded {len(meta):,} clips × {feats.shape[1] - 1} features.")
    st.download_button("Download the feature table (CSV)", meta.merge(feats, on="id").to_csv(index=False),
                       "something_sounds_off_features.csv", "text/csv")
    st.subheader("Use of AI tools")
    st.markdown(
        "As the course AI policy requires: Claude (Anthropic) was used to help write code and text for this app. "
        "The README describes exactly how."
    )
