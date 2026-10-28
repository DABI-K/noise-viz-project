# Something Sounds Off: a multi-modality view of audio defects

**DATS 6401 Visualization of Complex Data, mid-semester project (Fall 2026)**
Eungdap Kim · companion to the CSCI 6364 ML project *Something Sounds Off*

An interactive Streamlit app that looks at the same audio four ways (waveforms, spectrogram images, a
38-feature table and feature/class networks) to understand **what wind rumble, mains hum, digital clipping and
background noise look like**, and whether *synthetic* defects are a good stand-in for real ones.

## The narrative

Anyone who records voice audio runs into the same four problems. Each one needs a different fix (high-pass filter,
notch filter, de-clipper, noise reduction), so a tool has to say *which* problem a clip has, not just that it sounds
bad. My CSCI 6364 project trains a classifier on clean speech with **synthetic** defects added and asks whether it
transfers to **real** recordings. Hypothesis H2 says that making the synthetic data more varied ("domain
randomisation") should help.

Before training anything, this app asks four visual questions:

| Tab | Modality | Question | Main finding |
|---|---|---|---|
| 2 | **Time series** | What does each defect do to the signal over time? | Only clipping is visible in the raw waveform (flat tops, twin spikes in the amplitude histogram). The others hide in the pauses between words. |
| 3 | **Images** | Does each defect leave a visual signature in the spectrogram? | Yes: hum is a line at the bottom, wind is a low-frequency haze, noise fills everything, clipping adds highs. Clean LibriSpeech is **empty above 8 kHz**, which is a shortcut a model could cheat with. |
| 4 | **Tabular / multivariate** | Do the 38 proposal features separate the classes? | With narrow data, noise and clipping separate cleanly. Diverse data spreads every class out by about 1.6× and the classes overlap more. That is harder to learn, but it covers more real-world conditions. |
| 5 | **Network** | Which features repeat each other, and which classes look alike? | Features form a few tightly correlated groups. The top-band contrast feature is among the strongest separators, mostly because of the 8 kHz artefact. Wind and noise are the most confusable pair (k-NN neighbour overlap). |

Tab 1 introduces the story and tab 6 documents data sources, method, accessibility and reproducibility.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate    # Python 3.10–3.12
pip install -r requirements.txt
python scripts/download_data.py     # ~770 MB: Mini LibriSpeech dev-clean-2 + ESC-50
python scripts/build_dataset.py     # ~2 min: 1,500 clips -> data/processed/
streamlit run app/streamlit_app.py
pytest -q                           # smoke tests (feature shape, every tab renders)
```

`build_dataset.py` takes `--per-class N` (default 150 per class per strategy), `--seed` (default 6401) and
`--save-audio` (writes every clip as WAV, for training models later).

## Data

| Source | Used for | Licence |
|---|---|---|
| [Mini LibriSpeech dev-clean-2](https://www.openslr.org/31/) | Clean speech, 26 speakers (8 held out as a speaker-disjoint test split) | CC BY 4.0 |
| [ESC-50](https://github.com/karolpiczak/ESC-50) | Real wind (40 clips) and background noise (vacuum, washing machine, engine, airplane, train, rain, hand saw, clock, keyboard) | CC BY-NC 3.0 |
| `noisedx/defects.py` | Synthetic hum (50/60 Hz + harmonics), clipping, simulated room echo | own code |

The ML proposal names VCTK (~11 GB). Mini LibriSpeech (126 MB) keeps the mid-term reproducible in minutes, and only
the loader in `noisedx/audio_io.py` would change to swap VCTK in.

**Narrow vs diverse.** Narrow uses 4 training speakers, one wind and one vacuum-cleaner recording, 60 Hz hum with 3
harmonics at 10 dB SNR, and fixed clipping (+18 dB into a 0.5 ceiling). Diverse uses all 18 training speakers, every
wind file, nine noise categories, random SNR (−5 to 15 dB), 50 or 60 Hz hum with 1–6 harmonics, random clipping, and
synthetic room echo on half the clips. Class counts are identical.

## Accessibility

* Five class colours validated for colour-vision deficiency (worst adjacent ΔE 9.2, deuteranopia). Colour is never the
  only cue: marker shapes, line dashes and text labels repeat it.
* Spectrograms default to *cividis*. Difference maps use a diverging red–blue scale centred on 0 dB.
* Every chart has a written description (alt text) and a table view. A sidebar toggle shows all tables at once.
* Light theme, 14 px+ text, axis titles with units, and plain-English definitions of the jargon.

## Code layout

```
noisedx/            shared package (reused by the final project and the CSCI 6364 classifier)
  config.py         paths, audio settings, class names, accessible colours
  defects.py        hum / clipping / recording mix / reverb generators (narrow & diverse)
  features.py       the 38 proposal features + log-mel spectrograms
  audio_io.py       LibriSpeech / ESC-50 loaders
scripts/            download_data.py, build_dataset.py
app/                streamlit_app.py + views/ (one module per tab) + common.py
tests/              pytest smoke tests
docs/               Guide_How_The_App_Works.html (beginner walkthrough of every file and chart)
```

### Extending to the final project (client-server ML app)

The final project needs a FastAPI/Flask model server and a Streamlit client. The plan:

1. Train the CSCI 6364 classifiers on `data/processed/features.csv` (or `--save-audio` WAVs for a CNN) and save them with joblib.
2. Add `server/main.py` (FastAPI): `POST /predict` takes a WAV, calls `noisedx.features.extract_features`,
   and returns `{label, probabilities, latency_ms}`.
3. Add a tab to this app that uploads or records a clip, calls the endpoint, and shows the prediction with a
   probability bar, the clip's position in the PCA map, and its spectrogram. That covers model behaviour and uncertainty.

Because the feature code lives in `noisedx/`, training, serving and visualisation all use the same function.

## Use of AI tools

As the DATS 6401 AI policy requires, I am disclosing that I used AI tools (Claude, from Anthropic) throughout this
project. Claude helped write code, app text, documentation and the beginner guide in `docs/`, and helped check numbers
against the generated data.

The project itself is mine. The topic, research questions and dataset plan come from my own CSCI 6364 machine-learning
project, *Something Sounds Off*, which I developed with my teammate. I reviewed the files and chose what to keep or
drop, reviewed each cell and chart, and followed the course's weekly schedule for the mid-term. I can explain each part
of the app and the code in my own words.
