# Something Sounds Off: a multi-modality view of audio defects

DATS 6401 Visualization of Complex Data, mid-semester project (Fall 2026)
Eungdap Kim, companion to my CSCI 6364 ML project *Something Sounds Off*

This is an interactive Streamlit app that looks at the same audio four ways: waveforms, spectrogram images, a table
of 38 features, and feature and class networks. I wanted to see what wind rumble, mains hum, digital clipping and
background noise look like, and whether synthetic defects can stand in for real ones.

## The narrative

People who record voice audio keep running into the same four problems. Each needs a different fix (high-pass filter,
notch filter, de-clipper, noise reduction), so a tool has to say which problem a clip has, not just that it sounds
bad. My CSCI 6364 project trains a classifier on clean speech with synthetic defects added and asks whether it works
on real recordings. Hypothesis H2 says that more varied synthetic data ("domain randomisation") should help.

Before training anything, this app asks four visual questions:

| Tab | Modality | Question | Main finding |
|---|---|---|---|
| 2 | Time series | What does each defect do to the signal over time? | Only clipping shows in the raw waveform (flat tops, twin spikes in the amplitude histogram). The others hide in the pauses between words. |
| 3 | Images | Does each defect leave a visual signature in the spectrogram? | Yes. Hum is a line at the bottom, wind is a low-frequency haze, noise fills everything, and clipping adds highs. Clean LibriSpeech is empty above 8 kHz, which a model could use as a shortcut. |
| 4 | Tabular / multivariate | Do the 38 proposal features separate the classes? | With narrow data, noise and clipping separate cleanly. Diverse data spreads every class out by about 1.6 times, so the classes overlap more. That is harder to learn but covers more real-world conditions. |
| 5 | Network | Which features repeat each other, and which classes look alike? | Features form a few tightly correlated groups. The top-band contrast feature is one of the strongest separators, mostly because of the 8 kHz gap. Wind and noise are the most confusable pair (k-NN neighbour overlap). |

Tab 1 introduces the story, and tab 6 covers data sources, method, accessibility and reproducibility.

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

The ML proposal names VCTK (about 11 GB). I used Mini LibriSpeech (126 MB) so the mid-term can be reproduced in
minutes. Swapping VCTK in would only change the loader in `noisedx/audio_io.py`.

### Narrow vs diverse

Narrow uses 4 training speakers, one wind and one vacuum-cleaner recording, 60 Hz hum with 3 harmonics at 10 dB SNR,
and fixed clipping (+18 dB into a 0.5 ceiling). Diverse uses all 18 training speakers, every wind file, nine noise
categories, random SNR (−5 to 15 dB), 50 or 60 Hz hum with 1 to 6 harmonics, random clipping, and synthetic room echo
on half the clips. Class counts are identical.

## Accessibility

* The five class colours were checked for colour-vision deficiency (worst adjacent ΔE 9.2, deuteranopia). Colour is
  never the only cue: marker shapes, line dashes and text labels repeat it.
* Spectrograms default to *cividis*. Difference maps use a diverging red–blue scale centred on 0 dB.
* Every chart has a written description (alt text) and a table view, and a sidebar toggle shows all the tables at once.
* The theme is light, text is 14 px or larger, axis titles have units, and jargon is defined in plain English.

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
```

### Extending to the final project (client-server ML app)

The final project needs a FastAPI or Flask model server and a Streamlit client. My plan:

1. Train the CSCI 6364 classifiers on `data/processed/features.csv` (or `--save-audio` WAVs for a CNN) and save them with joblib.
2. Add `server/main.py` (FastAPI) with `POST /predict`. It takes a WAV, calls `noisedx.features.extract_features`,
   and returns `{label, probabilities, latency_ms}`.
3. Add a tab to this app that uploads or records a clip, calls the endpoint, and shows the prediction with a
   probability bar, the clip's position in the PCA map, and its spectrogram. That covers model behaviour and uncertainty.

The feature code lives in `noisedx/`, so training, serving and visualisation all use the same function.

## Course materials and references

I referred to the DATS 6401 lecture materials while building this app, and I used and modified code from them. The
data sources and licences are listed in the Data section above.

## Use of AI tools

As the DATS 6401 AI policy requires, I'm disclosing that I used AI tools (Claude, from Anthropic) throughout this
project. Claude helped write code, app text and documentation, and helped check numbers against the generated data.

The topic, research questions and dataset plan come from my own CSCI 6364 machine-learning project, which I'm
developing with my teammate. I went through the files and decided what to keep or drop, reviewed each cell and chart,
and followed the course's weekly schedule for the mid-term. I can explain each part of the app and the code in my own
words.
