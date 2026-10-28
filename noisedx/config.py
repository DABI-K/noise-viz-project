"""Project-wide constants: paths, audio settings, class names, accessible colors."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
CLIPS = DATA / "clips"
PROCESSED = DATA / "processed"

SR = 22_050          # sample rate (Hz), same as the CSCI 6364 proposal
CLIP_SECONDS = 3.0
CLIP_SAMPLES = int(SR * CLIP_SECONDS)
N_MELS = 64
HOP = 512

CLASSES = ["clean", "wind", "hum", "clipping", "noise"]
STRATEGIES = ["narrow", "diverse"]

CLASS_LABELS = {
    "clean": "Clean",
    "wind": "Wind rumble",
    "hum": "Mains hum",
    "clipping": "Digital clipping",
    "noise": "Background noise",
}

CLASS_HELP = {
    "clean": "Speech with nothing added.",
    "wind": "Real wind recordings (ESC-50) mixed under the speech: energy piles up below ~200 Hz.",
    "hum": "A 50 or 60 Hz sine plus harmonics: thin horizontal lines in the spectrogram.",
    "clipping": "Speech amplified and cut at a ceiling: flat-topped waveform, lower crest factor.",
    "noise": "Real fan, engine, rain and appliance recordings (ESC-50): the spectrum gets flatter.",
}

# Categorical slots validated for colour-vision deficiency (worst adjacent CVD dE 9.2).
# Colour is never the only cue: every chart also uses marker shape, dash or a text label.
CLASS_COLORS = {
    "clean": "#2a78d6",
    "wind": "#1baf7a",
    "hum": "#eb6834",
    "clipping": "#e34948",
    "noise": "#6250d6",
}
CLASS_SYMBOLS = {
    "clean": "circle",
    "wind": "triangle-up",
    "hum": "square",
    "clipping": "x",
    "noise": "diamond",
}
CLASS_DASH = {
    "clean": "solid",
    "wind": "dash",
    "hum": "dot",
    "clipping": "dashdot",
    "noise": "longdash",
}
STRATEGY_COLORS = {"narrow": "#2a78d6", "diverse": "#eb6834"}

# ESC-50 categories used as real defect sources.
ESC_WIND = ["wind"]
ESC_NOISE = ["vacuum_cleaner", "washing_machine", "engine", "airplane", "train",
             "rain", "hand_saw", "clock_tick", "keyboard_typing"]
# Narrow strategy only sees one source category for noise.
ESC_NOISE_NARROW = ["vacuum_cleaner"]
