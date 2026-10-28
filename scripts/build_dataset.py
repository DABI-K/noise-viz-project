"""Generate the synthetic defect dataset used by the app.

For each augmentation strategy (narrow, diverse) and each class (clean, wind, hum,
clipping, noise) we make --per-class 3-second clips from LibriSpeech speech and
ESC-50 backgrounds, then save:

  data/processed/metadata.csv   one row per clip (id, class, strategy, split, speaker, defect params)
  data/processed/features.csv   the 38 proposal features per clip
  data/processed/mels.npz       64-band log-mel spectrogram per clip (float16)
  data/clips/showcase/...       a few WAVs per class/strategy for listening in the app
  data/clips/sources/...        clean speech + background WAVs for the live "defect lab"

Usage:  python scripts/build_dataset.py [--per-class 120] [--seed 6401] [--save-audio]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from noisedx import defects  # noqa: E402
from noisedx.audio_io import esc50_index, librispeech_index, load, loudest_window, save_wav  # noqa: E402
from noisedx.config import (CLASSES, CLIPS, ESC_NOISE, ESC_NOISE_NARROW, ESC_WIND,  # noqa: E402
                            PROCESSED, STRATEGIES)
from noisedx.features import extract_features, feature_names, log_mel  # noqa: E402

N_TEST_SPEAKERS = 8      # speakers held out as a "test" split (speaker-disjoint, as in the proposal)
N_NARROW_SPEAKERS = 4    # the narrow strategy only ever hears this many training speakers
SHOWCASE_PER_CELL = 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=120)
    ap.add_argument("--seed", type=int, default=6401)
    ap.add_argument("--save-audio", action="store_true", help="save every clip as WAV (for model training)")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    libri = librispeech_index()
    speakers = sorted(libri["speaker"].unique())
    rng.shuffle(speakers)
    test_spk = set(speakers[:N_TEST_SPEAKERS])
    train_spk = [s for s in speakers if s not in test_spk]
    narrow_spk = set(train_spk[:N_NARROW_SPEAKERS])

    esc = esc50_index()
    wind_all = esc[esc.category.isin(ESC_WIND)].path.tolist()
    noise_all = esc[esc.category.isin(ESC_NOISE)]
    noise_narrow = noise_all[noise_all.category.isin(ESC_NOISE_NARROW)].path.tolist()[:1]
    wind_narrow = wind_all[:1]
    cache: dict[str, np.ndarray] = {}

    def bg(path: str) -> np.ndarray:
        if path not in cache:
            cache[path] = load(path)
        return cache[path]

    # Source files for the live defect lab in the app.
    src_dir = CLIPS / "sources"
    for i, row in enumerate(libri[libri.speaker.isin(test_spk)].sample(3, random_state=args.seed).itertuples()):
        save_wav(src_dir / f"speech_{i}.wav", defects.peak_normalize(loudest_window(load(row.path))))
    for cat in ["wind", "vacuum_cleaner", "rain", "engine"]:
        save_wav(src_dir / f"bg_{cat}.wav", loudest_window(bg(esc[esc.category == cat].path.iloc[0])))

    # Showcase clips reuse the same test utterances across every class and strategy,
    # so the app can compare one sentence with each defect side by side.
    showcase_utts = libri[libri.speaker.isin(test_spk)].sample(SHOWCASE_PER_CELL, random_state=args.seed + 1)

    meta, feats, mels, ids = [], [], [], []
    n_split = {"train": 0.8, "test": 0.2}
    for strategy in STRATEGIES:
        for cls in CLASSES:
            for k in range(args.per_class):
                split = "test" if k < int(args.per_class * n_split["test"]) else "train"
                if split == "test":
                    pool = libri[libri.speaker.isin(test_spk)]
                elif strategy == "narrow":
                    pool = libri[libri.speaker.isin(narrow_spk)]
                else:
                    pool = libri[libri.speaker.isin(train_spk)]
                is_showcase = split == "test" and k < SHOWCASE_PER_CELL
                utt = showcase_utts.iloc[k] if is_showcase else pool.iloc[int(rng.integers(len(pool)))]
                speech = defects.peak_normalize(loudest_window(load(utt.path)))
                params: dict = {}
                source = ""
                if cls == "wind":
                    source = wind_narrow[0] if strategy == "narrow" else str(rng.choice(wind_all))
                    y, params = defects.add_recording(speech, loudest_window(bg(source)), rng, strategy)
                elif cls == "noise":
                    source = noise_narrow[0] if strategy == "narrow" else str(rng.choice(noise_all.path))
                    y, params = defects.add_recording(speech, loudest_window(bg(source)), rng, strategy)
                elif cls == "hum":
                    y, params = defects.add_hum(speech, rng, strategy)
                elif cls == "clipping":
                    y, params = defects.add_clipping(speech, rng, strategy)
                else:
                    y = speech
                if strategy == "diverse" and rng.random() < 0.5:
                    y, rv = defects.add_reverb(y, rng)
                    params.update(rv)
                y = defects.peak_normalize(y)  # equal peak level so loudness alone gives nothing away

                cid = f"{strategy[0]}-{cls}-{k:04d}"
                ids.append(cid)
                meta.append({"id": cid, "label": cls, "strategy": strategy, "split": split,
                             "speaker": utt.speaker, "utterance": utt.utt,
                             "source": Path(source).stem if source else "", **params})
                feats.append(extract_features(y))
                mels.append(log_mel(y).astype(np.float16))
                if args.save_audio or is_showcase:
                    sub = "all" if args.save_audio else "showcase"
                    save_wav(CLIPS / sub / strategy / cls / f"{cid}.wav", y)
            print(f"  {strategy:8s} {cls:9s} {args.per_class} clips")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(meta).to_csv(PROCESSED / "metadata.csv", index=False)
    f = pd.DataFrame(np.vstack(feats), columns=feature_names())
    f.insert(0, "id", ids)
    f.to_csv(PROCESSED / "features.csv", index=False)
    np.savez_compressed(PROCESSED / "mels.npz", ids=np.array(ids), mels=np.stack(mels))
    print(f"Saved {len(ids)} clips to {PROCESSED}")


if __name__ == "__main__":
    main()
