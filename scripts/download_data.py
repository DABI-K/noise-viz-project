"""Download the two public source datasets into data/raw/.

  * Mini LibriSpeech dev-clean-2 (OpenSLR 31, CC BY 4.0)  -> clean read speech
  * ESC-50 (Piczak 2015, CC BY-NC 3.0, clips from Freesound) -> real wind and background-noise recordings

Usage:  python scripts/download_data.py
Re-running is safe: finished downloads are skipped.
"""
from __future__ import annotations

import ssl
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

try:  # python.org builds on macOS ship without CA certificates; certifi fixes that
    import certifi

    urllib.request.install_opener(urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=certifi.where()))))
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

SOURCES = {
    "librispeech": {
        "urls": [  # OpenSLR mirrors, tried in order
            "https://openslr.elda.org/resources/31/dev-clean-2.tar.gz",
            "https://www.openslr.org/resources/31/dev-clean-2.tar.gz",
            "https://us.openslr.org/resources/31/dev-clean-2.tar.gz",
        ],
        "archive": "dev-clean-2.tar.gz",
        "done_marker": "LibriSpeech",
    },
    "esc50": {
        "urls": ["https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"],
        "archive": "ESC-50-master.zip",
        "done_marker": "ESC-50-master",
    },
}


def _progress(block: int, block_size: int, total: int) -> None:
    if total > 0:
        pct = min(100, block * block_size * 100 // total)
        sys.stdout.write(f"\r  {pct:3d}%")
        sys.stdout.flush()


def fetch(name: str, spec: dict) -> None:
    if (RAW / spec["done_marker"]).exists():
        print(f"[skip] {name} already extracted")
        return
    archive = RAW / spec["archive"]
    if not archive.exists():
        tmp = archive.with_suffix(archive.suffix + ".part")
        for url in spec["urls"]:
            print(f"[download] {name}: {url}")
            try:
                urllib.request.urlretrieve(url, tmp, _progress)
                break
            except OSError as err:
                print(f"\n  failed ({err}); trying next mirror")
        else:
            sys.exit(f"Could not download {name}. Download it manually into {RAW}.")
        tmp.rename(archive)
        print()
    print(f"[extract] {archive.name}")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(RAW)
    else:
        with tarfile.open(archive) as t:
            t.extractall(RAW, filter="data")
    archive.unlink()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, spec in SOURCES.items():
        fetch(name, spec)
    print("Done. Next: python scripts/build_dataset.py")


if __name__ == "__main__":
    main()
