"""Smoke tests: the features have the right shape and every tab of the app renders without errors."""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_feature_vector_shape():
    import sys
    sys.path.insert(0, str(ROOT))
    from noisedx.config import CLIP_SAMPLES
    from noisedx.features import extract_features, feature_names
    y = np.random.default_rng(0).standard_normal(CLIP_SAMPLES).astype(np.float32) * 0.1
    assert extract_features(y).shape == (len(feature_names()),) == (38,)


@pytest.mark.skipif(not (ROOT / "data/processed/features.csv").exists(), reason="dataset not built")
def test_app_renders():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=180).run()
    assert not at.exception, at.exception
