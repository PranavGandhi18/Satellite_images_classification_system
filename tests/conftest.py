import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tile_service.config import Settings
from tile_service.model import RawOutput

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / 'be-mlsys-assignment-dataset'
CLASSES = ['AnnualCrop', 'Forest', 'Highway', 'Industrial', 'Residential', 'River', 'SeaLake']


def png_bytes(arr: np.ndarray, fmt: str = 'PNG') -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


def random_tile(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (64, 64, 3), dtype=np.uint8)


class FakeModel:
    """Same interface as OnnxTileModel, but deterministic and instant: the class is (mean pixel value) mod 7,
    and the embedding is that class's reference vector (so ood_score is 0)."""
    version = 'fake-v1'
    classes = CLASSES
    temperature = 1.0
    policy = {'version': 'p-test', 'min_confidence': 0.6, 'ood_threshold': 0.5}
    references = np.eye(7, 8, dtype=np.float32)
    manifest = {'model_version': 'fake-v1', 'classes': CLASSES, 'onnx': {'sha256': 'fake'}}

    def __init__(self):
        self.fail = False
        self.calls = 0

    def infer(self, pixels):
        self.calls += 1
        if self.fail:
            raise RuntimeError('simulated inference failure')
        k = int(pixels.mean()) % 7
        logits = np.zeros(7, dtype=np.float32); logits[k] = 5.0
        return RawOutput(logits=logits, embedding=self.references[k].copy())

    def self_test(self):
        return {'passed': True, 'tiles': []}


@pytest.fixture
def settings(tmp_path):
    return Settings(model_dir=tmp_path / 'no-model', data_dir=tmp_path / 'data', ort_threads=1)


@pytest.fixture
def fake_model():
    return FakeModel()
