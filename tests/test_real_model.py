"""Checks against the real trained artifact in models/active (skipped until `python training/train.py` has run)."""
import csv
import shutil
import subprocess
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tile_service.api import create_app
from tile_service.config import Settings
from tile_service.model import ArtifactError, OnnxTileModel

from .conftest import DATASET, ROOT, png_bytes, random_tile

ACTIVE = ROOT / 'models' / 'active'
pytestmark = pytest.mark.skipif(not (ACTIVE / 'manifest.json').exists(), reason='no trained model in models/active')


def test_artifact_verifies_and_passes_its_self_test():
    m = OnnxTileModel(ACTIVE, threads=2)
    st = m.self_test()
    assert st['passed'], st
    assert len(st['tiles']) == len(m.classes)


def test_tampered_artifact_is_refused(tmp_path):
    copy = tmp_path / 'artifact'
    shutil.copytree(ACTIVE.resolve(), copy)
    with open(copy / 'reference_embeddings.npy', 'r+b') as f:
        f.seek(-1, 2); b = f.read(1); f.seek(-1, 2); f.write(bytes([b[0] ^ 0xFF]))
    with pytest.raises(ArtifactError, match='checksum'):
        OnnxTileModel(copy)


@pytest.mark.skipif(not DATASET.exists(), reason='dataset not present')
def test_end_to_end_on_one_eval_tile_per_class(tmp_path):
    first = {}
    with open(DATASET / 'eval_labels.csv') as f:
        for row in csv.DictReader(f):
            first.setdefault(row['true_label'], row['filename'])
    with TestClient(create_app(Settings(model_dir=ACTIVE, data_dir=tmp_path, ort_threads=2))) as c:
        results = {}
        for label, name in first.items():
            r = c.post('/v1/tiles', files={'file': (name, (DATASET / 'eval_set' / name).read_bytes(), 'image/png')})
            assert r.status_code == 201, r.text
            results[label] = r.json()
        noise = c.post('/v1/tiles', files={'file': ('noise.png', png_bytes(random_tile(7)), 'image/png')}).json()
    correct = sum(r['label'] == label for label, r in results.items())
    assert correct >= 6, {k: v['label'] for k, v in results.items()}
    assert all(r['status'] in ('confident', 'needs_review', 'unfamiliar') for r in results.values())
    assert noise['ood_score'] > np.median([r['ood_score'] for r in results.values()])  # pure noise looks unfamiliar


def test_runtime_imports_no_ml_or_network_client_libraries():
    heavy = ['torch', 'timm', 'huggingface_hub', 'transformers', 'requests', 'httpx', 'urllib3']
    code = f'import sys, tile_service.api, tile_service.model; print([m for m in {heavy!r} if m in sys.modules])'
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    assert out == '[]'
