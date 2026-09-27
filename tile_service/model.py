"""Step 5 of the tile flow: load a versioned model artifact and run it with ONNX Runtime on CPU.

On load, the artifact is checked before it is trusted:
  * model.onnx and reference_embeddings.npy must match the SHA-256 recorded in manifest.json;
  * the golden self-test tiles must reproduce their recorded probabilities (catches a corrupt file, a runtime
    that behaves differently on this machine, or a preprocessing mismatch).
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


class ArtifactError(RuntimeError):
    """The model artifact is missing, corrupt, or failed its self-test."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class RawOutput:
    logits: np.ndarray     # (n_classes,)
    embedding: np.ndarray  # (embedding_dim,)


class OnnxTileModel:
    def __init__(self, model_dir: Path, threads: int = 4):
        self.dir = Path(model_dir).resolve()
        manifest_path = self.dir / 'manifest.json'
        if not manifest_path.exists():
            raise ArtifactError(f'no manifest.json in {self.dir} (train one with: python training/train.py)')
        self.manifest = json.loads(manifest_path.read_text())
        m = self.manifest
        onnx_path, ref_path = self.dir / m['onnx']['file'], self.dir / m['ood']['reference_file']
        for path, expected in ((onnx_path, m['onnx']['sha256']), (ref_path, m['ood']['reference_sha256'])):
            if not path.exists():
                raise ArtifactError(f'missing artifact file {path.name}')
            if _sha256(path) != expected:
                raise ArtifactError(f'checksum mismatch for {path.name}: the artifact is corrupt or was modified')

        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        self.session = ort.InferenceSession(str(onnx_path), so, providers=['CPUExecutionProvider'])
        self.references = np.load(ref_path).astype(np.float32)  # L2-normalised training embeddings
        self.version: str = m['model_version']
        self.classes: list[str] = m['classes']
        self.temperature: float = float(m['calibration']['temperature'])
        self.policy: dict = m['policy']

    def infer(self, pixels: np.ndarray) -> RawOutput:
        """pixels: (64, 64, 3) uint8 RGB. Preprocessing (resize + normalisation) happens inside the ONNX graph."""
        x = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None], dtype=np.float32)
        logits, embedding = self.session.run(['logits', 'embedding'], {'tile': x})
        return RawOutput(logits=logits[0], embedding=embedding[0])

    def self_test(self) -> dict:
        """Re-classify the golden tiles shipped in the artifact and compare with the recorded expectations."""
        st = self.manifest['selftest']
        results = []
        for t in st['tiles']:
            pixels = np.asarray(Image.open(self.dir / t['file']).convert('RGB'))
            z = self.infer(pixels).logits / self.temperature
            p = np.exp(z - z.max()); p /= p.sum()
            max_diff = float(np.abs(p - np.array(t['expected_probabilities'])).max())
            label = self.classes[int(p.argmax())]
            results.append(dict(tile=t['file'], expected=t['expected_label'], got=label, max_prob_diff=max_diff,
                                ok=label == t['expected_label'] and max_diff <= st['tolerance']))
        return dict(passed=all(r['ok'] for r in results), tiles=results)
