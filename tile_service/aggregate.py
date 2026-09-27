"""Level 3 of "querying the results": class counts corrected for the classifier's known confusions.

Raw counts of predicted labels are biased: if Highway tiles are sometimes called River, River is over-counted.
With M[i, j] = P(predicted = j | true = i), measured on labelled held-out tiles, the expected share of predicted
labels is  p_pred = M^T @ p_true.  Solving that for p_true ("adjusted classify-and-count") gives the corrected class
shares. Using P(predicted | true) rather than P(true | predicted) matters: it does not depend on the class mix of the
labelled set (ours is balanced, 30 per class), only on the per-class error rates, so it stays valid when the real
class mix is very different.

Uncertainty: a bootstrap that resamples both the calibration matrix (Dirichlet per true-class row, from its counts)
and the observed labels (multinomial), and reports the 2.5-97.5 percentile band of the corrected shares.
"""
import json
from pathlib import Path

import numpy as np

SMOOTHING = 0.5  # pseudo-count per cell; keeps rows with no errors from being treated as perfectly error-free


def load_calibration(model_dir: Path, model_version: str, classes: list[str]) -> dict | None:
    """models/<version>/aggregate_calibration.json, written by training/evaluate.py (not part of the checksummed artifact)."""
    path = Path(model_dir) / 'aggregate_calibration.json'
    if not path.exists():
        return None
    cal = json.loads(path.read_text())
    if cal.get('model_version') != model_version or cal.get('classes') != classes:
        return None  # measured for a different model: using it would be wrong
    return cal


def _solve(M: np.ndarray, p_pred: np.ndarray) -> np.ndarray:
    p, *_ = np.linalg.lstsq(M.T, p_pred, rcond=None)
    p = np.clip(p, 0, None)
    s = p.sum()
    return p / s if s > 0 else np.full_like(p, 1 / len(p))


def corrected_shares(counts: np.ndarray, confusion_counts: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    counts = np.asarray(counts, dtype=float)
    C = np.asarray(confusion_counts, dtype=float) + SMOOTHING
    n = counts.sum()
    if n == 0:
        return dict(share=[0.0] * len(counts), low=[0.0] * len(counts), high=[0.0] * len(counts))
    M = C / C.sum(1, keepdims=True)
    point = _solve(M, counts / n)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, len(counts)))
    for b in range(n_boot):
        Mb = np.stack([rng.dirichlet(row) for row in C])
        pb = rng.multinomial(int(n), counts / n) / n
        boots[b] = _solve(Mb, pb)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return dict(share=point.tolist(), low=lo.tolist(), high=hi.tolist())
