"""Step 6 of the tile flow: turn raw model output into stored scores and a status.

Three different kinds of "unsure" are kept apart (DESIGN_NOTE.md §4.2):
  * unfamiliar   - the embedding is far from every training tile (clouds, land types outside the 7 classes, ...);
                   softmax confidence cannot detect this, so it is a separate score
  * needs_review - ambiguous between known classes (low calibrated confidence) or flagged by the quality checks
  * confident    - everything else
All numbers are stored, and the thresholds come from the model's manifest (a versioned policy), so the status can
be recomputed later without re-running the model.
"""
from dataclasses import dataclass

import numpy as np

REVIEW_FLAGS = {'blank', 'grey_low_texture'}


@dataclass(frozen=True)
class Scores:
    label: str
    confidence: float
    margin: float
    probabilities: dict[str, float]
    ood_score: float
    status: str


def calibrated_probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits.astype(np.float64) / temperature
    p = np.exp(z - z.max())
    return p / p.sum()


def ood_score(embedding: np.ndarray, references: np.ndarray) -> float:
    """1 - cosine similarity to the nearest training tile: 0 = identical to something seen in training."""
    e = embedding / (np.linalg.norm(embedding) + 1e-12)
    return float(1.0 - (references @ e).max())


def decide_status(confidence: float, ood: float, flags: list[str], policy: dict) -> str:
    if ood > policy['ood_threshold']:
        return 'unfamiliar'
    if confidence < policy['min_confidence'] or REVIEW_FLAGS & set(flags):
        return 'needs_review'
    return 'confident'


def score(logits, embedding, *, classes, temperature, references, policy, flags) -> Scores:
    p = calibrated_probabilities(logits, temperature)
    top2 = np.argsort(p)[::-1][:2]
    confidence, ood = float(p[top2[0]]), ood_score(embedding, references)
    return Scores(label=classes[int(top2[0])], confidence=confidence, margin=float(p[top2[0]] - p[top2[1]]),
                  probabilities={c: round(float(v), 6) for c, v in zip(classes, p)}, ood_score=ood,
                  status=decide_status(confidence, ood, flags, policy))
