import numpy as np

from tile_service.scoring import calibrated_probabilities, decide_status, ood_score, score

POLICY = {'version': 'p', 'min_confidence': 0.8, 'ood_threshold': 0.3}
CLASSES = list('abc')


def test_probabilities_are_calibrated_by_temperature():
    z = np.array([2.0, 1.0, 0.0])
    p1, p_sharp = calibrated_probabilities(z, 1.0), calibrated_probabilities(z, 0.5)
    assert np.isclose(p1.sum(), 1) and np.isclose(p_sharp.sum(), 1)
    assert p_sharp.max() > p1.max()


def test_ood_score_is_zero_for_a_seen_embedding_and_grows_with_distance():
    refs = np.eye(3, dtype=np.float32)
    assert abs(ood_score(np.array([2.0, 0, 0]), refs)) < 1e-6
    assert ood_score(np.array([1.0, 1.0, 1.0]), refs) > 0.3


def test_status_precedence():
    assert decide_status(0.99, 0.9, [], POLICY) == 'unfamiliar'          # unfamiliar wins even when confident
    assert decide_status(0.50, 0.1, [], POLICY) == 'needs_review'        # ambiguous between known classes
    assert decide_status(0.99, 0.1, ['grey_low_texture'], POLICY) == 'needs_review'
    assert decide_status(0.99, 0.1, ['saturated'], POLICY) == 'confident'  # informational flag only
    assert decide_status(0.99, 0.1, [], POLICY) == 'confident'


def test_score_reports_label_margin_and_all_probabilities():
    s = score(np.array([3.0, 1.0, 0.0]), np.array([1.0, 0, 0]), classes=CLASSES, temperature=1.0,
              references=np.eye(3, dtype=np.float32), policy=POLICY, flags=[])
    assert s.label == 'a' and set(s.probabilities) == set(CLASSES)
    assert 0 < s.margin < s.confidence <= 1
