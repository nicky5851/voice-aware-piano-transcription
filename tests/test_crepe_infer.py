import numpy as np

from vapt.data.crepe_infer import jointly_voiced_agreement, voicing_cohen_kappa
from vapt.data.pitch import PitchTrack


def _track(f0, voiced):
    f0 = np.asarray(f0, dtype=np.float32)
    voiced = np.asarray(voiced, dtype=bool)
    return PitchTrack(f0_hz=f0, salience=np.ones_like(f0), voiced=voiced, source="test", hop_s=0.01)


def test_jointly_voiced_agreement_all_match():
    a = _track([440.0, 440.0], [True, True])
    b = _track([440.0, 440.0], [True, True])
    assert jointly_voiced_agreement(a, b) == 1.0


def test_jointly_voiced_agreement_one_octave_off_fails():
    a = _track([440.0, 440.0], [True, True])
    b = _track([440.0, 880.0], [True, True])  # second frame an octave off
    assert jointly_voiced_agreement(a, b) == 0.5


def test_voicing_kappa_perfect_agreement():
    a = np.array([True, True, False, False])
    b = np.array([True, True, False, False])
    assert voicing_cohen_kappa(a, b) == 1.0


def test_voicing_kappa_no_agreement_beyond_chance():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, size=1000).astype(bool)
    b = rng.integers(0, 2, size=1000).astype(bool)
    kappa = voicing_cohen_kappa(a, b)
    assert abs(kappa) < 0.15
