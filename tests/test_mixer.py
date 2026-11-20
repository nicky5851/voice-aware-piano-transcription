import numpy as np
import pyloudnorm
import pytest

from vapt.constants import SAMPLE_RATE
from vapt.mixing.mixer import SilentSegmentError, mix


def _sine(freq: float, dur_s: float, amp: float = 0.3, phase: float = 0.0) -> np.ndarray:
    t = np.arange(int(dur_s * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)


@pytest.fixture
def meter() -> pyloudnorm.Meter:
    return pyloudnorm.Meter(SAMPLE_RATE)


@pytest.mark.parametrize("target_db", [-9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0])
def test_vpr_measured_close_to_target(meter, target_db):
    piano = _sine(220.0, 5.0, amp=0.25)
    voice = _sine(440.0, 5.0, amp=0.15, phase=0.3)
    _, info = mix(piano, voice, target_db, meter)
    assert abs(info.vpr_measured - target_db) <= 0.1
    assert info.vpr_target == target_db


def test_peak_scale_does_not_change_vpr(meter):
    # Loud sources that will clip and need the peak-scale path.
    piano = _sine(220.0, 5.0, amp=0.9)
    voice = _sine(440.0, 5.0, amp=0.9, phase=0.3)
    mixed, info = mix(piano, voice, 6.0, meter)
    assert info.peak_scale < 1.0
    assert np.max(np.abs(mixed)) <= 0.99 + 1e-6
    assert abs(info.vpr_measured - 6.0) <= 0.1


def test_equal_length_required(meter):
    piano = _sine(220.0, 5.0)
    voice = _sine(440.0, 4.0)
    with pytest.raises(ValueError):
        mix(piano, voice, 0.0, meter)


def test_silent_voice_raises(meter):
    piano = _sine(220.0, 5.0)
    voice = np.zeros_like(piano)
    with pytest.raises(SilentSegmentError):
        mix(piano, voice, 0.0, meter)


def test_too_short_segment_raises(meter):
    piano = _sine(220.0, 0.1)
    voice = _sine(440.0, 0.1)
    with pytest.raises(SilentSegmentError):
        mix(piano, voice, 0.0, meter)
