import math

import numpy as np

from vapt.data.targets import voice_roll


def test_voice_roll_440hz_peak():
    f0 = np.array([440.0], dtype=np.float32)
    voiced = np.array([True])
    roll = voice_roll(f0, voiced)
    assert roll.shape == (1, 88)
    assert math.isclose(roll[0, 48], 1.0, abs_tol=1e-5)
    assert math.isclose(roll[0, 47], math.exp(-2.0), rel_tol=1e-3)
    assert math.isclose(roll[0, 49], math.exp(-2.0), rel_tol=1e-3)
    assert roll[0, 46] == 0.0
    assert roll[0, 50] == 0.0


def test_voice_roll_out_of_piano_range_is_zero():
    f0 = np.array([20.0, 5000.0], dtype=np.float32)
    voiced = np.array([True, True])
    roll = voice_roll(f0, voiced)
    assert np.all(roll == 0.0)


def test_voice_roll_unvoiced_is_zero():
    f0 = np.array([440.0], dtype=np.float32)
    voiced = np.array([False])
    roll = voice_roll(f0, voiced)
    assert np.all(roll == 0.0)


def test_voice_roll_sharp_pitch_peak_shifts():
    # m = 69.3 -> 30 cents sharp of A4 (index 48's note is 69).
    f0 = np.array([440.0 * 2 ** (0.3 / 12.0)], dtype=np.float32)
    voiced = np.array([True])
    roll = voice_roll(f0, voiced)
    assert np.argmax(roll[0]) == 48
    assert roll[0, 49] > roll[0, 47]


def test_voice_roll_is_float32():
    f0 = np.array([440.0], dtype=np.float64)
    voiced = np.array([True])
    roll = voice_roll(f0, voiced)
    assert roll.dtype == np.float32
