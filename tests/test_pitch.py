import numpy as np

from vapt.data.pitch import PitchTrack, postprocess_voicing


def _loud_audio(n_frames: int, sample_rate: int = 16000, hop_s: float = 0.01) -> np.ndarray:
    n_samples = int(round(n_frames * hop_s * sample_rate)) + 1000
    rng = np.random.default_rng(0)
    return (rng.standard_normal(n_samples) * 0.5).astype(np.float32)


def test_short_voiced_run_removed():
    voiced = np.array([False, True, True, True, True, False, False], dtype=bool)  # run of 4
    track = PitchTrack(
        f0_hz=np.full(7, 440.0, dtype=np.float32),
        salience=np.ones(7, dtype=np.float32),
        voiced=voiced,
        source="test",
        hop_s=0.01,
    )
    audio = _loud_audio(7)
    out = postprocess_voicing(track, audio, min_run=5, floor_dbfs=-120.0)
    assert not np.any(out.voiced)


def test_voiced_run_of_five_kept():
    voiced = np.array([False, True, True, True, True, True, False], dtype=bool)  # run of 5
    track = PitchTrack(
        f0_hz=np.full(7, 440.0, dtype=np.float32),
        salience=np.ones(7, dtype=np.float32),
        voiced=voiced,
        source="test",
        hop_s=0.01,
    )
    audio = _loud_audio(7)
    out = postprocess_voicing(track, audio, min_run=5, floor_dbfs=-120.0)
    assert np.array_equal(out.voiced, voiced)


def test_quiet_frame_forced_unvoiced():
    voiced = np.ones(10, dtype=bool)
    track = PitchTrack(
        f0_hz=np.full(10, 440.0, dtype=np.float32),
        salience=np.ones(10, dtype=np.float32),
        voiced=voiced,
        source="test",
        hop_s=0.01,
    )
    silence = np.zeros(int(round(10 * 0.01 * 16000)) + 1000, dtype=np.float32)
    out = postprocess_voicing(track, silence, min_run=1, floor_dbfs=-60.0)
    assert not np.any(out.voiced)
