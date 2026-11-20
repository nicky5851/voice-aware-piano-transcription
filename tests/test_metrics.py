import numpy as np

from vapt.data.pitch import PitchTrack
from vapt.eval.metrics import score_clip


def _notes(*triples):
    return [{"midi_note": m, "onset_time": o, "offset_time": off} for (m, o, off) in triples]


def test_ground_truth_against_itself_gives_perfect_f1():
    notes = _notes((60, 1.0, 1.5), (64, 2.0, 2.4), (67, 3.0, 3.9))
    scores = score_clip(notes, notes, None, dur_s=10.0)
    assert scores.n_fp == 0
    assert abs(scores.onset_f1 - 100.0) < 1e-6
    assert abs(scores.onoff_f1 - 100.0) < 1e-6


def _track(n_frames: int, voiced_slice: slice, midi_pitch: int) -> PitchTrack:
    voiced = np.zeros(n_frames, dtype=bool)
    voiced[voiced_slice] = True
    f0 = np.zeros(n_frames, dtype=np.float32)
    f0_hz = 440.0 * 2 ** ((midi_pitch - 69) / 12)
    f0[voiced_slice] = f0_hz
    return PitchTrack(f0_hz=f0, salience=np.ones(n_frames, dtype=np.float32), voiced=voiced, source="test")


def test_degenerate_zero_duration_estimated_note_is_dropped_not_crashed():
    ref = _notes((60, 1.0, 1.5))
    # the postprocessor occasionally emits a note whose offset frame is at or before its onset
    # frame (noisy regression output); mir_eval rejects the whole clip if such an interval reaches
    # it, so it must be dropped before scoring rather than raising.
    est = _notes((60, 1.0, 1.5), (72, 2.0, 2.0), (90, 5.5, 5.0))
    scores = score_clip(ref, est, None, dur_s=10.0)
    assert scores.n_est == 1
    assert abs(scores.onset_f1 - 100.0) < 1e-6


def test_two_false_positives_one_leaked():
    ref = _notes((60, 1.0, 1.5))
    # 72 matches ref; 65 and 90 are both false positives.
    est = _notes((60, 1.0, 1.5), (65, 2.0, 2.4), (90, 5.0, 5.5))
    # Singer sings MIDI 65 from t=1.9s to 2.5s (190-250 frames at 100fps); not voiced near 5.0s.
    track = _track(1000, slice(190, 250), midi_pitch=65)
    scores = score_clip(ref, est, track, dur_s=10.0)
    assert scores.n_fp == 2
    assert scores.n_leak == 1
    assert scores.n_leak / scores.n_fp == 0.5


def test_octave_leak_detected_only_in_octave_variant():
    ref = []
    est = _notes((77, 2.0, 2.4))  # 77 = 65 + 12 (one octave above the sung pitch)
    track = _track(1000, slice(190, 250), midi_pitch=65)
    scores = score_clip(ref, est, track, dur_s=10.0)
    assert scores.n_fp == 1
    assert scores.n_leak == 0
    assert scores.n_leak_oct == 1


def test_clean_condition_gives_zero_leak():
    ref = _notes((60, 1.0, 1.5))
    est = _notes((60, 1.0, 1.5), (65, 2.0, 2.4))
    scores = score_clip(ref, est, None, dur_s=10.0)
    assert scores.n_leak == 0
    assert scores.n_fp == 1


def test_no_notes_either_side_is_nan_and_flagged():
    scores = score_clip([], [], None, dur_s=10.0)
    assert scores.n_ref == 0 and scores.n_est == 0
    assert np.isnan(scores.onset_f1)
