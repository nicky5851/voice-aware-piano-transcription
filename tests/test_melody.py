import numpy as np

from vapt.constants import BEGIN_NOTE, CLASSES_NUM
from vapt.data.pitch import PitchTrack
from vapt.eval.melody import decode_voice, voice_metrics


def _midi_to_hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def _ref_track(t_len: int, voiced_midi: dict[int, float]) -> PitchTrack:
    """voiced_midi: {frame_index: midi_note} for voiced frames; everything else unvoiced."""
    voiced = np.zeros(t_len, dtype=bool)
    f0 = np.zeros(t_len, dtype=np.float32)
    for t, midi in voiced_midi.items():
        voiced[t] = True
        f0[t] = _midi_to_hz(midi)
    return PitchTrack(f0_hz=f0, salience=np.ones(t_len, dtype=np.float32), voiced=voiced, source="test")


def test_perfect_one_hot_head_gives_rpa_1():
    t_len = 200
    note = 60
    voiced_midi = {t: note for t in range(0, t_len, 2)}
    ref = _ref_track(t_len, voiced_midi)

    y = np.zeros((t_len, CLASSES_NUM), dtype=np.float32)
    y[:, note - BEGIN_NOTE] = 1.0

    metrics = voice_metrics(y, ref)
    assert abs(metrics["rpa"] - 1.0) < 1e-9


def test_all_zeros_gives_auroc_half_and_rpa_matches_default_argmax_pitch():
    t_len = 400
    # Half the voiced frames sit exactly at note BEGIN_NOTE (the default all-zero
    # argmax's pitch); the other half sit at a different note.
    voiced_midi = {}
    for t in range(0, 200, 2):
        voiced_midi[t] = BEGIN_NOTE
    for t in range(200, 400, 2):
        voiced_midi[t] = 60
    ref = _ref_track(t_len, voiced_midi)

    y = np.zeros((t_len, CLASSES_NUM), dtype=np.float32)
    metrics = voice_metrics(y, ref)

    assert abs(metrics["auroc"] - 0.5) < 1e-9
    assert abs(metrics["rpa"] - 0.5) < 1e-9


def test_one_semitone_shift_gives_rpa_zero():
    t_len = 100
    note = 60
    voiced_midi = {t: note for t in range(t_len)}
    ref = _ref_track(t_len, voiced_midi)

    y = np.zeros((t_len, CLASSES_NUM), dtype=np.float32)
    y[:, (note + 1) - BEGIN_NOTE] = 1.0  # one semitone above the true pitch

    metrics = voice_metrics(y, ref)
    assert metrics["rpa"] == 0.0


def test_decode_voice_shapes_and_voicing_score():
    t_len = 10
    y = np.zeros((t_len, CLASSES_NUM), dtype=np.float32)
    y[3, 40] = 0.8
    v_t, f_hz = decode_voice(y)
    assert v_t.shape == (t_len,)
    assert f_hz.shape == (t_len,)
    assert abs(v_t[3] - 0.8) < 1e-9
    assert v_t[0] == 0.0
