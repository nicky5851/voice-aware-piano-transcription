import numpy as np
import torch

from vapt.infer import deframe, enframe, transcribe
from vapt.models.voice_aware import VoiceAwareKong


def test_enframe_segment_count_and_hop():
    segment_samples = 1000
    audio = np.zeros((1, 4 * segment_samples), dtype=np.float32)  # 4x -> 7 overlapping segments at 50% hop
    segments = enframe(audio, segment_samples)
    # pointer goes 0, 500, 1000, ..., up to <= 3*segment_samples (last start where +segment_samples <= 4000)
    expected_starts = list(range(0, 4 * segment_samples - segment_samples + 1, segment_samples // 2))
    assert segments.shape == (len(expected_starts), segment_samples)


def test_enframe_single_segment_when_audio_equals_segment_length():
    segment_samples = 1000
    audio = np.arange(segment_samples, dtype=np.float32)[None, :]
    segments = enframe(audio, segment_samples)
    assert segments.shape == (1, segment_samples)
    assert np.array_equal(segments[0], audio[0])


def test_deframe_single_segment_passthrough():
    x = np.random.rand(1, 40, 3)
    out = deframe(x)
    assert np.array_equal(out, x[0])


def test_deframe_stitches_middle_quarters():
    # 3 segments of 21 frames each; after dropping the last frame, 20 remain (20 % 4 == 0).
    n, segment_frames, c = 3, 21, 1
    x = np.zeros((n, segment_frames, c))
    for i in range(n):
        x[i, :, 0] = np.arange(segment_frames) + i * 100

    out = deframe(x)
    trimmed = x[:, 0:-1, :]  # 20 frames/segment
    expected = np.concatenate(
        [
            trimmed[0, 0 : int(20 * 0.75)],
            trimmed[1, int(20 * 0.25) : int(20 * 0.75)],
            trimmed[2, int(20 * 0.25) :],
        ],
        axis=0,
    )
    assert np.array_equal(out, expected)


def test_transcribe_runs_on_random_audio_and_voice_roll_shape_matches_audio():
    torch.manual_seed(0)
    model = VoiceAwareKong(100, 88)
    model.train()  # starting mode; transcribe() must switch to eval() internally, then restore it
    was_training_during_forward = []
    model.voice_head.register_forward_hook(lambda m, i, o: was_training_during_forward.append(model.training))

    audio = np.random.randn(16000 * 12).astype(np.float32) * 0.01  # 12s, short enough to be quick
    note_events, voice_roll_pred = transcribe(model, audio, device="cpu")

    assert isinstance(note_events, list)
    assert voice_roll_pred.ndim == 2 and voice_roll_pred.shape[1] == 88
    assert was_training_during_forward and all(t is False for t in was_training_during_forward)
    assert model.training  # restored to the caller's original mode afterwards
