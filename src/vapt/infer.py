"""Inference: Kong's segment-and-stitch, fixed post-processing thresholds (protocol §9.1, spec WP3 §4).

Thresholds (onset 0.3, offset 0.3, frame 0.1) are constants, never
configurable. Always fp32 strict and eval(). The test-manifest path is
reachable only through src/vapt/eval/final.py (WP6); this CLI only takes
manifest="val".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from vapt.constants import FRAMES_PER_SECOND, SAMPLE_RATE
from vapt.models.voice_aware import VoiceAwareKong, load_pretrained

THIRD_PARTY_UTILS = Path(__file__).resolve().parents[2] / "third_party" / "piano_transcription" / "utils"

ONSET_THRESHOLD = 0.3
OFFSET_THRESHOLD = 0.3
FRAME_THRESHOLD = 0.1
PEDAL_OFFSET_THRESHOLD = 0.2  # unused (no pedal model); the vendored class's constructor still requires it
SEGMENT_S = 10.0
SEGMENT_SAMPLES = int(SEGMENT_S * SAMPLE_RATE)


def _ensure_vendored_on_path() -> None:
    p = str(THIRD_PARTY_UTILS)
    if p not in sys.path:
        sys.path.insert(0, p)


def enframe(x: np.ndarray, segment_samples: int) -> np.ndarray:
    """[1, audio_samples] (pre-padded to a multiple of segment_samples) -> [N, segment_samples], 50% hop.

    Reimplements piano_transcription_inference.PianoTranscription.enframe
    exactly (that package is a reproduction-check dependency only, spec
    WP3 §5.1, not a runtime dependency of this module).
    """
    assert x.shape[1] % segment_samples == 0
    batch = []
    pointer = 0
    while pointer + segment_samples <= x.shape[1]:
        batch.append(x[:, pointer : pointer + segment_samples])
        pointer += segment_samples // 2
    return np.concatenate(batch, axis=0)


def deframe(x: np.ndarray) -> np.ndarray:
    """[N, segment_frames, C] -> [frames, C]. Matches PianoTranscription.deframe exactly."""
    if x.shape[0] == 1:
        return x[0]
    x = x[:, 0:-1, :]  # drop the extra frame each segment's center=True spectrogram adds
    n, segment_frames, _c = x.shape
    assert segment_frames % 4 == 0
    y = [x[0, 0 : int(segment_frames * 0.75)]]
    for i in range(1, n - 1):
        y.append(x[i, int(segment_frames * 0.25) : int(segment_frames * 0.75)])
    y.append(x[-1, int(segment_frames * 0.25) :])
    return np.concatenate(y, axis=0)


def transcribe(model: VoiceAwareKong, audio_16k: np.ndarray, device: str = "cpu") -> tuple[list[dict], np.ndarray]:
    """audio_16k: mono float32 [S]. Returns (note_events, voice_roll_pred [T, 88]). Always fp32, always eval()."""
    _ensure_vendored_on_path()
    from utilities import RegressionPostProcessor

    was_training = model.training
    model.eval()
    try:
        audio = audio_16k.astype(np.float32)[None, :]
        audio_len = audio.shape[1]
        pad_len = int(np.ceil(audio_len / SEGMENT_SAMPLES)) * SEGMENT_SAMPLES - audio_len
        audio = np.concatenate([audio, np.zeros((1, pad_len), dtype=np.float32)], axis=1)
        segments = enframe(audio, SEGMENT_SAMPLES)

        collected: dict[str, list[np.ndarray]] = {}
        with torch.no_grad():
            for seg in segments:
                x = torch.from_numpy(seg[None, :]).to(device)
                out = model(x, lam=1.0)  # lam only affects gradients; irrelevant under no_grad/eval
                for k, v in out.items():
                    collected.setdefault(k, []).append(v.cpu().numpy()[0])
        output_dict = {k: np.stack(v, axis=0) for k, v in collected.items()}

        for k in list(output_dict.keys()):
            output_dict[k] = deframe(output_dict[k])[0:audio_len]  # a no-op bound (frames << audio_len samples)

        post_processor = RegressionPostProcessor(
            FRAMES_PER_SECOND, classes_num=output_dict["frame_output"].shape[-1],
            onset_threshold=ONSET_THRESHOLD, offset_threshold=OFFSET_THRESHOLD,
            frame_threshold=FRAME_THRESHOLD, pedal_offset_threshold=PEDAL_OFFSET_THRESHOLD,
        )
        note_events, _pedal_events = post_processor.output_dict_to_midi_events(output_dict)
        voice_roll_pred = output_dict["voice_output"]
        return note_events, voice_roll_pred
    finally:
        model.train(was_training)


def _load_model(checkpoint_path: Path, system: str, device: str) -> VoiceAwareKong:
    from vapt.constants import CLASSES_NUM

    model = VoiceAwareKong(FRAMES_PER_SECOND, CLASSES_NUM).to(device)
    if system == "pretrained":
        load_pretrained(model, checkpoint_path)  # voice head left randomly initialised, and ignored downstream
    else:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state)
    return model


def main() -> None:
    from vapt.mixing.manifests import render

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--system", default="trained", choices=["trained", "pretrained"])
    parser.add_argument("--manifest", default="val", choices=["val"])  # "test" is reachable only via eval/final.py
    parser.add_argument("--manifest-path", type=Path, default=Path("manifests/val_manifest.json"))
    parser.add_argument("--hdf5s-dir", type=Path, default=Path("/workspace/vapt/work/hdf5s/maestro"))
    parser.add_argument("--vocals16k-dir", type=Path, default=Path("/workspace/vapt/work/vocals16k"))
    parser.add_argument("--musdb-split", default="val", choices=["val"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    _ensure_vendored_on_path()
    from utilities import write_events_to_midi

    model = _load_model(args.checkpoint, args.system, args.device)
    entries = json.loads(args.manifest_path.read_text())
    if args.limit:
        entries = entries[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        audio, _info = render(entry, args.hdf5s_dir, args.vocals16k_dir, musdb_split=args.musdb_split)
        note_events, voice_roll_pred = transcribe(model, audio, device=args.device)
        write_events_to_midi(start_time=0, note_events=note_events, pedal_events=None, midi_path=str(args.out / f"{entry['clip_id']}.mid"))
        np.savez(args.out / f"{entry['clip_id']}_voice.npz", voice_roll=voice_roll_pred)
    print(f"wrote {len(entries)} predictions to {args.out}")


if __name__ == "__main__":
    main()
