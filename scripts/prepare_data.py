#!/usr/bin/env python
"""Thin CLI for WP1 data acquisition and preprocessing (spec WP1).

Every subcommand prints its config, git sha, hardware record and seed (where
applicable) at start, per CLAUDE.md §5.5, and logs to work/logs/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vapt.data.checksums import sha256_file, write_checksums  # noqa: E402
from vapt.data.maestro import pack_maestro_parallel  # noqa: E402
from vapt.data.musdb import musdb_split  # noqa: E402
from vapt.data.pitch import postprocess_voicing, save_pitch  # noqa: E402
from vapt.data.windows import voiced_windows  # noqa: E402
from vapt.infra.hardware import record_hardware  # noqa: E402

MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0.zip"
MUSDB_URL = "https://zenodo.org/api/records/3338373/files/musdb18hq.zip/content"
MUSDB_MD5 = "12d4f2ecd55245a4688754dd76363103"


def _log_header(name: str, config: dict) -> None:
    print(f"=== prepare_data.py {name} ===")
    print(f"config: {json.dumps(config, default=str)}")
    print(f"hardware: {json.dumps(record_hardware())}")


def _md5_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {dest}")
    t0 = time.time()
    urllib.request.urlretrieve(url, dest)
    print(f"downloaded {dest.stat().st_size} bytes in {time.time() - t0:.1f}s")


def cmd_download_maestro(args: argparse.Namespace, cfg: dict) -> None:
    _log_header("download-maestro", cfg["maestro"])
    dataset_dir = Path(cfg["maestro"]["dataset_dir"])
    zip_path = dataset_dir.parent / "maestro-v2.0.0.zip"
    if not zip_path.exists():
        _download(MAESTRO_URL, zip_path)
    digest = sha256_file(zip_path)
    print(f"maestro-v2.0.0.zip sha256={digest} size={zip_path.stat().st_size}")
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dataset_dir.parent)
    csv_path = dataset_dir / cfg["maestro"]["csv_name"]
    if not csv_path.exists():
        print(f"FATAL: {csv_path} missing after extraction", file=sys.stderr)
        sys.exit(1)
    print(f"maestro extracted to {dataset_dir}, csv present")


def cmd_download_musdb(args: argparse.Namespace, cfg: dict) -> None:
    _log_header("download-musdb", cfg["musdb"])
    dataset_dir = Path(cfg["musdb"]["dataset_dir"])
    zip_path = dataset_dir.parent / "musdb18hq.zip"
    if not zip_path.exists():
        _download(MUSDB_URL, zip_path)
    digest = _md5_file(zip_path)
    expected = cfg["musdb"]["archive_md5"]
    if digest != expected:
        print(f"FATAL: musdb18hq.zip md5 mismatch: got {digest}, expected {expected}", file=sys.stderr)
        sys.exit(1)
    print(f"musdb18hq.zip md5 OK ({digest})")
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dataset_dir)
    print(f"musdb18hq extracted to {dataset_dir}")


def cmd_maestro(args: argparse.Namespace, cfg: dict) -> None:
    _log_header("maestro", cfg["maestro"])
    dataset_dir = Path(cfg["maestro"]["dataset_dir"])
    workspace = Path(cfg["workspace"])
    t0 = time.time()
    counts = pack_maestro_parallel(dataset_dir, workspace, num_workers=cfg["maestro"]["num_workers"])
    elapsed = time.time() - t0
    print(f"maestro packing counts (from CSV, never hard-coded): {counts}")
    print(f"maestro packing took {elapsed:.1f}s")

    report_path = workspace / "reports" / "wp1_maestro_counts.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"counts": counts, "elapsed_s": elapsed}, indent=2))

    hdf5_dir = workspace / "hdf5s" / "maestro"
    files = sorted(hdf5_dir.rglob("*.h5"))
    write_checksums(workspace, files, workspace / "checksums" / "maestro_hdf5.sha256")
    print(f"wrote checksums for {len(files)} HDF5 files")


def cmd_musdb_split(args: argparse.Namespace, cfg: dict) -> None:
    _log_header("musdb-split", cfg["musdb"])
    dataset_dir = Path(cfg["musdb"]["dataset_dir"])
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"
    train_songs = [p.name for p in sorted(train_dir.iterdir()) if p.is_dir()]
    test_songs = [p.name for p in sorted(test_dir.iterdir()) if p.is_dir()]
    if len(train_songs) != cfg["musdb"]["n_train_official"]:
        print(
            f"FATAL: expected {cfg['musdb']['n_train_official']} MUSDB train songs, found {len(train_songs)}",
            file=sys.stderr,
        )
        sys.exit(1)
    if len(test_songs) != cfg["musdb"]["n_test_official"]:
        print(
            f"FATAL: expected {cfg['musdb']['n_test_official']} MUSDB test songs, found {len(test_songs)}",
            file=sys.stderr,
        )
        sys.exit(1)

    split = musdb_split(train_songs, test_songs, seed=cfg["musdb"]["split_seed"])
    workspace = Path(cfg["workspace"])
    split_path = workspace / "splits" / "musdb_split.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split, indent=2))
    print(f"musdb split sizes: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")
    print(f"wrote {split_path}")


def cmd_musdb_vocals(args: argparse.Namespace, cfg: dict) -> None:
    _log_header("musdb-vocals", cfg["musdb"])
    import numpy as np
    import soundfile as sf

    from vapt.data.musdb import resample_to_16k, to_mono

    dataset_dir = Path(cfg["musdb"]["dataset_dir"])
    workspace = Path(cfg["workspace"])
    split_path = workspace / "splits" / "musdb_split.json"
    split = json.loads(split_path.read_text())

    name_to_subdir = {}
    for sub in ("train", "test"):
        for p in (dataset_dir / sub).iterdir():
            if p.is_dir():
                name_to_subdir[p.name] = sub

    rows = []
    for split_name, songs in split.items():
        out_dir = workspace / "vocals16k" / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for song in songs:
            src = dataset_dir / name_to_subdir[song] / song / "vocals.wav"
            audio, sr = sf.read(src, always_2d=True)  # [T, C]
            assert sr == 44100, f"expected 44.1kHz source, got {sr} for {song}"
            mono = to_mono(audio.T)  # [T,2]->[2,T] expected by to_mono
            mono16k = resample_to_16k(mono.astype(np.float32))
            dst = out_dir / f"{song}.flac"
            sf.write(dst, mono16k, 16000, subtype="PCM_24", format="FLAC")
            rms = float(np.sqrt(np.mean(mono16k.astype(np.float64) ** 2) + 1e-12))
            duration_s = len(mono16k) / 16000
            rows.append({"song": song, "split": split_name, "rms": rms, "duration_s": duration_s})
            print(f"{split_name}/{song}: rms={rms:.4f} duration_s={duration_s:.1f}")

    report_path = workspace / "reports" / "wp1_musdb_vocals.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(rows, indent=2))

    out_root = workspace / "vocals16k"
    files = sorted(out_root.rglob("*.flac"))
    write_checksums(workspace, files, workspace / "checksums" / "vocals16k.sha256")
    print(f"wrote checksums for {len(files)} vocal stems")


def _iter_vocal_stems(workspace: Path):
    for split_dir in sorted((workspace / "vocals16k").iterdir()):
        if not split_dir.is_dir():
            continue
        for flac in sorted(split_dir.glob("*.flac")):
            yield split_dir.name, flac.stem, flac


def cmd_pitch_label(args: argparse.Namespace, cfg: dict) -> None:
    """Run RMVPE (or CREPE, via --source) on every vocal stem; post-process voicing."""
    import soundfile as sf

    _log_header("pitch-label", cfg["pitch"])
    workspace = Path(cfg["workspace"])
    source = args.source

    model = None
    if source == "rmvpe":
        from vapt.data.rmvpe_infer import load_rmvpe_model, run_rmvpe

        model = load_rmvpe_model(Path(args.checkpoint), device=args.device, is_half=False)

    stems = list(_iter_vocal_stems(workspace))
    print(f"labelling {len(stems)} vocal stems with source={source}")
    t0 = time.time()
    for split_name, song, flac_path in stems:
        audio, sr = sf.read(flac_path)
        assert sr == 16000
        audio = audio.astype("float32")

        if source == "rmvpe":
            track = run_rmvpe(audio, model)
        else:
            from vapt.data.crepe_infer import run_crepe

            track = run_crepe(audio, device=args.device)

        track = postprocess_voicing(
            track, audio, min_run=cfg["pitch"]["min_run_frames"], floor_dbfs=cfg["pitch"]["floor_dbfs"]
        )
        out_dir = workspace / "pitch" / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        save_pitch(out_dir / f"{song}.npz", track)
        print(f"{split_name}/{song}: voiced_frac={track.voiced.mean():.3f}")
    print(f"pitch labelling ({source}) took {time.time() - t0:.1f}s for {len(stems)} stems")

    files = sorted((workspace / "pitch").rglob("*.npz"))
    write_checksums(workspace, files, workspace / "checksums" / f"pitch_{source}.sha256")


def cmd_pitch_crepe_check(args: argparse.Namespace, cfg: dict) -> None:
    """CREPE cross-check against the chosen label source, on 10 seeded songs (spec §4.4.4)."""
    import numpy as np

    from vapt.data.crepe_infer import jointly_voiced_agreement, run_crepe, voicing_cohen_kappa
    from vapt.data.pitch import load_pitch

    _log_header("pitch-crepe-check", cfg["pitch"]["crepe_check"])
    workspace = Path(cfg["workspace"])
    stems = list(_iter_vocal_stems(workspace))
    rng = np.random.default_rng(cfg["pitch"]["crepe_check"]["seed"])
    n = cfg["pitch"]["crepe_check"]["n_songs"]
    idx = rng.choice(len(stems), size=min(n, len(stems)), replace=False)

    import soundfile as sf

    rows = []
    for i in idx:
        split_name, song, flac_path = stems[i]
        audio, sr = sf.read(flac_path)
        assert sr == 16000
        audio = audio.astype("float32")
        primary = load_pitch(workspace / "pitch" / split_name / f"{song}.npz")
        crepe_track = run_crepe(audio, device=args.device)
        agreement = jointly_voiced_agreement(
            primary, crepe_track, cents_tol=cfg["pitch"]["crepe_check"]["agreement_cents"]
        )
        kappa = voicing_cohen_kappa(primary.voiced, crepe_track.voiced)
        rows.append({"split": split_name, "song": song, "agreement": agreement, "voicing_kappa": kappa})
        print(f"{split_name}/{song}: agreement={agreement:.3f} kappa={kappa:.3f}")

    mean_agreement = float(np.nanmean([r["agreement"] for r in rows]))
    mean_kappa = float(np.nanmean([r["voicing_kappa"] for r in rows]))
    print(f"mean agreement={mean_agreement:.3f} mean kappa={mean_kappa:.3f} (gate: agreement >= 0.85)")

    out = {"songs": rows, "mean_agreement": mean_agreement, "mean_voicing_kappa": mean_kappa}
    report_path = workspace / "reports" / "wp1_crepe_check.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(out, indent=2))


def cmd_pitch_sanity_plot(args: argparse.Namespace, cfg: dict) -> None:
    """f0 over the log-spectrogram for 3 seeded songs, 20s excerpts (spec §4.4.5, the G1 gate)."""
    import librosa
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile as sf

    from vapt.data.pitch import load_pitch

    cfg_sanity = cfg["pitch"]["sanity_check"]
    _log_header("pitch-sanity-plot", cfg_sanity)
    workspace = Path(cfg["workspace"])
    stems = list(_iter_vocal_stems(workspace))
    rng = np.random.default_rng(cfg_sanity["seed"])
    idx = rng.choice(len(stems), size=min(cfg_sanity["n_songs"], len(stems)), replace=False)

    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    excerpt_s = cfg_sanity["excerpt_s"]

    for i in idx:
        split_name, song, flac_path = stems[i]
        audio, sr = sf.read(flac_path)
        assert sr == 16000
        n = int(excerpt_s * sr)
        start = max(0, (len(audio) - n) // 2)
        excerpt = audio[start : start + n].astype("float32")

        track = load_pitch(workspace / "pitch" / split_name / f"{song}.npz")
        frame0 = start // 160
        n_frames = n // 160
        f0 = track.f0_hz[frame0 : frame0 + n_frames]
        voiced = track.voiced[frame0 : frame0 + n_frames]
        t_frames = np.arange(len(f0)) * 0.01

        spec = np.abs(librosa.stft(excerpt, n_fft=1024, hop_length=160))
        log_spec = librosa.amplitude_to_db(spec, ref=np.max)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        t_spec = np.arange(log_spec.shape[1]) * (160 / sr)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.pcolormesh(t_spec, freqs, log_spec, shading="auto", cmap="magma")
        ax.set_ylim(0, 1200)
        f0_plot = np.where(voiced, f0, np.nan)
        ax.plot(t_frames, f0_plot, color="cyan", linewidth=1.5, label="f0 (voiced)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("frequency (Hz)")
        ax.set_title(f"{split_name}/{song} ({track.source})")
        ax.legend(loc="upper right")
        out_path = fig_dir / f"pitch_check_{split_name}_{song}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")


def cmd_windows(args: argparse.Namespace, cfg: dict) -> None:
    """Voiced-window index per split; 30s windows for val/test too (spec §4.6)."""
    import numpy as np

    from vapt.data.pitch import load_pitch

    _log_header("windows", cfg["windows"])
    workspace = Path(cfg["workspace"])

    for split_dir in sorted((workspace / "pitch").iterdir()):
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name
        win_specs = [("voiced_windows_10s.json", cfg["windows"]["win_10s"])]
        if split_name in ("val", "test"):
            win_specs.append(("voiced_windows_30s.json", cfg["windows"]["win_30s"]))

        for out_name, win_s in win_specs:
            windows = []
            songs_with_zero = []
            fracs = []
            for npz_path in sorted(split_dir.glob("*.npz")):
                song = npz_path.stem
                track = load_pitch(npz_path)
                starts = voiced_windows(
                    track.voiced, win_s=win_s, hop_s=cfg["windows"]["hop_s"], min_frac=cfg["windows"]["min_voiced_frac"]
                )
                if not starts:
                    songs_with_zero.append(song)
                fracs.append(float(track.voiced.mean()))
                windows.extend({"song": song, "start_s": s} for s in starts)

            out_path = split_dir / out_name
            out_path.write_text(json.dumps(windows, indent=2))
            median_frac = float(np.median(fracs)) if fracs else float("nan")
            print(
                f"{split_name}/{out_name}: {len(windows)} windows, "
                f"{len(songs_with_zero)} songs with zero windows "
                f"({songs_with_zero}), median voiced fraction {median_frac:.3f}"
            )


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download-maestro")
    sub.add_parser("download-musdb")
    sub.add_parser("maestro")
    sub.add_parser("musdb-split")
    sub.add_parser("musdb-vocals")

    p_label = sub.add_parser("pitch-label")
    p_label.add_argument("--source", choices=["rmvpe", "crepe"], default="rmvpe")
    p_label.add_argument("--checkpoint", type=Path, default=Path("/workspace/vapt/data/rmvpe.pt"))
    p_label.add_argument("--device", default="cuda")

    p_check = sub.add_parser("pitch-crepe-check")
    p_check.add_argument("--device", default="cuda")

    sub.add_parser("pitch-sanity-plot")
    sub.add_parser("windows")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dispatch = {
        "download-maestro": cmd_download_maestro,
        "download-musdb": cmd_download_musdb,
        "maestro": cmd_maestro,
        "musdb-split": cmd_musdb_split,
        "musdb-vocals": cmd_musdb_vocals,
        "pitch-label": cmd_pitch_label,
        "pitch-crepe-check": cmd_pitch_crepe_check,
        "pitch-sanity-plot": cmd_pitch_sanity_plot,
        "windows": cmd_windows,
    }
    dispatch[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
