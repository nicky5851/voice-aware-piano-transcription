import csv

import pytest

from vapt.data.maestro import MaestroPackingTruncatedError, read_maestro_rows, verify_maestro_file_count


def test_read_maestro_rows(tmp_path):
    csv_path = tmp_path / "maestro-v2.0.0.csv"
    fieldnames = ["canonical_composer", "canonical_title", "split", "year", "midi_filename", "audio_filename", "duration"]
    rows = [
        {
            "canonical_composer": "Bach",
            "canonical_title": "Prelude",
            "split": "train",
            "year": "2018",
            "midi_filename": "2018/a.midi",
            "audio_filename": "2018/a.wav",
            "duration": "100.0",
        },
        {
            "canonical_composer": "Chopin",
            "canonical_title": "Nocturne",
            "split": "validation",
            "year": "2018",
            "midi_filename": "2018/b.midi",
            "audio_filename": "2018/b.wav",
            "duration": "200.0",
        },
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    parsed = read_maestro_rows(csv_path)
    assert len(parsed) == 2
    assert parsed[0]["split"] == "train"
    assert parsed[1]["canonical_composer"] == "Chopin"


def test_verify_maestro_file_count_passes_when_complete(tmp_path):
    hdf5_dir = tmp_path / "hdf5s"
    (hdf5_dir / "2018").mkdir(parents=True)
    (hdf5_dir / "2018" / "a.h5").touch()
    (hdf5_dir / "2018" / "b.h5").touch()
    assert verify_maestro_file_count({"train": 1, "validation": 1}, hdf5_dir) == 2


def test_verify_maestro_file_count_fails_loudly_when_truncated(tmp_path):
    hdf5_dir = tmp_path / "hdf5s"
    (hdf5_dir / "2018").mkdir(parents=True)
    (hdf5_dir / "2018" / "a.h5").touch()  # only 1 of 2 expected files, e.g. a quota cut packing short
    with pytest.raises(MaestroPackingTruncatedError):
        verify_maestro_file_count({"train": 1, "validation": 1}, hdf5_dir)
