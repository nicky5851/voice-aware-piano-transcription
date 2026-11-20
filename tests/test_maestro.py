import csv

from vapt.data.maestro import read_maestro_rows


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
