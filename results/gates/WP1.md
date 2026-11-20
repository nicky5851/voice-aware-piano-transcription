# WP1 gate report: data acquisition, pitch labels, voiced windows

**Status: G1 passed.** Approved 2026-11-20, with the Giselle-Moss RMVPE
octave glitch accepted as a known label-noise limitation (recorded in
`docs/reproducibility.md`) rather than a blocker, and a follow-up hard
file-count assertion added to the MAESTRO extraction so a future truncation
fails loudly instead of needing a by-hand count (see WP2's first commit on
`maestro.py`/`prepare_data.py`).

## Acceptance criteria

| # | Criterion | Result |
|---|---|---|
| 1 | All data packed; counts printed from the CSV, never hard-coded | Pass. MAESTRO: 967 train / 137 validation / 178 test = 1282 rows, 1282 HDF5 files (`results/data/wp1_data_report.json`). The first extraction attempt hit the network volume's quota mid-extraction (year 2004 cut short, 108 files missing); the volume was resized 300->350GB, MAESTRO re-downloaded and re-extracted, and all 1282 rows verified present before packing. `verify_maestro_file_count()` now makes that check automatic. |
| 2 | MUSDB18-HQ md5 OK; split 90/10/50 | Pass. Archive md5 matched the protocol's recorded value. Split: 90 train / 10 held-out-for-validation / 50 test (`manifests/musdb_split.json`), deterministic from `musdb_split()` with seed 0. 150 vocal stems extracted and resampled to 16kHz. |
| 3 | Label source decided and licensed | Pass. RMVPE (protocol's primary choice), checkpoint pinned by commit and sha256, licensed MIT; vendored inference code pinned and licensed (`docs/reproducibility.md`). The CREPE fallback was not needed. |
| 4 | CREPE agreement >= 85% | Pass. Mean agreement 0.906 (threshold 0.85), mean voicing kappa 0.707, over 10 seeded cross-check songs. |
| 5 | Pitch plots look right | Pass, with one noted exception. 3 seeded sanity plots reviewed (`results/figures/pitch_check_*.png`). Two look correct; "Giselle - Moss" shows an RMVPE octave error on some voiced frames. Accepted as-is and recorded as a known data-card limitation rather than switching label source, since the aggregate CREPE-agreement gate still passed. |

## Known limitation carried forward

RMVPE's octave error on "Giselle - Moss" means some training/validation
voice targets for that song are wrong by an octave on affected frames.
This is a limitation of the automatic pitch labels (protocol §3.5), not a
pipeline bug, and is documented in `docs/reproducibility.md` so the
report's limitations section can reference a concrete example.

## Spend

Total WP1 spend: **$0.44** (download/pack/split/vocals on a CPU pod,
RMVPE+CREPE pitch labeling on one RTX 2000 Ada GPU pod). Cumulative
WP0+WP1 spend: **$0.59** of the $25 preparation/pilot cap. All pods used
are stopped.
