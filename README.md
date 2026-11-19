# Voice-aware piano transcription

A study of whether adding an auxiliary vocal-pitch prediction head, trained
jointly with a piano transcription model on synthetic piano+vocal mixtures,
changes what the shared representation learns and how it transcribes piano
under vocal interference.

This repository is private during the study and will be documented for
public, independent reproduction before release. See:

- `docs/protocol.md` — the frozen research protocol (method, hypotheses,
  evaluation plan).
- `docs/method.md` — implementation details of data, mixing, model and
  training.
- `docs/reproducibility.md` — environment, seeds, and exact reproduction
  steps.
- `docs/compute.md` — hardware used and how it was selected.
- `docs/deviations.md` — any approved deviation from the frozen protocol.

## Status

Work in progress. This README will be filled in as the pipeline is built;
a full "how to reproduce on any Linux GPU box" section will land before
release.

## Credits

This project builds on Kong et al.'s piano transcription model and code
(Apache License 2.0), and on the MAESTRO and MUSDB18-HQ datasets. See
`NOTICE` for full attribution and `third_party/piano_transcription/` for
the vendored code and its license.
