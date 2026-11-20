# Reproducibility

## Pitch label source

Singer-pitch labels for the vocal stems are produced by RMVPE, an open
pitch-tracking network (Wei, Bai, Zhou, Wang; INTERSPEECH 2023), chosen over
CREPE per the decision procedure in `docs/protocol.md` §6.1.

- **Code**: `third_party/rmvpe/rmvpe.py`, vendored from RVC-Project's
  Retrieval-based-Voice-Conversion-WebUI, file `infer/rmvpe.py`, pinned
  commit `81eed5e8f68b6bed1789f682fe78cdd324495afc`. License: MIT (see
  `third_party/rmvpe/LICENSE`). Two small changes from the pinned commit's
  original file are recorded in `third_party/rmvpe/NOTICE` (both drop a
  dependency on the rest of that project's webui, which is not vendored
  here; neither changes the computation).
- **Checkpoint**: `rmvpe.pt`, hosted at
  https://huggingface.co/lj1995/VoiceConversionWebUI . License: MIT (per
  that repository's license tag). This file has been replaced on that
  repository's main branch at least once historically, so the exact commit
  is pinned here, not just "main": commit `e6d0c1a17da07c33557852f9dfa2bd44cc75737d`,
  sha256 `6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193`
  (181184272 bytes), verified against the file downloaded at that commit on
  2026-11-20.
- **Configuration**: 16kHz audio, hop 160 samples (10ms), as the vendored
  code's own `MelSpectrogram` construction uses -- not the original RMVPE
  paper's own training configuration (20ms hop), which belongs to a
  different, non-matching checkpoint the authors' own reference repository
  trains. The checkpoint actually used was trained inside the RVC pipeline
  at the hop this project uses, so the vendored code and checkpoint are a
  matched pair.
- **Voicing threshold**: salience >= 0.03, the common implementation's
  default (spec WP1 §4.4.2), then post-processed per §4.3 (short voiced
  runs removed, quiet frames forced unvoiced).
- **Validation**: a CREPE cross-check on 10 seeded songs and a 3-song
  pitch-sanity plot (`results/figures/pitch_check_*.png`) gate this choice;
  see `results/gates/WP1.md`.

## Known limitations (data card)

- **RMVPE octave errors on "Giselle - Moss".** The pitch-sanity plot for
  this song (`results/figures/pitch_check_train_Giselle - Moss.png`) shows
  RMVPE's estimate jumping to the wrong octave on some voiced frames,
  rather than an unvoiced or near-miss error. The CREPE cross-check still
  passed its gate overall (mean agreement 0.906, mean voicing kappa 0.707,
  threshold 0.85 on agreement), so this is accepted as a known label-noise
  limitation of the automatic pitch labels rather than a reason to change
  the label source: the auxiliary target for affected frames is simply
  wrong by an octave, which this project has no way to detect or correct
  without manual annotation. It is reported here rather than silently
  present only in a figure, so the report's limitations section (protocol
  §3.5, "the pitch labels come from a model, not from human annotation")
  can point to a concrete example.

## Environment

See `docs/compute.md` for the base image and `requirements.lock` for the
locked dependency set. `uv pip compile` was run on an EU-RO-1 pod targeting
Python 3.11.
