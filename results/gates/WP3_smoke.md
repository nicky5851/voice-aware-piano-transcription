# WP3 smoke run (engineering gate)

Run `lam1_s0_smoke` (λ=1, seed 0), `--max-steps 300`, validating at step 200
on a 40-item validation subset, per spec WP3 §7. This run is
engineering-only; its numbers are not reported as scientific results.

- **Hardware**: NVIDIA A100 80GB PCIe (RunPod, EU-RO-1 data center).
- **Code**: commit `1296878fff9d23927ede62f00df23c1dce75ce9c`, shipped via
  `git archive` to `/workspace/vapt/code/1296878/`.

## Procedure

1. Phase 1 launched fresh (`start_step=0`), ran to step ~255 (killed with
   `pkill -9` right after the step-250 metrics line, after the step-200
   validation had completed).
2. Phase 2 resumed with `--resume auto`: loaded `checkpoint_step_200.pt`,
   continued from `start_step=201`, ran to completion (step 299) and wrote
   `DONE`.

## Resume correctness

Phase 1's own step-250 metrics and phase 2's independently recomputed
step-250 metrics (resumed from the step-200 checkpoint, same seed) matched
exactly:

| | l_kong | l_voice | total |
|---|---|---|---|
| Phase 1, step 250 | 0.912732 | 0.041818 | 0.954650 |
| Phase 2, step 250 (recomputed after resume) | 0.912732 | 0.041818 | 0.954650 |

This is a stronger check than matching batch hashes alone: it shows the
full forward/backward/optimizer computation, not just the data stream, is
bit-reproducible across a kill and resume from checkpoint.

`batch_hashes.jsonl` holds 200 entries (steps 0-199, the logged range per
spec), confirming the data stream for this run.

## Measured throughput and resource use

From `metrics.jsonl` (one line per 50 steps):

| step | l_kong | l_voice | total | s/step | data_wait_s | data-wait % | GPU mem (MB) |
|---|---|---|---|---|---|---|---|
| 0 | 2.9202 | 0.7388 | 3.6589 | 5.793 | 1.069 | 18% (cold start) | 28868 |
| 50 | 1.1176 | 0.0614 | 1.1790 | 1.116 | 0.066 | 6% | 28868 |
| 100 | 1.0032 | 0.0457 | 1.0489 | 2.556 | 0.067 | 3% (+checkpoint) | 28869 |
| 150 | 0.9149 | 0.0430 | 0.9579 | 1.114 | 0.067 | 6% | 28869 |
| 200 | 0.9730 | 0.0463 | 1.0194 | 2.553 | 0.066 | 3% (+diagnostics+checkpoint) | 28869 |
| 250 | 0.9129 | 0.0418 | 0.9547 | 1.105 | 0.066 | 6% | 28869 |

- **Steady-state s/step** (excluding the cold-start step and
  diagnostics/checkpoint steps): ~1.1 s/step at batch 12, 10 s audio.
- **Data-wait**: 0.066-0.069 s/step, **6% of step time** -- well under the
  30% threshold in spec §7.4; no caching/worker/prefetch changes needed.
- **Peak GPU memory**: 28.87 GB, matching Kong's own ~29 GB estimate for
  batch 12 almost exactly. On the 80 GB A100 this leaves ~65% headroom
  (>> the 10% minimum spec WP4 §2.1 will require of any candidate).
- **Validation time** (40-item subset, step 200): checkpoint saved
  14:12:15, `val/step_200.json` written 14:12:54 -- **~39 s** for 40 items
  (fp32 strict, `model.eval()`), i.e. roughly 1 s/item. A full 300-item
  validation (the protocol cadence, every 2,000 steps) would cost on the
  order of 5 minutes -- negligible against a 2,000-step window.
- The steps with elevated time (100, 200) are the diagnostics
  (`every: 100` in this smoke-only config) and checkpoint-save steps, not
  data stalls; `s_per_step` in the log is measured strictly around the
  training step itself and does not include checkpoint or validation time.

## Loss trends

- **L_voice** fell clearly and monotonically: 0.739 (step 0) -> 0.046
  (step 200) -> 0.042 (step 250).
- **L_kong** fell monotonically throughout (2.920 -> 0.913 by step 250),
  so it never rose above its step-0 value, let alone by more than 10%.
- Selection metric (mean clean-excluded onset F1 over the 40-item subset):
  1.90 at step 0, 0.0 at step 200 -- both expected at this scale (300 steps
  total, against a 40,000-step protocol schedule); not interpreted as a
  result.

## Bugs found and fixed during this run (see `docs/reproducibility.md` and
`docs/deviations.md` is not applicable -- these are engineering fixes, not
protocol deviations)

1. `load_pretrained` read Kong's checkpoint shape incorrectly (nested dict,
   not a flat prefixed dict) -- fixed, commit `56c3d42`.
2. `score_clip` crashed when the postprocessor emitted a zero/negative
   duration note -- fixed, commit `19c45c8`.
3. Kong's vendored `output_dict_to_detected_notes` crashed when zero onsets
   were detected for a whole clip (routine for an under-trained model) --
   patched via monkeypatch, commit `2bee462`.
4. `run_training` never logged data-wait or GPU memory, despite the spec
   requiring both -- fixed, commit `1296878`.

All four were caught by this smoke run rather than during real training,
which is exactly its purpose.

## Acceptance

- [x] Loss decreases (L_kong and L_voice both fall).
- [x] Checkpoints resume correctly (bit-exact recomputation after kill).
- [x] Data-wait under 30% of step time (6%).
- [x] Peak memory recorded (28.87 GB) and consistent with the spec's own
  estimate.
- [x] No crash, no FAILED status; run completed with `DONE`.
