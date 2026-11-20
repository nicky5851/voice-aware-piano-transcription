# WP3 gate: voice-aware model, training loop, inference, smoke run

## §5 reproduction and sanity checks

Run on an NVIDIA A100 80GB PCIe pod (`tests/test_reproduction.py`, pytest
marker `gpu`). Full numbers and the exact-reproduction note are in
`docs/reproducibility.md`.

| Check | Result | Threshold | Pass |
|---|---|---|---|
| §5.1 Exact reproduction vs `piano_transcription_inference` | max abs diff 0.0 | < 1e-4 | yes |
| §5.2 Piano path unchanged, λ ∈ {0, 0.5, 1} | max abs diff 0.0 (all 4 output keys, all 3 λ) | = 0 | yes |
| §5.3 Pretrained clean onset F1 (validation, 75 clean items) | mean 95.67, min 81.13 | ≥ 85 | yes |

No version pinning was needed for §5.1 (current `librosa`/`torchlibrosa`
already reproduce Kong's released weights exactly).

## §6 tests

Full suite (CPU-testable tests plus the GPU-marked tests in this session,
since this pod has CUDA): **121 passed, 1 skipped**. Every row in the
spec's §6 test table is covered:

- Load, shapes, gradient λ=0/0.5/1, isolation (offset/velocity/refinement
  GRUs): `tests/test_voice_aware.py`.
- Same data (λ0 vs λ1, seed 0, 20 steps): `tests/test_train.py`.
- Resume (CPU, synthetic tiny batches): `tests/test_train.py`.
- Diagnostics isolation (CPU): `tests/test_train.py`.
- Assertions (`batch_size: 6`, `lam: 0.3` refuse): `tests/test_train.py`.
- Reproduction (§5.1, GPU): `tests/test_reproduction.py`.

## §7 smoke run

Full report: `results/gates/WP3_smoke.md`. Summary: loss decreases
(L_kong and L_voice both fall monotonically), checkpoints resume correctly
(bit-exact recomputation of step 250 after a kill-and-resume from the
step-200 checkpoint), data-wait 6% of step time (well under the 30%
threshold, no engineering fixes needed), peak memory 28.87 GB (matches
Kong's own ~29 GB estimate for batch 12), no crash.

Four real bugs were found and fixed along the way (checkpoint-loading
shape, a metrics crash on a degenerate note, a crash on the zero-detections
edge case in Kong's own vendored postprocessor, and missing data-wait/GPU
memory logging) -- see `results/gates/WP3_smoke.md` for detail and the
corresponding commits. All are engineering fixes to code we wrote or a
missing defensive guard in vendored code; none change any protocol-fixed
scientific setting.

## Spend

GPU time for this gate (A100 80GB PCIe, EU-RO-1, $2.09/h): reproduction
checks + four smoke-run iterations (each re-shipped after a bug fix) +
resume test. See `runpod/ledger.csv` (workspace-internal) for the exact
figure; this pod continues into WP4's benchmarks and pilot in the same
session to amortize setup cost, per the combined-session plan.

## Acceptance criteria

- [x] Every §5 check passes; outcomes recorded (`docs/reproducibility.md`).
- [x] Every §6 test passes on a pod; GPU tests pass on a GPU pod.
- [x] The smoke run report exists and resume works
  (`results/gates/WP3_smoke.md`).
- [ ] Pods are stopped and the ledger is updated -- pending: this pod
  continues into WP4 in the same session; it will be stopped and the
  ledger finalized at the end of that work.
- [x] `walkthroughs/WP3.md` answers §9 (workspace-internal, not part of
  this repository per CLAUDE.md §2.4.1).

WP3 is complete pending the final pod-stop/ledger step, which happens at
the end of the combined WP3+WP4 GPU session.
