# WP2 gate report: mixing, seeded example stream, manifests, metrics, statistics

**Status: G2 passed.**

## Acceptance criteria

| # | Criterion | Result |
|---|---|---|
| 1 | All tests pass on a CPU pod, including the `slow` tests | Pass. 78 passed, 1 skipped (a pre-existing WP1 environment-conditional test, unrelated to WP2), in 59s on a 4-vcpu CPU pod; the `slow` CI-coverage test (`test_ci_coverage_over_null_datasets`) passes separately in under 9s. |
| 2 | `val_manifest.json` and `test_manifest.json` committed with sha256; VPR check logged, max error <= 0.5 dB | Pass. Generated from commit `f403df6` on a clean shipped tree (`git archive`), against the real WP1 data on the network volume: 75 validation clips / 300 items, all 178 MAESTRO test performances / 712 items, zero non-silent-fallback flags on either. Every validation item's measured VPR is within **0.215 dB** of target (`results/data/wp2_val_manifest_vpr_check.json`). Test items are not rendered or scored (test-set discipline, protocol §9.7); their VPR check happens in WP6 at render time. |
| 3 | `stats.py` reproduces the full T2 decision table on a synthetic dataset in under 5 minutes on 8 CPU cores | Pass, with large margin. `tests/test_stats.py::test_true_h3_effect_passes` and `test_report.py` run the full `aggregate -> compute_primary_statistics -> decide -> table_t2` path on a 120-clip x 3-seed synthetic dataset in well under 10 seconds on 4 vCPUs. |
| 4 | `walkthroughs/WP2.md` answers §11 | Pass (workspace-internal, not in this repository). |

## Mixer and stream (gate's own criteria, EXECUTION_PLAN.md G2)

- Mixer within 0.1 dB on synthetic sine-tone signals for targets -9..+9 dB: pass (`test_mixer.py`), including the peak-protection path (loud sources that clip still land within 0.1 dB of target VPR).
- Manifests checksummed; regeneration gives an identical sha256: pass.
- Stream determinism: `batch_specs(step)` is identical across two fresh instances, across DataLoader worker counts 0 and 4, and at a simulated resume at step 137, for a fixed seed; different seeds diverge; `lambda` is not an argument anywhere in the stream. Clean fraction (10,000 draws) and VPR range both match the protocol's targets.
- Stats library passes its synthetic-effect and coverage tests: a true -20% leak-rate effect passes H3a/b/c/d and the intersection-union decision; a null case (identical lam0/lam1 distributions) fails H3a in 0/8 repeated synthetic draws at the 98.75% CI; the 95% CI's empirical coverage over 300 null datasets is 0.90-1.0 (measured; see `test_ci_coverage_over_null_datasets`).
- Isolation test passes, and fails against a planted violation (`tests/test_isolation.py`).

## Engineering choices (recorded `# ENGINEERING CHOICE:` in code and in `walkthroughs/WP2.md`)

- Kong's `Sampler` segment list is built by a sorted (year, filename) directory scan rather than by replaying his `os.walk` order, since the latter is filesystem-dependent and not reproducible across machines; the segment list's *contents* are identical either way, and the stream always visits it through a seeded permutation, never the raw scan order (`src/vapt/mixing/stream.py`).
- The bootstrap engine (`paired_bootstrap`) uses a plain per-resample Python loop rather than a vectorized or multiprocessing implementation; measured runtime (under 10s for B=20000 single-threaded on a small synthetic dataset) is far inside the 5-minute/8-core acceptance bound, so the simpler implementation was kept.

## Spend

No new GPU/CPU spend beyond the existing CPU dev pod used for this work package's tests and manifest generation (`vapt-wp2-dev`, cpu3c x4, $0.12/h). See `docs/compute.md`/the workspace ledger for the running total against the $25 cap.
