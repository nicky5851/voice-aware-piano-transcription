# Research protocol v3 (frozen): Voice-aware piano transcription

**Status.** Frozen on 2026-11-19. The authors approved Proposal v2 with the refinements in §15, then approved this v3 (including D1) with the §10.2 hardware-selection revision (D14, D15). The hypotheses, metrics, comparisons, decision rules, and data and training settings in this document are fixed. Implementation may change only engineering details that §10.3 lists as free. Any other change is a **deviation**. It needs the authors' written approval, a new protocol version, and an entry in `docs/deviations.md` in the repository. Results from any analysis not defined here are labelled **exploratory**.

**Working title.** Learning to Ignore the Singer: Voice-Aware Multi-Task Fine-Tuning for Piano Transcription in Piano–Vocal Mixtures

---

## 1. Summary

| Item | Content |
|---|---|
| Problem | Solo-piano transcribers trained only on solo piano may report sung notes as piano notes when a singer performs over the piano ("vocal leakage"). |
| Main question | Does training the transcriber to also predict the singer's pitch (an auxiliary task) reduce vocal leakage, compared with identical fine-tuning where that task cannot shape the shared features? |
| Base model | Kong et al.'s high-resolution piano transcriber [3], with its released weights. The model code and weights are reused with credit; they are not our contribution. |
| Data | MAESTRO v2.0.0 piano and MIDI [2]; MUSDB18-HQ vocal stems [21]; singer-pitch labels from RMVPE [19]. All mixtures are synthetic. |
| Primary experiment | λ = 0 against λ = 1, with 3 seeds each (6 fine-tuning runs), plus the pretrained model as a reference. |
| Secondary experiment | λ = 0.5 with 3 seeds, if the approved compute budget allows. It is a sensitivity analysis, not a primary test. |
| Primary hypotheses | H1 degradation and dose–response; H2 recovery by fine-tuning; H3 reduced leakage (proportional and absolute) with accuracy guards; H4 clean-piano non-inferiority. |
| Compute | RunPod only. Hardware is chosen at run time by benchmark and approved cost quote (§10). |
| Outputs | Private GitHub repository `voice-aware-piano-transcription` (public at release), 4–5 page ISMIR-template report, arXiv preprint. |

## 2. Problem and motivation

Deep piano transcribers reach a note F1 of about 97% on solo piano [3, 4], but they are trained and tested only on solo piano. Piano with a singer is a common real-world format. Examples are accompanists, singer-songwriters and shows such as *Piano & I*. That show inspired the project, but none of its audio is used.

A singer is a harder kind of interference than noise, for three reasons:

- The singing range overlaps the piano's middle register.
- An accompaniment shares the song's key and chords.
- The voice is strongly harmonic, so its partials resemble piano partials.

The expected failure is that the transcriber outputs piano notes at the singer's pitch. Published robustness work on piano transcription covers acoustic perturbations such as noise, reverberation, EQ and detuning [5]. It does not cover a simultaneous musical source.

Mixture fine-tuning exposes a network to singing, but it never tells the network what the singing is. An auxiliary output that predicts the singer's pitch gives the network an explicit target for the voice. This is the "hint" form of auxiliary learning [17], and it follows a direction suggested in singing-voice multi-task work [15].

## 3. Literature, prior work and the claimed contribution

The evidence is a **focused** review of 22 sources, each read in full on 19 Nov 2026. They are listed in the Literature matrix tab and in the References. The review is not systematic, so any absence-of-prior-work statement below is qualified.

### 3.1 Established findings we rely on (not our contribution)

- Solo-piano transcription reaches about 96.7–97.4% note F1 on MAESTRO [1–4].
- Kong-style transcribers lose up to 19.2 F1 points under some acoustic perturbations, and augmentation recovers much of the loss [5].
- Multi-instrument transcribers, such as MT3 and YourMT3+, train on mixtures that contain singing. YourMT3+ uses cross-dataset stem mixing [6, 7].
- Multi-task and joint models can help related tasks, but the gains are often small and shrink on real data [12–17].
- RMVPE estimates vocal pitch with high raw pitch accuracy (97.27% on clean vocals, as reported) [19].

### 3.2 Methods and assets we reuse directly (credited, not claimed)

- Kong et al.'s model architecture, training loss, target processing, sampler, post-processing and released checkpoint [3]. The code is Apache-2.0 and is vendored with its licence and a NOTICE.
- The MAESTRO v2.0.0 data and splits [2], and the MUSDB18-HQ vocal stems and splits [21].
- The RMVPE pitch estimator [19] as a label generator, cross-checked against CREPE [18].
- `mir_eval` for transcription and melody metrics, and `pyloudnorm` (ITU-R BS.1770) for loudness.

### 3.3 Our specific contribution

1. A controlled measurement of how much simultaneous singing degrades a pretrained state-of-the-art piano transcriber, as a function of voice-to-piano ratio.
2. A leakage measure: the proportion of false-positive notes at the sung pitch (voice-leak rate) together with the absolute number of such notes per minute.
3. A matched test of a singer-pitch auxiliary task. λ = 0 and λ = 1 differ only in whether the auxiliary gradient reaches the shared features (§7.3). The design includes a pre-registered check that the auxiliary head actually learned the voice.
4. A reproducible recipe: code, seeded manifests and configs. It contains no redistributed audio.

### 3.4 Novelty statement (qualified)

Wording for the report:

> "In a focused review of 22 sources (searched up to November 2026), we found no study that measures vocal-note leakage into piano transcription, or that uses the interfering voice's pitch as an auxiliary target for a piano transcriber. YourMT3+ [7] trains with singing in mixtures but reports piano results only on solo-piano sets. We do not claim the review is exhaustive."

The report must not use "first" without this qualification. Claims about absence of prior work are labelled as **unverified beyond the focused review**.

### 3.5 Limitations of synthetic piano–vocal mixtures (stated in the report)

- The voice and the piano come from different recordings. They do not share key, tempo, harmony or phrasing, so there is less pitch coincidence than in real accompaniment. This may make leakage easier or harder to detect than in real performances.
- There is no shared room acoustics, microphone bleed or performer interaction.
- MAESTRO is mostly classical concert repertoire, which is unlike pop accompaniment.
- The pitch labels come from a model (RMVPE), not from human annotation.
- One transcriber architecture is tested.
- Singer identity is not controlled across MUSDB18-HQ songs.
- Joint models in prior work lost much of their gain on real mixtures [12], so external validity is not established.

## 4. Aim, objectives and research questions

**Aim.** Determine whether a singer-pitch auxiliary task makes a deep piano transcriber more robust to simultaneous singing.

| # | Objective | Done when |
|---|---|---|
| O1 | Build a controlled mixing pipeline with singer-pitch labels | Measured VPR is within ±0.5 dB of the target for every test item; the manifests reproduce by checksum |
| O2 | Quantify how much the pretrained model degrades under singing | H1 is evaluated |
| O3 | Implement the voice-aware model and its matched control, and verify the auxiliary head learns | WP4 gates pass; the 6 primary runs complete |
| O4 | Test the auxiliary task's effect on leakage and accuracy | H2–H4 are evaluated under §9 |
| O5 | Release reproducibly | A fresh GPU machine reproduces the tables from the README |

**RQ-main.** Does a singer-pitch auxiliary task during fine-tuning reduce vocal leakage, compared with identical fine-tuning where the task cannot shape the shared features?

- **RQ1.** How much does singing degrade the pretrained transcriber, and does the degradation grow with VPR?
- **RQ2.** How much of that loss does mixture fine-tuning recover?
- **RQ3.** Does the voice-aware model keep its clean-piano accuracy?

## 5. Definitions

**Audio and frames.** Audio is mono, float32, at 16 kHz. There are 100 frames per second (hop 160 samples). Keys are indexed k = 0…87, for MIDI notes 21–108.

**Voice-to-piano ratio (VPR).** L_p and L_v are the integrated loudness in LUFS (BS.1770, `pyloudnorm`) of the exact piano and voice segments being mixed. For a target r dB, the voice is scaled by g = 10^((L_p + r − L_v)/20), and the mixture is x = piano + g·voice. If max|x| > 0.99, both sources are scaled by the same factor; this leaves the VPR unchanged and the factor is recorded.

**Singer-pitch target.** Let m_t be the MIDI value of the RMVPE pitch at frame t. For voiced frames with 21 ≤ m_t ≤ 108, the target is y_{t,k} = exp(−(21 + k − m_t)² / (2σ²)), with σ = 0.5 semitone; values below 0.01 are set to 0. Unvoiced frames, out-of-range frames and clean (no-voice) examples have all-zero targets.

**Onset matching and accuracy.** `mir_eval.transcription` is used with default tolerances: onset ±50 ms, pitch ±50 cents, offset ratio 0.2 and minimum offset tolerance 50 ms.
- Onset precision P, recall R and F1 ignore offsets; onset-and-offset F1 uses offsets as well.
- Accuracy metrics are reported in **points** on a 0–100 scale.
- Per clip and system: n_ref is the number of reference notes, n_est the number of estimated notes, and n_fp = n_est − n_matched is the number of false-positive notes.

**Leaked note.** A false-positive note leaks if, within ±2 frames of its onset frame, the singer is voiced (reference label) and the note's MIDI pitch equals round(m_t) at some frame in that window. In the octave variant, the pitch class must match instead.

**Leakage measures, per clip:**
- n_leak is the number of leaked notes.
- **Leaked notes per minute:** LPM = n_leak / (clip seconds / 60). This is always defined; it is 0 when there are no false positives.
- **False positives per minute:** FPM = n_fp / (clip seconds / 60).
- **Voice-leak rate:** LR = n_leak / n_fp. It is undefined when n_fp = 0.

**Aggregate leak rate.** This is the pooled ratio LR_pool = Σ_i n_leak,i / Σ_i n_fp,i over the clips in the analysis. It is defined unless no clip has a false positive. Clips with n_fp = 0 enter both sums as 0, so they are handled without exclusion.
- The mean of per-clip LR, over clips with n_fp > 0, is a secondary descriptive figure; the number of excluded clips is reported.
- LR is a proportion and can fall simply because unrelated false positives rise. For that reason it is never interpreted without LPM and FPM (§9.4).

**Voice-head outputs (§8).**
- Voicing score: v_t = max_k ŷ_{t,k}. A frame is predicted voiced if v_t ≥ 0.5; this threshold is fixed.
- Pitch estimate: k* = argmax_k ŷ_{t,k}. The refined key is k̂ = the ŷ-weighted mean of keys k*−1…k*+1. The predicted MIDI value is 21 + k̂, converted to Hz.

## 6. Data

### 6.1 Sources

| Dataset | Role | Version and access | Licence |
|---|---|---|---|
| MAESTRO [2] | Piano audio and ground-truth notes | v2.0.0, direct download; counts read from its CSV | CC BY-NC-SA 4.0 |
| MUSDB18-HQ [21] | Vocal stems | 1.0.0, Zenodo 3338373 (md5 checked) | Non-commercial, educational |
| Kong et al. weights [3] | Initialisation | Zenodo 4034264, `CRNN_note_F1=0.9677_pedal_F1=0.9186.pth` | Code Apache-2.0 |
| RMVPE [19] | Pitch labels | Pinned public checkpoint; licence verified in WP1 | Verified in WP1 |
| GTSinger [22] | Optional E1 only | Hugging Face | CC BY-NC-SA 4.0 |

MAESTRO v2.0.0 is used because the released weights were trained on it, so the test pieces are unseen by the pretrained model.

**Label-source fallback (pre-registered).** WP1 checks the RMVPE checkpoint's provenance and licence. If they cannot be confirmed, or if RMVPE fails the WP1 pitch-sanity check, the labels come from CREPE ("full" model, Viterbi decoding) instead. Everything else in this protocol stays the same, and the change is recorded as a deviation.

### 6.2 Preprocessing

- **Piano.** MAESTRO is packed into Kong's HDF5 format with his unchanged packing code.
- **Vocals.** Each stem is averaged to mono and resampled from 44.1 to 16 kHz with soxr at high quality.
- **Pitch labels.**
  - Run RMVPE on each vocal stem at its native 10 ms hop, aligned to the 100 fps grid. If the implementation uses another hop, f0 is interpolated in cents.
  - A frame is voiced if RMVPE's salience reaches the voicing threshold of its reference implementation (0.03 in the reference code; the exact value used is recorded in WP1). For the CREPE fallback, voiced means periodicity ≥ 0.21 (torchcrepe's documented example threshold). Voiced runs shorter than 5 frames are removed. Frames below −60 dBFS stem level are forced unvoiced.
- **Cross-check.** CREPE is run on 10 seeded songs. The report gives the share of frames voiced by both where the two estimates agree within 50 cents.
- **Voiced windows.** Every 10 s window (1 s hop) with ≥ 50% voiced frames is indexed per split. For the test set, 30 s windows are indexed in the same way.

### 6.3 Splits and leakage controls

| Split | Piano | Voice | Use |
|---|---|---|---|
| Train | MAESTRO v2.0.0 train | MUSDB18-HQ train minus 10 held-out songs (90) | Gradient updates |
| Validation | MAESTRO v2.0.0 validation | The 10 held-out train songs (seed 0) | Checkpoint selection, pilot gates, voice-head manipulation check, sanity checks |
| Test | MAESTRO v2.0.0 test | MUSDB18-HQ test (50) | Final evaluation only, once (§9.7) |

Compositions do not cross MAESTRO splits. Vocal splits are by song. RMVPE's reported training data does not include MUSDB18-HQ.

### 6.4 Manifests

- **Validation manifest.** It is generated with seed 0: 75 MAESTRO validation clips of 30 s, each paired with a held-out vocal window. Each pair is rendered under 4 conditions (clean, −6, 0 and +6 dB), giving 300 items.
- **Test manifest.**
  - One 30 s clip per MAESTRO v2.0.0 test performance, starting at a non-silent position chosen with seed 0.
  - Each clip is paired with one 30 s MUSDB18-HQ test window that is ≥ 50% voiced. Pairs are assigned round-robin with seed 0, and no window is reused until all windows have been used.
  - Each pair is rendered under the 4 conditions (clean, −6, 0 and +6 dB).
- **Storage.** Manifests hold IDs and offsets only. They are committed with their SHA-256 checksums. Rendered audio is never committed.

## 7. Model and training

### 7.1 Base model

Kong et al.'s `Regress_onset_offset_frame_velocity_CRNN` [3]:
- Four separate sub-networks (frame, onset, offset and velocity). Each has conv blocks, a 768-unit projection, a 2-layer BiGRU with 512 features, and a linear layer to 88 keys.
- Refinement GRUs.
- Log-mel input: window 2048, hop 160, 229 bins, 30 Hz–8 kHz.

### 7.2 Voice head

The frame and onset sub-networks are subclassed so that they also return their 512-d GRU features. The parameter names stay the same, so the checkpoint loads unchanged.

- Head: features h = concat(frame, onset) → dropout 0.5 → Linear(1024 → 88) → sigmoid. It adds 90,200 parameters.
- The piano outputs, post-processing and thresholds are unchanged.

### 7.3 Loss and the meaning of λ (refined from v2; see §15, D1)

    L = L_kong(piano outputs) + L_voice( head( GS_λ(h) ) )

- GS_λ is a gradient-scaling identity. Its forward pass returns h unchanged, and its backward pass multiplies the gradient by λ. In code: GS_λ(h) = λ·h + (1 − λ)·h.detach().
- L_voice is Kong's masked BCE averaged over frames and keys (no reweighting, unless the §8.4 fallback triggers).
- **The head is always trained with the full voice loss.**
  - λ controls only how much of the voice-loss gradient reaches the shared frame and onset sub-networks.
  - λ = 0 is the matched control: same architecture, same trained head, same data stream. No auxiliary signal reaches the shared features, so the trained head acts as a probe of the control's features.
  - λ = 1 is ordinary multi-task learning.
  - λ = 0.5 is the secondary sensitivity condition.

### 7.4 Training data

Each example is built on the fly:
- A random 10 s MAESTRO train segment, chosen by Kong's sampler and processed by his unchanged target processor.
- A random voiced 10 s window from a training vocal.
- A VPR drawn from Uniform(−9, +9) dB.
- With probability 0.2, the voice is omitted and the voice target is all zeros.
- No other augmentation is applied.

The seed fixes the stream of piano segments, voice windows, VPRs and clean flags. **The λ0, λ0.5 and λ1 runs with the same seed see identical examples in identical order**, regardless of data-loader worker count (WP3 spec).

### 7.5 Fixed training settings (scientific invariants)

| Setting | Value |
|---|---|
| Initialisation | Released Kong note weights; voice head initialised with Kong's `init_layer` |
| Optimiser | Adam (β 0.9, 0.999; ε 1e-8; no weight decay), lr 1e-4, multiplied by 0.9 every 5,000 optimiser steps |
| Batch | **12 segments of 10 s per optimiser step, in one forward pass on one GPU**. No gradient accumulation and no data-parallel splitting, because the model uses BatchNorm, whose statistics depend on the per-device batch (§15, D2). |
| Optimiser steps | **40,000**, the same for every run and every hardware choice |
| Gradient clipping | None (as in Kong); the gradient norm is logged |
| Training precision | One mode for all primary runs: strict fp32 (default), or TF32 or bf16 autocast if WP4 validates it (§10.2). Inference and evaluation are always fp32. |
| Validation | Every 2,000 steps on the validation manifest |
| Checkpoint selection | The step with the highest validation onset F1 averaged over −6, 0 and +6 dB; ties go to the earlier step. The rule is the same for every λ. |
| Seeds | 0, 1, 2 for each λ |
| Software | One container image (pinned by digest), one git commit and one lockfile for all primary runs |
| Hardware | One GPU type for all primary runs (λ0 and λ1, all seeds), recorded in full |

## 8. Auxiliary-head validation (manipulation check)

The concern is that the 88-output sigmoid head could minimise BCE by predicting near-zeros, because only about 1% of target entries are positive. If the head does not learn the voice, H3 is **not tested**. A failure to learn is not evidence against multi-task learning.

### 8.1 Metrics

Metrics are computed on the validation manifest, at −6, 0 and +6 dB, pooled. Reference: RMVPE labels; frames where the reference pitch is outside the piano range count as unvoiced.

| Metric | Definition |
|---|---|
| Voicing AUROC | Threshold-free: v_t against reference voicing |
| Voicing recall, false-alarm rate | At v_t ≥ 0.5 (`mir_eval.melody`) |
| Raw pitch accuracy (RPA) | Share of reference-voiced frames where the estimated pitch is within 50 cents, regardless of the voicing decision |
| Raw chroma accuracy, overall accuracy | `mir_eval.melody` |
| Loss relative to a trivial predictor | Voice BCE ÷ the BCE of the constant per-key training prior |

Two baselines are always reported: the all-zeros head, and the constant-prior head (its RPA equals the share of frames at the most frequent key).

### 8.2 Diagnostics recorded before full training

These are recorded in WP4 and appear in the report's appendix:
1. **Class imbalance.** The fraction of positive target entries, the per-key positive frequency, and the share of voiced frames in training batches.
2. **Loss scale.** The ratio of L_voice to L_kong at step 0, and the ratio of their gradient norms on the shared parameters. This is logged every 1,000 steps on a fixed batch.
3. **Gradient flow.**
   - For λ = 1 and λ = 0.5, L_voice must give a non-zero gradient to the frame and onset sub-networks.
   - For λ = 0 that gradient must be exactly zero.
   - In every case, L_voice must give zero gradient to the offset and velocity sub-networks.
   - Also logged: the cosine similarity between ∇L_voice and ∇L_kong on the shared parameters.
4. λ is **never** rescaled to balance loss magnitudes, because λ is the experimental factor.

### 8.3 Pilot gate (WP4)

The pilot is one λ = 1 run (seed 0) and one λ = 0 run (seed 0), each for 4,000 steps on the selected hardware. All of the following must hold:

| Gate | Criterion |
|---|---|
| G-V1 learning | λ1 head: RPA ≥ 0.40 and voicing AUROC ≥ 0.75 |
| G-V2 beyond trivial | λ1 voice BCE ≤ 0.7 × the constant-prior BCE, and RPA ≥ 2 × the constant-prior RPA |
| G-V3 gradient flow | All checks in §8.2(3) pass |
| G-P1 no early collapse of piano | Validation clean onset F1 at step 4,000 ≥ pretrained − 2.0 points, for both runs |
| G-E1 engineering | No NaN or Inf; resume test passes; measured throughput recorded |

### 8.4 Pre-registered fallback

If G-V1 or G-V2 fails:
- L_voice becomes BCE with a positive-class weight of 20 on every key. This applies to **all λ conditions alike**.
- The pilot is repeated once.
- If it fails again, the work **stops**. The authors receive a diagnostic report, and no primary runs are launched.

If G-P1 fails, the work also stops and is reported; the learning rate is not changed silently.

### 8.5 Final manipulation check (after training, validation only)

For each λ1 run, the selected checkpoint's head must reach RPA ≥ 0.60 and voicing AUROC ≥ 0.85.
- If all 3 λ1 seeds pass, H3 is interpreted normally.
- If any seed fails, H3 is reported as **"not adequately tested"** and is not counted as support or refutation.
- The report also gives the λ1 head's RPA minus the λ0 (probe) head's RPA, with a bootstrap CI over validation clips. This is a descriptive measure of how much the auxiliary gradient changed the shared features.

## 9. Evaluation and statistical analysis plan

### 9.1 Systems and conditions

The systems are: the pretrained model (P); λ0 with seeds 0–2; λ1 with seeds 0–2; and, secondary, λ0.5 with seeds 0–2. Every test pair is evaluated under clean, −6, 0 and +6 dB conditions.

Inference uses Kong's segment-and-stitch procedure with default thresholds (onset 0.3, offset 0.3, frame 0.1). These are identical for all systems and never tuned.

### 9.2 Units and aggregation

- **Unit of analysis: the test clip** (one per MAESTRO test performance).
- For fine-tuned systems, each clip's metric is the **mean over the 3 seeds** of that clip's metric. For pooled ratios, the counts are summed over seeds.
- **VPR-pooled** means the per-clip average over −6, 0 and +6 dB (counts summed).

### 9.3 Inference method

- **Paired percentile bootstrap over clips.** Each resample draws one clip index set, used for every system and condition, so all comparisons remain paired. B = 20,000 resamples, generator seed 0.
- **Family of primary hypotheses:** {H1, H2, H3, H4}, m = 4.
  - Family-wise α = 0.05 is controlled by Bonferroni: each decision uses a **two-sided 98.75% CI**. For a directional or non-inferiority claim this is equivalent to a one-sided test at α = 0.625%.
  - Each hypothesis is an **intersection-union test**: it is supported only if every component passes at that level. This needs no further correction within a hypothesis.
- Tables also show conventional 95% CIs for description. Decisions use only the 98.75% CIs.

### 9.4 Primary hypotheses and frozen decision rules

| # | Claim | Component | Estimate (paired, per clip) | Passes if (98.75% CI) |
|---|---|---|---|---|
| **H1** | Singing degrades P, and degradation grows with VPR | H1a degradation | P: VPR-pooled onset F1 − clean onset F1 | Upper bound < 0 |
| | | H1b dose–response | P: per-clip OLS slope of onset F1 on VPR ∈ {−6, 0, +6} (points per dB), averaged over clips | Upper bound < 0 |
| **H2** | Mixture fine-tuning recovers accuracy | H2 | λ0 − P, VPR-pooled onset F1 | Lower bound > 0 |
| **H3** | The auxiliary task reduces vocal leakage without hidden costs | H3a proportional | λ1 − λ0, LR_pool (VPR-pooled) | Upper bound < 0 |
| | | H3b absolute | λ1 − λ0, VPR-pooled LPM | Upper bound < 0 |
| | | H3c accuracy guard | λ1 − λ0, VPR-pooled onset F1 | Lower bound > −1.0 point |
| | | H3d false-positive guard | Ratio of means FPM(λ1) / FPM(λ0), VPR-pooled | Upper bound < 1.05 |
| **H4** | Clean piano is preserved | H4 | λ1 − P, clean onset F1 | Lower bound > −1.0 point (non-inferiority margin) |

Notes on the rules:
- **H1b** tests the dose–response claim directly. The slope uses only the three vocal conditions, because "clean" has no finite VPR.
- **H3.** H3a and H3b together require both the proportion and the absolute count of leaked notes to fall. H3c and H3d rule out "improvements" that come from transcribing fewer notes or from shifting errors to other false positives.
  - If H3a passes but H3b or H3d fails, the result is reported as a **change in error composition, not an improvement**.
  - If H3 is not adequately tested (§8.5), it is reported as such, whatever the CIs show.
- **H4.** The margin is **1.0 F1 point in absolute terms** on the 0–100 scale (0.01 on the 0–1 scale), not a relative 1%. The primary comparison is λ1 against P, because λ1 is the proposed model. λ0 − P and λ1 − λ0 on clean audio are secondary and use the same margin.

### 9.5 Practical significance

Statistical significance and practical size are reported separately. The smallest effect sizes of interest (SESOI) are:

| Quantity | SESOI |
|---|---|
| Onset F1 differences (H1a, H2) | 1.0 point |
| H1b slope | −0.1 points per dB (a 1.2-point change across the 12 dB range) |
| H3a, H3b | A 10% relative reduction vs λ0 (that is, (λ1 − λ0) / λ0 ≤ −0.10) |

Each passing component is labelled **"supported, practically meaningful"** if its point estimate reaches the SESOI. Otherwise it is labelled **"supported, below SESOI"**. Effects are reported in raw units with CIs, along with relative change where it is defined.

### 9.6 Secondary and sensitivity analyses (not used for primary decisions)

1. Per-VPR versions of every primary comparison, with 95% CIs.
2. Onset-and-offset F1; onset precision and recall; octave LR; per-clip mean LR (with the number of excluded clips).
3. **Seed robustness.** Primary estimates computed separately for each seed pair (λ1 seed s against λ0 seed s), with the number of seeds whose sign agrees.
4. **Cluster sensitivity.** The bootstrap is repeated resampling MUSDB test songs as clusters (each song with all its clips), to account for shared voices. The report states whether any decision changes.
5. λ0.5 compared with λ0 and λ1, on the same metrics. This is descriptive and gives no decision.
6. H2 recovery fraction: (λ0 − P) / (P_clean − P), VPR-pooled. Descriptive.

### 9.7 Test-set discipline

- No test audio is rendered or scored until all selected checkpoints exist and the final validation manipulation checks are complete. This includes the pretrained model's baseline.
- Sanity checks during development use only the validation manifest. One example: the pretrained model's clean onset F1 must be ≥ 85 on validation; a lower score means a pipeline bug.
- The test manifest is scored exactly once per system. Timestamps and the commit hash are logged.
- A re-run is allowed only to fix an engineering fault that is shown not to depend on the results, such as a crash or a file-naming bug. The fault and the re-run are logged in `docs/deviations.md`.

### 9.8 Outcome reading

| Outcome | Reading |
|---|---|
| H3 supported, H4 holds | The auxiliary task reduces leakage at no measured clean-piano cost |
| H3 supported, H4 fails | Trade-off: leakage falls, clean piano suffers |
| H3a passes but H3b or H3d fails | The composition of errors changed; not an improvement |
| H3 fails, manipulation check passed, H2 supported | Informative null: under this design, mixture fine-tuning alone achieves what the auxiliary task adds |
| Manipulation check failed | H3 not adequately tested; reported as a limitation |
| H2 fails | The fine-tuning recipe is not working; H3 is reported but not interpreted as a method result |
| H1 effects small | Singing barely hurts this model; still a new measurement |

Every pre-registered result is reported whichever way it goes.

## 10. Compute protocol (RunPod)

### 10.1 Where things run

All data processing, labelling, tests, training, inference and evaluation run on RunPod. The authors' development machine is used only for planning, documentation, code review and remote access.

### 10.2 Hardware selection (at run time)

1. **Survey.** The authors list the GPU types available now, with VRAM, on-demand price, stock level and data-centre location. It respects the data centre of the network volume.
2. **Benchmark.** Up to 4 candidates are benchmarked with the fixed scientific settings (batch 12, one run per GPU). Measured for each: seconds per step, data-loader wait, peak memory, and fp32 against bf16 throughput.
3. **Reduced-precision validation.** TF32 or bf16 is allowed only if:
   - the pretrained model's validation onset F1 in that mode is within 0.2 points of strict fp32; and
   - a 1,000-step pilot in that mode has a mean L_kong over steps 801–1,000 inside the range of two strict-fp32 seeds, widened on each side by their difference.
4. **Selection rule (speed first, within an approved budget).**
   - Compare the compatible configurations on: measured training speed (s/step), VRAM headroom at batch 12, CPU cores and system RAM (data-loader wait), local SSD and network-volume needs, wall-clock time for the full batch, and total cost.
   - **Recommendation:** the fastest practical compatible configuration whose total quoted cost fits the budget the authors approve. Practical means in stock in the volume's data centre, available on demand, and faster by a margin that matters: a dearer option must cut the batch's wall-clock time by at least 10% against the next cheaper option to be preferred.
   - **Alternative:** the most cost-efficient compatible configuration (lowest total cost), shown beside the recommendation. All primary runs use one GPU type. Scientific settings are never changed to fit cheaper hardware; if no compatible configuration fits the budget, the work escalates for a budget decision.
5. **Quote and approval.** The quote lists both options with per-run hours, wall-clock time, GPU-hours, pod-hours, CPU, RAM and storage, total cost and assumptions. **The authors pick one option and approve its budget before any paid full-scale training starts.**
   - Benchmarks and pilots have a hard cap of $25 in total (approved by the authors, 19 Nov). Spend is monitored, and work stops before the cap would be exceeded. Primary runs use on-demand GPUs; spot instances are allowed only for recoverable preparation tasks such as download, packing and labelling.
   - Spend that would exceed the approved quote by more than 20% stops the work for re-approval.

### 10.3 What implementation may and may not change

**Free (engineering):**
- The GPU model, number of pods and runs packed per pod (if each run still has a whole GPU's batch of 12).
- The CPU and RAM allocation, and data-loader workers and prefetch.
- Local caching of HDF5 files and vocals.
- Checkpoint I/O frequency (≤ 2,000 steps), logging and monitoring.
- Pod type for preparation tasks (spot allowed, with resume). Primary runs are on-demand.
- Retry and resume mechanics.

**Fixed (scientific):** everything in §5–§9, and §7.5 in particular.

Any out-of-memory error at batch 12 means choosing a larger GPU. It never means a smaller batch, accumulation or data-parallel splitting.

### 10.4 Recorded for reproducibility

For every run, the following are written to `results/compute/<run>.json`:
- GPU model, count and driver; CUDA, PyTorch and cuDNN versions; container image digest.
- CPU, RAM and disk.
- Precision mode and TF32 flags.
- Throughput, wall-clock time and cost.
- Interruptions and resumes.

## 11. Optional experiments, in priority order

These start only after the primary analysis is complete, and only with the authors' go-ahead and budget approval. All of them are exploratory.

- **E1 GTSinger generalisation.** The primary systems are evaluated on test clips mixed with GTSinger vocals (unseen singers and languages). The same metrics are reported descriptively.
- **E2 Representation probing.** Linear probes for voice pitch are trained on frozen frame and onset features of the P, λ0 and λ1 checkpoints, using validation data. The aim is to understand what the auxiliary task changed.
- **E3 Separation reference.** A public vocal-separation model followed by pretrained Kong, reported as a reference point. It is not a pipeline comparison claim.

## 12. Deliverables and report outline

**Deliverables:**
- **Repository** (private until release): code, configs, tests, scripts, docs, results CSVs and figures, LICENSE and NOTICE.
- **Manifests** with checksums.
- **Results:** T1 main table, T2 primary decisions, T3 per-seed table, F1 F1-vs-VPR, F2 leakage (LR and LPM) vs VPR, F3 example piano roll. Appendix: voice-head diagnostics and the compute record.
- **Report:** 4–5 pages in the ISMIR template.
- **arXiv preprint:** cs.SD, cross-listed to eess.AS, after the authors arrange endorsement.

**Report outline.** Introduction (with the qualified novelty statement); Related work; Method; Experimental setup (including this statistical plan); Results; Discussion and limitations (§3.5); Conclusion.
- An acknowledgement credits Kong et al.'s code and weights and the dataset creators.
- A short statement discloses AI-assisted software development, as arXiv's policy asks for significant tool use. The authors decide the wording.

## 13. Reproducibility and integrity rules

1. **Ground-truth isolation.** Only the evaluation module reads test MIDI. A static test enforces this.
2. **No tuning on test data.** All choices are made on validation; §9.7 applies.
3. **Pre-registration.** This protocol is committed as `docs/protocol.md` and tagged `prereg-v3` **before** any test-set evaluation. The authors may optionally also deposit it on OSF for a public timestamp.
4. **Determinism.** Every random choice takes a seed from config. The manifests are checksummed.
5. **Pinned versions.** Datasets, Kong commit, RMVPE checkpoint hash, container digest and lockfile are all pinned.
6. **Nothing licensed is committed.** No audio, MIDI, HDF5, pitch tracks or checkpoints go into git.
7. **Provenance.** Every log records the commit, config and hardware. Every reported number comes from a script-generated CSV.
8. **Explainability gate.** A work package is done only when its tests pass and its implementation and results can be explained clearly, without relying on supplementary materials outside this repository.
9. **Honest reporting.** All pre-registered outcomes are reported, including nulls, failed gates and deviations.

## 14. Risks

| Risk | Early warning | Response |
|---|---|---|
| Released weights don't reproduce with current libraries | WP3 reproduction check fails | Pin the older library versions in the container; the vendored code stays unchanged |
| The voice head predicts near-zeros | WP4 G-V1 or G-V2 | Pre-registered fallback (§8.4), then stop |
| Fine-tuning damages the piano path | G-P1 | Stop and report; no silent change to the learning rate |
| No GPU type fits batch 12 at an approved price | WP4 survey | Stop and report options for a decision |
| Pod interruption | Heartbeat stops | Resume from full-state checkpoint (≤ 2,000 steps lost) |
| Data-loader bottleneck | GPU idle > 30% | Local cache, more workers, or a pre-rendered pool built with the same seeded stream |
| H3 null | Validation leakage of λ0 and λ1 matches | Informative-null reporting (§9.8) |
| Closer prior work appears | Literature watch before submission | Reposition around the measurement and metric |

## 15. Decision log (changes from v2)

| ID | Decision | By | Reason |
|---|---|---|---|
| D0 | Approved: direction, method and scope of v2 | the authors, 19 Nov | Review |
| D1 | λ now scales the auxiliary gradient into the shared features; the head is trained in every condition (λ0 head = probe) | the authors, 19 Nov | Tighter control: λ0 and λ1 now differ only in the gradient path. The voice head can be evaluated in both conditions, which gives a built-in manipulation check. The architecture and parameter count were already identical. |
| D2 | Batch 12 per GPU in one pass, no gradient accumulation, no data-parallel; one GPU type for all primary runs | the authors, 19 Nov | The model uses BatchNorm, so accumulation or data-parallel splitting changes the training dynamics; keeping these fixed keeps conditions comparable across hardware |
| D3 | 40,000 steps fixed, independent of hardware | the authors, 19 Nov | The deadline was removed; step count is a scientific setting, not a scheduling one |
| D4 | All test scoring, including the pretrained baseline, happens once at the end; the sanity gate moved to validation | the authors, 19 Nov | Stricter test-set discipline |
| D5 | 3 seeds per primary condition; λ0.5 secondary | the authors, 19 Nov | Reliability over breadth |
| D6 | Voice-head metrics, diagnostics, gates and fallback (§8) | the authors, 19 Nov | Separate a failed manipulation from a null result |
| D7 | H1 dose–response slope; H3 proportional + absolute + guards; H4 margin defined; family m = 4 with intersection-union tests; SESOI; sensitivity analyses | the authors, 19 Nov | Preprint-grade statistics |
| D8 | Novelty claims qualified; contribution categories separated | the authors, 19 Nov | Accuracy |
| D9 | RunPod-only compute; hardware chosen at run time; the authors approve before paid full training | the authors, 19 Nov | Infrastructure policy |
| D10 | Optional order: E1 GTSinger, E2 probing, E3 separation | the authors, 19 Nov | Focus on understanding our model |
| D11 | Repository `voice-aware-piano-transcription`, private until release; no agent files or AI signatures in it | the authors, 19 Nov | Professional public artefact |
| D12 | No fixed deadline; validity over schedule | the authors, 19 Nov | Quality |
| D13 | The RMVPE hop is aligned natively to 10 ms where possible (v2 assumed 20 ms) | the authors, 19 Nov | Engineering detail, so the labels match the 100 fps frames |
| D14 | Hardware selection: fastest practical compatible configuration within an approved budget, with a cost-efficient alternative; escalate if none fits | the authors, 19 Nov | Throughput first, without unapproved changes to the science |
| D15 | $25 cap for benchmarks and pilots; on-demand for primary runs; spot only for preparation; v3 approved and frozen | the authors, 19 Nov | Budget control |

## 16. Budget and setup

1. Benchmarks and pilots: hard cap of $25 in total. Full-scale training needs a separate quote and the authors' approval.
2. On-demand GPUs for primary runs; spot instances only for recoverable preparation tasks.
3. Still to set up before WP0: a funded compute account with programmatic API and SSH access configured for remote execution.

## References

1. C. Hawthorne et al., "Onsets and Frames: Dual-Objective Piano Transcription," ISMIR 2018. arXiv:1710.11153
2. C. Hawthorne et al., "Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset," ICLR 2019 [verify]. arXiv:1810.12247
3. Q. Kong, B. Li, X. Song, Y. Wan, Y. Wang, "High-resolution Piano Transcription with Pedals by Regressing Onset and Offset Times," IEEE/ACM TASLP 29, 2021 [verify]. arXiv:2010.01815
4. K. Toyama et al., "Automatic Piano Transcription with Hierarchical Frequency-Time Transformer," ISMIR 2023. arXiv:2307.04305
5. D. Edwards, S. Dixon, E. Benetos, A. Maezawa, Y. Kusaka, "A Data-Driven Analysis of Robust Automatic Piano Transcription," IEEE SPL 2024 [verify]. arXiv:2402.01424
6. J. Gardner, I. Simon, E. Manilow, C. Hawthorne, J. Engel, "MT3: Multi-Task Multitrack Music Transcription," ICLR 2022. arXiv:2111.03017
7. S. Chang, E. Benetos, H. Kirchhoff, S. Dixon, "YourMT3+: Multi-instrument Music Transcription with Enhanced Transformer Architectures and Cross-dataset Stem Augmentation," 2024 [verify venue]. arXiv:2407.04822
8. S. Rouard, F. Massa, A. Défossez, "Hybrid Transformers for Music Source Separation," ICASSP 2023 [verify]. arXiv:2211.08553
9. W.-T. Lu, J.-C. Wang, Q. Kong, Y.-N. Hung, "Music Source Separation with Band-Split RoPE Transformer," 2023 [verify venue]. arXiv:2309.02612
10. J.-C. Wang, W.-T. Lu, M. Won, "Mel-Band RoFormer for Music Source Separation," ISMIR 2023 LBD. arXiv:2310.01809
11. J.-C. Wang, W.-T. Lu, J. Chen, "Mel-RoFormer for Vocal Separation and Vocal Melody Transcription," 2024. arXiv:2409.04702
12. E. Manilow, P. Seetharaman, B. Pardo, "Simultaneous Separation and Transcription of Mixtures with Multiple Polyphonic and Percussive Instruments," ICASSP 2020 [verify]. arXiv:1910.12621
13. K. W. Cheuk et al., "Jointist: Joint Learning for Multi-instrument Transcription and Its Applications," 2022 [verify venue]. arXiv:2206.10805
14. K. W. Cheuk et al., "Jointist: Simultaneous Improvement of Multi-instrument Transcription and Music Source Separation via Joint Training," 2023 [verify venue]. arXiv:2302.00286
15. D. Stoller, S. Ewert, S. Dixon, "Jointly Detecting and Separating Singing Voice: A Multi-Task Approach," 2018 [verify venue]. arXiv:1804.01650
16. S. Kum, J. Nam, "Joint Detection and Classification of Singing Voice Melody Using Convolutional Recurrent Neural Networks," Applied Sciences 9(7):1324, 2019.
17. S. Ruder, "An Overview of Multi-Task Learning in Deep Neural Networks," arXiv 2017. arXiv:1706.05098
18. J. W. Kim, J. Salamon, P. Li, J. P. Bello, "CREPE: A Convolutional Representation for Pitch Estimation," ICASSP 2018 [verify]. arXiv:1802.06182
19. H. Wei, X. Cao, T. Dan, Y. Chen, "RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music," Interspeech 2023. arXiv:2306.15412
20. J. Choi, K. Lee, "Pop2Piano: Pop Audio-based Piano Cover Generation," 2023 [verify venue]. arXiv:2211.00895
21. Z. Rafii, A. Liutkus, F.-R. Stöter, S. I. Mimilakis, R. Bittner, "MUSDB18-HQ: an uncompressed version of MUSDB18," Zenodo, 2019. Zenodo 3338373
22. Y. Zhang et al., "GTSinger: A Global Multi-Technique Singing Corpus with Realistic Music Scores for All Singing Tasks," 2024 [verify venue]. arXiv:2409.13832
