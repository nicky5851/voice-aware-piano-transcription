# Compute

All data processing, training, inference and evaluation for this project run
on RunPod GPU/CPU pods. See `docs/protocol.md` §10 for the frozen compute
protocol (what may and may not be changed to fit hardware).

## Data-centre choice

The project's network volume lives in a single RunPod data centre, which
constrains which GPU types are reachable from it (a network volume cannot
be mounted across data centres).

**Chosen data centre: `EU-RO-1`** (Europe), selected on 2026-11-19 by
surveying on-demand GPU stock across all data centres for GPU types with
at least 40GB of VRAM (Kong et al.'s released model needs about 29GB at
the fixed batch size of 12; 40GB is the floor used for headroom).

At survey time, `EU-RO-1` had the widest on-demand stock of qualifying GPU
types of any surveyed data centre: four distinct types with 40GB+ VRAM
(two 80GB-class and two 96GB-class cards), high CPU-pod availability across
six CPU product tiers, and standard network-volume support. No other
surveyed data centre offered more than two qualifying GPU types. The full
survey data (every data centre checked, stock levels, prices) is kept with
the project's internal records and is not duplicated here; this file states
only the choice and its justification, which is what later work in this
repository depends on.

## Environment

- Base image: a RunPod PyTorch image (CUDA 12.x, Python 3.10/3.11), pinned
  by digest once the venv is built on a pod (recorded in
  `configs/hardware/env.yaml`).
- Dependencies are locked with `uv` into `requirements.lock`, committed to
  this repository.
- `scripts/pod/setup_env.sh` builds the locked virtual environment on a
  pod and is idempotent.

## Hardware selection for primary runs

Hardware for the primary training runs (WP4/WP5) is chosen by benchmarking
real candidates at the fixed batch size, under a budget approved in writing
before any paid full-scale training starts. The selection procedure is
`src/vapt/infra/select_hardware.py`; its rules are fixed by
`docs/protocol.md` §10.2 and are not re-derived here. The resulting quote,
decision and measured benchmark numbers are recorded in `results/compute/`
once WP4 runs.

## Reproducibility records

Every run writes a compute record (`results/compute/<run>.json`) covering
GPU model, count and driver; CUDA/PyTorch/cuDNN versions; container image
digest; CPU/RAM/disk; precision mode; throughput; wall-clock time; cost;
and any interruptions or resumes. See `docs/protocol.md` §10.4.
