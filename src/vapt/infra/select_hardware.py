"""Pure hardware-selection logic for the primary training runs (protocol SS10.2).

Implements the frozen selection rule: the fastest practical compatible
configuration within an approved budget, plus a cost-efficient alternative.
This function never changes batch size, step count, precision or any other
scientific setting to make an option fit (protocol SS10.3) -- it only
chooses among GPU types, pod packing and run packing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

_MIN_VRAM_HEADROOM = 0.10
_MAX_DATA_WAIT = 0.10
_MIN_WALL_CLOCK_GAIN = 0.10
_MIN_PACKING_THROUGHPUT_GAIN = 1.30


@dataclass(frozen=True)
class BenchmarkResult:
    """One measured (GPU type, runs-per-GPU) point from scripts/benchmark.py."""

    gpu_type: str
    runs_per_gpu: int  # 1, or 2 only if two concurrent runs fit at batch 12
    s_per_step: float
    vram_headroom_frac: float
    data_wait_frac: float
    fits_batch_12: bool
    in_stock_on_demand: bool
    local_disk_sufficient: bool


@dataclass(frozen=True)
class GpuPrice:
    gpu_type: str
    price_per_gpu_hour: float
    gpus_per_pod: int


@dataclass(frozen=True)
class Option:
    gpu_type: str
    gpus_per_pod: int
    runs_per_gpu: int
    s_per_step: float
    hours_per_run: float
    wall_clock_hours: float
    gpu_hours: float
    pod_hours: float
    total_cost: float
    vram_headroom_frac: float


@dataclass(frozen=True)
class Selection:
    status: Literal["ok", "escalate"]
    recommendation: Option | None = None
    alternative: Option | None = None
    all_compatible: list[Option] = field(default_factory=list)
    reason: str | None = None


def _is_compatible(b: BenchmarkResult) -> bool:
    return (
        b.fits_batch_12
        and b.vram_headroom_frac >= _MIN_VRAM_HEADROOM
        and b.in_stock_on_demand
        and b.data_wait_frac < _MAX_DATA_WAIT
        and b.local_disk_sufficient
    )


def _packing_gain_ok(benchmarks: list[BenchmarkResult], gpu_type: str) -> bool:
    """A runs_per_gpu=2 entry needs a matched runs_per_gpu=1 entry and >=1.3x combined throughput."""
    one = next((b for b in benchmarks if b.gpu_type == gpu_type and b.runs_per_gpu == 1), None)
    two = next((b for b in benchmarks if b.gpu_type == gpu_type and b.runs_per_gpu == 2), None)
    if one is None or two is None:
        return False
    combined_throughput = 2 / two.s_per_step
    single_throughput = 1 / one.s_per_step
    return combined_throughput >= _MIN_PACKING_THROUGHPUT_GAIN * single_throughput


def _build_option(
    b: BenchmarkResult,
    price: GpuPrice,
    n_runs: int,
    steps: int,
    validation_hours: float,
    storage_cost: float,
) -> Option:
    hours_per_run = steps * b.s_per_step / 3600 + validation_hours
    effective_parallelism = price.gpus_per_pod * b.runs_per_gpu
    waves = math.ceil(n_runs / effective_parallelism)
    wall = waves * hours_per_run
    pod_hours = wall
    gpu_hours = wall * price.gpus_per_pod
    cost = pod_hours * price.gpus_per_pod * price.price_per_gpu_hour + storage_cost
    return Option(
        gpu_type=b.gpu_type,
        gpus_per_pod=price.gpus_per_pod,
        runs_per_gpu=b.runs_per_gpu,
        s_per_step=b.s_per_step,
        hours_per_run=hours_per_run,
        wall_clock_hours=wall,
        gpu_hours=gpu_hours,
        pod_hours=pod_hours,
        total_cost=cost,
        vram_headroom_frac=b.vram_headroom_frac,
    )


def select_hardware(
    benchmarks: list[BenchmarkResult],
    prices: list[GpuPrice],
    n_runs: int,
    steps: int,
    validation_hours: float,
    storage_cost: float = 0.0,
    budget: float | None = None,
) -> Selection:
    """Pick the recommended and cost-efficient alternative hardware configurations.

    `budget` is the total cost approved so far. If `budget` is omitted, the
    recommendation uses the shortest wall-clock time overall with no cost
    ceiling ("before a budget exists", protocol SS10.2 step 4). Returns
    `status="escalate"` with a `reason` if no configuration is compatible,
    or none fits the budget.
    """
    price_by_gpu = {p.gpu_type: p for p in prices}
    options: list[Option] = []

    for b in benchmarks:
        if b.runs_per_gpu == 2 and not _packing_gain_ok(benchmarks, b.gpu_type):
            continue
        if not _is_compatible(b):
            continue
        price = price_by_gpu.get(b.gpu_type)
        if price is None:
            continue
        options.append(_build_option(b, price, n_runs, steps, validation_hours, storage_cost))

    if not options:
        return Selection(
            status="escalate",
            reason=(
                "No GPU type is compatible: batch 12 must fit with >=10% VRAM "
                "headroom, be in stock on demand, keep data-wait under 10% of "
                "step time, and have enough local disk for the training cache."
            ),
        )

    affordable = [o for o in options if budget is None or o.total_cost <= budget]
    if not affordable:
        cheapest = min(o.total_cost for o in options)
        return Selection(
            status="escalate",
            reason=(
                f"No compatible configuration fits the approved budget of "
                f"${budget:.2f}. The cheapest compatible option costs ${cheapest:.2f}."
            ),
            all_compatible=sorted(options, key=lambda o: o.total_cost),
        )

    by_cost = sorted(affordable, key=lambda o: o.total_cost)
    recommendation = by_cost[0]
    for candidate in by_cost[1:]:
        if candidate.wall_clock_hours <= recommendation.wall_clock_hours * (1 - _MIN_WALL_CLOCK_GAIN):
            recommendation = candidate

    alternative = min(affordable, key=lambda o: o.total_cost)

    return Selection(
        status="ok",
        recommendation=recommendation,
        alternative=alternative,
        all_compatible=by_cost,
    )
