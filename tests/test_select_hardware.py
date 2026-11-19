from vapt.infra.select_hardware import BenchmarkResult, GpuPrice, select_hardware

STEPS = 40_000
VALIDATION_HOURS = 1.0
N_RUNS = 6
BUDGET = 200.0

GPU_A = BenchmarkResult(
    gpu_type="A",
    runs_per_gpu=1,
    s_per_step=1.0,
    vram_headroom_frac=0.20,
    data_wait_frac=0.05,
    fits_batch_12=True,
    in_stock_on_demand=True,
    local_disk_sufficient=True,
)
PRICE_A = GpuPrice(gpu_type="A", price_per_gpu_hour=1.0, gpus_per_pod=1)

# B is faster by more than 10% wall-clock than A, but costs more.
GPU_B = BenchmarkResult(
    gpu_type="B",
    runs_per_gpu=1,
    s_per_step=0.8,
    vram_headroom_frac=0.20,
    data_wait_frac=0.05,
    fits_batch_12=True,
    in_stock_on_demand=True,
    local_disk_sufficient=True,
)
PRICE_B = GpuPrice(gpu_type="B", price_per_gpu_hour=2.0, gpus_per_pod=1)

# C is faster than A, but by less than 10% wall-clock.
GPU_C = BenchmarkResult(
    gpu_type="C",
    runs_per_gpu=1,
    s_per_step=0.95,
    vram_headroom_frac=0.20,
    data_wait_frac=0.05,
    fits_batch_12=True,
    in_stock_on_demand=True,
    local_disk_sufficient=True,
)
PRICE_C = GpuPrice(gpu_type="C", price_per_gpu_hour=1.5, gpus_per_pod=1)

# D does not fit batch 12 at all.
GPU_D = BenchmarkResult(
    gpu_type="D",
    runs_per_gpu=1,
    s_per_step=0.5,
    vram_headroom_frac=0.20,
    data_wait_frac=0.05,
    fits_batch_12=False,
    in_stock_on_demand=True,
    local_disk_sufficient=True,
)
PRICE_D = GpuPrice(gpu_type="D", price_per_gpu_hour=1.0, gpus_per_pod=1)


def _select(benchmarks, prices, budget=BUDGET):
    return select_hardware(
        benchmarks=benchmarks,
        prices=prices,
        n_runs=N_RUNS,
        steps=STEPS,
        validation_hours=VALIDATION_HOURS,
        budget=budget,
    )


def test_dearer_option_wins_when_it_cuts_wall_clock_by_10_percent_or_more():
    result = _select([GPU_A, GPU_B], [PRICE_A, PRICE_B])
    assert result.status == "ok"
    assert result.recommendation.gpu_type == "B"
    gain = 1 - result.recommendation.wall_clock_hours / result.alternative.wall_clock_hours
    assert gain >= 0.10


def test_dearer_option_loses_when_gain_is_under_10_percent():
    result = _select([GPU_A, GPU_C], [PRICE_A, PRICE_C])
    assert result.status == "ok"
    assert result.recommendation.gpu_type == "A"


def test_alternative_is_always_the_lowest_cost_compatible_option():
    result = _select([GPU_A, GPU_B], [PRICE_A, PRICE_B])
    assert result.alternative.gpu_type == "A"
    assert result.alternative.total_cost == min(o.total_cost for o in result.all_compatible)


def test_batch_that_does_not_fit_is_excluded_not_downsized():
    result = _select([GPU_D], [PRICE_D])
    assert result.status == "escalate"
    assert "compatible" in result.reason.lower()


def test_escalates_when_nothing_fits_the_budget():
    result = _select([GPU_A, GPU_B], [PRICE_A, PRICE_B], budget=1.0)
    assert result.status == "escalate"
    assert "budget" in result.reason.lower()


def test_mixed_runs_per_gpu_needs_a_matching_single_run_benchmark():
    lone_double = BenchmarkResult(
        gpu_type="E",
        runs_per_gpu=2,
        s_per_step=1.0,
        vram_headroom_frac=0.20,
        data_wait_frac=0.05,
        fits_batch_12=True,
        in_stock_on_demand=True,
        local_disk_sufficient=True,
    )
    price_e = GpuPrice(gpu_type="E", price_per_gpu_hour=1.0, gpus_per_pod=1)
    result = _select([lone_double], [price_e])
    assert result.status == "escalate"
