from vapt.infra.hardware import record_hardware


def test_record_hardware_has_all_keys_and_runs_on_cpu():
    record = record_hardware()
    for key in (
        "gpu",
        "cpu",
        "ram_gb",
        "disk_free_gb",
        "image_digest",
        "runpod_pod_id",
        "runpod_gpu_type_id",
        "git_sha",
        "cuda_version",
        "cudnn_version",
        "torch_version",
        "tf32_matmul",
        "tf32_cudnn",
    ):
        assert key in record

    for key in ("name", "count", "memory_gb", "driver"):
        assert key in record["gpu"]

    for key in ("count", "model"):
        assert key in record["cpu"]

    assert isinstance(record["gpu"]["count"], int)
