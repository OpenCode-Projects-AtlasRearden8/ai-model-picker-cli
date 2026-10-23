"""Tests for hardware detection with mocked data sources."""

import subprocess

from ai_model_picker.hardware import (
    GIB,
    GpuInfo,
    detect_gpus,
    parse_nvidia_smi_output,
    read_ram_from_procfs,
)


class TestRamParsing:
    def test_typical_linux_meminfo(self):
        text = """MemTotal:       16386368 kB
MemFree:          338836 kB
MemAvailable:     8624336 kB
Buffers:           200000 kB
Cached:           1250000 kB
"""
        total, available = read_ram_from_procfs(text)
        assert total == (16386368 * 1024)
        assert available == (8624336 * 1024)

    def test_missing_memavailable_uses_fallback(self):
        text = """MemTotal: 8000000 kB
MemFree: 1000000 kB
Buffers: 500000 kB
Cached: 2000000 kB
"""
        total, available = read_ram_from_procfs(text)
        assert total == (8000000 * 1024)
        assert available == (3500000 * 1024)

    def test_available_never_exceeds_total(self):
        text = "MemTotal: 1000000 kB\nMemAvailable: 5000000 kB\n"
        total, available = read_ram_from_procfs(text)
        assert available == total


class TestNvidiaSmiParsing:
    def test_mib_units(self):
        text = "NVIDIA GeForce RTX 3070, 8192 MiB\n"
        gpus = parse_nvidia_smi_output(text)
        assert len(gpus) == 1
        assert gpus[0].name == "NVIDIA GeForce RTX 3070"
        assert gpus[0].vram_bytes == 8192 * 1024 * 1024

    def test_mb_units_converts(self):
        gpus = parse_nvidia_smi_output("Fake GPU, 1024 MB\n")
        assert gpus[0].vram_bytes == int(1024 * 1e6)

    def test_multiple_gpus_and_blank_lines(self):
        text = "GPU A, 8192 MiB\n\nGPU B, 8192 MiB\n"
        assert len(parse_nvidia_smi_output(text)) == 2

    def test_garbage_lines_skipped(self):
        text = "NVIDIA-smi has failed because it couldn't communicate with driver.\n"
        assert parse_nvidia_smi_output(text) == []


class TestGpuDetection:
    def test_skipped_when_nvidia_smi_missing(self):
        assert (
            detect_gpus(
                runner=lambda *a, **k: (_ for _ in ()).throw(
                    AssertionError("should not run")
                ),
                which=lambda name: None,
            )
            == []
        )

    def test_parses_successful_run(self):
        def fake_runner(cmd, **kwargs):
            result = subprocess.CompletedProcess(cmd, 0)
            result.stdout = "NVIDIA RTX 9999, 16384 MiB\n"
            result.stderr = ""
            return result

        gpus = detect_gpus(runner=fake_runner, which=lambda name: "/fake/nvidia-smi")
        assert gpus == [GpuInfo(name="NVIDIA RTX 9999", vram_bytes=16 * GIB)]

    def test_subprocess_failure_returns_empty(self):
        def boom(cmd, **kwargs):
            raise subprocess.SubprocessError("driver gone")

        assert detect_gpus(runner=boom, which=lambda name: "/fake/nvidia-smi") == []


class TestHardwareSummary:
    def test_describe_includes_gpu(self, mid_tier_machine):
        text = mid_tier_machine.describe()
        assert "3.0 GiB" not in text or "8.0 GiB RAM" in text
        assert "NVIDIA Fictional 8080" in text
        assert "VRAM" in text

    def test_describe_without_gpu(self, budget_machine):
        assert "no CUDA GPU detected" in budget_machine.describe()
        assert budget_machine.total_vram_bytes() == 0
