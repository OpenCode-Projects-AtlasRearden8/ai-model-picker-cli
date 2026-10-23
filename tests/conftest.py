"""Shared fixtures: fake hardware and fake endpoint data."""

import pytest

from ai_model_picker.hardware import GpuInfo, Hardware

GiB = 1024**3


@pytest.fixture
def mid_tier_machine() -> Hardware:
    """16 GiB RAM, 8 GiB free right now, 8 GiB NVIDIA GPU."""
    return Hardware(
        total_ram_bytes=16 * GiB,
        available_ram_bytes=8 * GiB,
        cpu_count=8,
        cpu_model="Testing Fictional Ryzen",
        gpus=[GpuInfo(name="NVIDIA Fictional 8080", vram_bytes=8 * GiB)],
        platform_name="Test OS x86_64",
    )


@pytest.fixture
def budget_machine() -> Hardware:
    """8 GiB RAM, 4 GiB free, no GPU."""
    return Hardware(
        total_ram_bytes=8 * GiB,
        available_ram_bytes=4 * GiB,
        cpu_count=4,
        cpu_model="Testing Old i5",
        gpus=[],
        platform_name="Test OS x86_64",
    )


FAKE_MODELS_RESPONSE = """
{
  "object": "list",
  "data": [
    {"id": "llama3.2:1b", "object": "model"},
    {"id": "llama3.2:3b", "object": "model"},
    {"id": "qwen2.5:7b-instruct-q4_K_M", "object": "model"},
    {"id": "gemma3:12b", "object": "model"},
    {"id": "llama3.1:70b-q4_K_M", "object": "model"}
  ]
}
"""


@pytest.fixture
def fake_models_response() -> str:
    return FAKE_MODELS_RESPONSE
