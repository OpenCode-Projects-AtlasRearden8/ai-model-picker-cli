"""Tests for model ID parsing and size estimation."""

from ai_model_picker.models import (
    QUANT_BYTES_PER_PARAM,
    GiB,
    ModelEstimate,
    estimate_model,
    ram_needed_bytes,
)


class TestParamParsing:
    def test_explicit_size_tag(self):
        est = estimate_model("qwen2.5:7b-instruct-q4_K_M")
        assert est.params_billions == 7.0
        assert est.quant == "q4"

    def test_decimal_size_tag(self):
        est = estimate_model("tiny-llm:1.5b-q8_0")
        assert est.params_billions == 1.5
        assert est.quant == "q8"

    def test_fp16_detected(self):
        est = estimate_model("some-model:8b-f16")
        assert est.quant == "f16"
        assert est.download_bytes > QUANT_BYTES_PER_PARAM["q4"] * 8e9 / 1e9 * GiB

    def test_known_family_without_size(self):
        est = estimate_model("mistral-nemo:latest")
        assert est.params_billions == 12.2

    def test_known_map_wins_for_below_1b_when_no_tag(self):
        est = estimate_model("smollm2:360m")
        assert est.params_billions == 0.36

    def test_unrecognized_model_returns_none(self):
        assert estimate_model("mystery-thing:latest") is None

    def test_empty_id_returns_none(self):
        assert estimate_model("") is None
        assert estimate_model("   ") is None

    def test_download_size_reasonable_for_llama3_8b_q4(self):
        est = estimate_model("llama3.1:8b")
        # Llama 3 8B at Q4_K_M lands near 4.9 GiB in the real world.
        assert 3.5 * GiB <= est.download_bytes <= 5.5 * GiB

    def test_quant_beats_size_only_when_tagged(self):
        e = estimate_model("m:0.5b-fp16")
        assert e.quant == "fp16"


class TestRamNeeded:
    def test_overhead_added(self):
        download = 5 * GiB
        ram = ram_needed_bytes(download)
        assert ram > download
        # floor: overhead >= 2 GiB total beyond raw weights
        assert ram - download >= int(2 * GiB)

    def test_scales_with_model_size(self):
        small = ram_needed_bytes(int(1e9))
        big = ram_needed_bytes(int(50e9))
        assert big > small * 2


class TestSizesAndUnits:
    def test_download_gib_rounding(self):
        est = ModelEstimate(
            model_id="x:1b",
            params_billions=1.0,
            quant="q4",
            download_bytes=int(0.6 * GiB),
            family="x",
        )
        assert 0.5 <= est.download_gib() <= 0.65
