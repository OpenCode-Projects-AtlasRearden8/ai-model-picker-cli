"""Tests for endpoint scanning with a mocked HTTP layer."""

import urllib.error

from ai_model_picker.endpoint import (
    EndpointScan,
    normalize_models_url,
    parse_models_json,
    scan_endpoint,
)


class TestUrlNormalization:
    def test_bare_host(self):
        assert (
            normalize_models_url("http://localhost:11434")
            == "http://localhost:11434/v1/models"
        )

    def test_v1_suffix(self):
        assert (
            normalize_models_url("http://localhost:1234/v1")
            == "http://localhost:1234/v1/models"
        )

    def test_trailing_slash(self):
        assert (
            normalize_models_url("http://localhost:1234/v1/")
            == "http://localhost:1234/v1/models"
        )

    def test_already_models_url_unchanged(self):
        assert normalize_models_url("http://x/v1/models") == "http://x/v1/models"


class TestJsonParsing:
    def test_openai_style(self, fake_models_response):
        assert "llama3.2:1b" in parse_models_json(fake_models_response)

    def test_plain_list_style(self):
        assert parse_models_json('[{"id": "a"}, {"id": "b"}]') == ["a", "b"]

    def test_ollama_native_style(self):
        body = '{"models": [{"name": "llama3"}, {"name": "phi3"}]}'
        assert parse_models_json(body) == ["llama3", "phi3"]

    def test_dedupe_preserves_order(self):
        body = '{"data": [{"id": "a"}, {"id": "b"}, {"id": "a"}]}'
        assert parse_models_json(body) == ["a", "b"]


def scan_with(feedback):
    return scan_endpoint(
        "http://fake:11434/v1", raw_fetcher=lambda base, timeout: feedback
    )


class TestScanEndpoint:
    def test_success(self, fake_models_response):
        scan = scan_with(fake_models_response)
        assert scan.error is None
        assert scan.model_ids == [
            "llama3.2:1b",
            "llama3.2:3b",
            "qwen2.5:7b-instruct-q4_K_M",
            "gemma3:12b",
            "llama3.1:70b-q4_K_M",
        ]

    def test_empty_list_reports_error(self):
        scan = scan_with('{"data": []}')
        assert scan.error is not None
        assert scan.model_ids == []

    def test_invalid_json_reports_error(self):
        scan = scan_with("<html>not json</html>")
        assert scan.error is not None
        assert "JSON" in scan.error

    def test_connection_error_is_caught(self):
        def boom(base, timeout):
            raise urllib.error.URLError("Connection refused")

        scan = scan_endpoint("http://fake/v1", raw_fetcher=boom)
        assert scan.error is not None
        assert scan.model_ids == []

    def test_http_error_mapping(self):
        def deny(base, timeout):
            raise urllib.error.HTTPError(
                "http://fake/v1/models", 401, "denied", None, None
            )  # type: ignore[arg-type]

        scan = scan_endpoint("http://fake/v1", raw_fetcher=deny)
        assert scan.error is not None
        assert "401" in scan.error
        assert "AI_MODEL_PICKER_API_KEY" in scan.error

    def test_unexpected_exception_is_caught(self):
        def weird(base, timeout):
            raise TimeoutError("hung")

        scan = scan_endpoint("http://fake/v1", raw_fetcher=weird)
        assert scan.error is not None

    def test_hard_error_scan_dataclass(self):
        s = EndpointScan(base_url="b", resolved_url="r", model_ids=[], error="x")
        assert s.model_ids == []
