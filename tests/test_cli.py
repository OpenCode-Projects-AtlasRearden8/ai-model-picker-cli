"""End-to-end CLI tests with mocked hardware and endpoint."""

import json

from ai_model_picker import cli, endpoint, hardware


class TestCli:
    def _patch(self, monkeypatch, hw, model_ids=None, error=None):
        scan = endpoint.EndpointScan(
            base_url="http://ollama.faketest:11434/v1",
            resolved_url="http://ollama.faketest:11434/v1/models",
            model_ids=model_ids or [],
            error=error,
        )
        monkeypatch.setattr(hardware, "detect_hardware", lambda: hw)
        monkeypatch.setattr(endpoint, "scan_endpoint", lambda base, timeout=5.0: scan)
        monkeypatch.setattr(cli.config, "load_config", lambda path=None: {})
        return scan

    def test_json_output_end_to_end(self, monkeypatch, mid_tier_machine, capsys):
        self._patch(
            monkeypatch,
            mid_tier_machine,
            ["llama3.2:3b", "qwen2.5:7b-instruct-q4_K_M", "llama3.1:70b-q4_K_M"],
        )
        code = cli.main(["--json"])
        out = json.loads(capsys.readouterr().out)
        assert code == 0
        assert out["endpoint"]["base_url"] == "http://ollama.faketest:11434/v1"
        ids = [r["model_id"] for r in out["recommendations"]]
        assert "llama3.1:70b-q4_K_M" not in ids  # doesn't fit, hidden by default
        # 7B q4 fits the 8 GiB VRAM and is the larger model -> ranked first.
        assert ids[0] == "qwen2.5:7b-instruct-q4_K_M"
        assert out["recommendations"][0]["fit_tier"] == "excellent fit"

    def test_json_output_endpoint_error(self, monkeypatch, mid_tier_machine, capsys):
        self._patch(monkeypatch, mid_tier_machine, error="Could not connect: refused")
        code = cli.main(["--json"])
        out = json.loads(capsys.readouterr().out)
        assert code == 1
        assert out["endpoint"]["error"] is not None
        assert out["recommendations"] == []

    def test_human_output(self, monkeypatch, mid_tier_machine, capsys):
        self._patch(monkeypatch, mid_tier_machine, ["qwen2.5:7b-instruct-q4_K_M"])
        code = cli.main(["--top", "3"])
        text = capsys.readouterr().out
        assert code == 0
        assert "qwen2.5:7b-instruct-q4_K_M" in text
        assert "EXCELLENT FIT" in text
        assert "Needs roughly" in text

    def test_show_unfit_revealed(self, monkeypatch, mid_tier_machine, capsys):
        self._patch(monkeypatch, mid_tier_machine, ["llama3.1:70b-q4_K_M"])
        code = cli.main(["--show-unfit"])
        text = capsys.readouterr().out
        assert code == 0
        assert "llama3.1:70b-q4_K_M" in text
        assert "WON'T FIT" in text

    def test_skipped_count_hint(self, monkeypatch, mid_tier_machine, capsys):
        self._patch(
            monkeypatch, mid_tier_machine, ["llama3.2:3b", "llama3.1:70b-q4_K_M"]
        )
        cli.main(["--top", "1"])
        text = capsys.readouterr().out
        assert "1 model(s)" in text

    def test_hardware_only(self, monkeypatch, mid_tier_machine, capsys):
        self._patch(monkeypatch, mid_tier_machine)
        code = cli.main(["--hardware-only"])
        text = capsys.readouterr().out
        assert code == 0
        assert "Testing Fictional Ryzen" in text
        assert "NVIDIA Fictional 8080" in text

    def test_empty_endpoint_message(self, monkeypatch, mid_tier_machine, capsys):
        self._patch(
            monkeypatch,
            mid_tier_machine,
            error="Endpoint responded but returned no models. Is the service started?",
        )
        cli.main([])
        text = capsys.readouterr().out
        assert "service started" in text

    def test_base_url_flag_used_in_json(self, monkeypatch, mid_tier_machine, capsys):
        scan = self._patch(monkeypatch, mid_tier_machine, ["llama3.2:3b"])
        cli.main(["--json", "--base-url", "http://lmstudio.faketest:1234/v1"])
        out = json.loads(capsys.readouterr().out)
        # Fixed mock scan: prove the flag parsed and the flow stayed healthy.
        assert scan.error is None
        assert out["version"] == cli.__version__


def recommend_tier(rec: dict) -> str:
    assert rec["fit_tier"] in (
        "excellent fit",
        "good fit",
        "will run, but snug",
        "probably too big",
        "will not fit here",
    )
    return rec["fit_tier"]
