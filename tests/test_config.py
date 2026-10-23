"""Tests for TOML config loading."""

from ai_model_picker import config


class TestLoadConfig:
    def test_missing_optional_file_returns_empty(self, tmp_path):
        assert config.load_config(tmp_path / "nope.toml") == {}

    def test_none_and_no_default_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv(config.CONFIG_ENV_VAR, str(tmp_path / "also-nope.toml"))
        assert config.load_config(None) == {}

    def test_valid_file(self, tmp_path):
        path = tmp_path / "c.toml"
        path.write_text('base_url = "http://x:9/v1"\ntop = 3\n')
        assert config.load_config(path) == {"base_url": "http://x:9/v1", "top": 3}

    def test_invalid_toml_returns_empty(self, tmp_path):
        path = tmp_path / "c.toml"
        path.write_text("not = [valid\n")
        assert config.load_config(path) == {}

    def test_non_dict_toml_returns_empty(self, tmp_path):
        path = tmp_path / "c.toml"
        path.write_text("123\n")
        assert config.load_config(path) == {}

    def test_env_var_path_used(self, tmp_path, monkeypatch):
        path = tmp_path / "env.toml"
        path.write_text('base_url = "http://env-host/v1"\n')
        monkeypatch.setenv(config.CONFIG_ENV_VAR, str(path))
        assert config.load_config(None)["base_url"] == "http://env-host/v1"
