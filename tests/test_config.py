"""Tests for user preferences (upscaler.config)."""

from upscaler import config


def test_defaults_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.json")
    cfg = config.load()
    assert cfg == config.DEFAULTS
    assert cfg is not config.DEFAULTS  # a copy, not the shared dict


def test_save_then_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.json")
    assert config.save(device="cuda", model="realesrgan-x2plus", output_dir="/tmp/out")
    cfg = config.load()
    assert cfg["device"] == "cuda"
    assert cfg["model"] == "realesrgan-x2plus"
    assert cfg["output_dir"] == "/tmp/out"


def test_unknown_keys_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.json")
    config.save(device="cpu", bogus="nope")
    cfg = config.load()
    assert cfg["device"] == "cpu"
    assert "bogus" not in cfg


def test_corrupt_file_falls_back(tmp_path, monkeypatch):
    p = tmp_path / "config.json"
    p.write_text("{ not valid json")
    monkeypatch.setattr(config, "CONFIG_PATH", p)
    assert config.load() == config.DEFAULTS
