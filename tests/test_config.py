from config import Config


def test_config_loads_defaults_and_preserves_overrides(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"click": {"mode": "Dwell"}, "tracking": {"sensitivity": 1.4}}', encoding="utf-8")

    config = Config(str(config_file))

    assert config.data["click"]["mode"] == "Dwell"
    assert config.data["tracking"]["sensitivity"] == 1.4
    assert "video" in config.data
    assert config.data["ui"]["mirror_preview"] is True
    assert config.data["tracking"]["invert_y"] is True
