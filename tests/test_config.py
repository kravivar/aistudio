import pytest
from aistudio.config import resolve_model_path, scan_available_models

def test_resolve_model_path():
    path = resolve_model_path("non_existent_model_id")
    assert path == "non_existent_model_id"

def test_scan_available_models():
    models = scan_available_models()
    assert isinstance(models, list)
    assert len(models) > 0
    assert "id" in models[0]

def test_resolve_recursive_custom_safetensors(tmp_path, monkeypatch):
    custom_subfolder = tmp_path / "custom_safetensors"
    custom_subfolder.mkdir()
    dummy_file = custom_subfolder / "test_model.safetensors"
    dummy_file.write_text("dummy model data")

    monkeypatch.setattr("aistudio.config.MODEL_SEARCH_PATHS", [tmp_path])
    resolved = resolve_model_path("test_model.safetensors")
    assert resolved == str(dummy_file.resolve())

    scanned = scan_available_models()
    scanned_ids = [m["id"] for m in scanned]
    assert "test_model.safetensors" in scanned_ids

def test_get_model_config():
    from aistudio.config import get_model_config
    cfg = get_model_config("mlx-community/Qwen3.6-35B-A3B-4bit")
    assert cfg.get("max_tokens") == 8192
    assert cfg.get("thinking_mode") == "stream"

    cfg_gemma = get_model_config("mlx-community/gemma-4-31b-it-8bit")
    assert cfg_gemma.get("max_tokens") == 8192
    assert cfg_gemma.get("context_size") == 32768

def test_default_model_resolution():
    from aistudio.config import DEFAULT_MODEL
    assert isinstance(DEFAULT_MODEL, str)
    assert len(DEFAULT_MODEL) > 0

def test_aistudio_home_paths():
    from aistudio.config import AISTUDIO_HOME, DATA_DIR, OUTPUT_DIR, LOG_FILE
    assert AISTUDIO_HOME.name == "aistudio"
    assert "Document" in str(AISTUDIO_HOME) or "aistudio" in str(AISTUDIO_HOME)
    assert DATA_DIR.parent.name == "aistudio" or DATA_DIR.name == "webui"
    assert OUTPUT_DIR.name == "output"
    assert LOG_FILE.name == "server.log"

def test_load_yaml_config_priority():
    from aistudio.config import load_yaml_config, CONFIG_FILE_PATH
    cfg, path = load_yaml_config()
    assert isinstance(cfg, dict)
    assert path is not None
    assert path.name in ("config.yml", "config.yaml")




