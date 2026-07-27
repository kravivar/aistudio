import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import yaml

# Load environment variables from .env if present
load_dotenv()

def load_yaml_config() -> dict:
    config_path = Path("./config.yml").resolve()
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}

APP_CONFIG = load_yaml_config()
_server_cfg = APP_CONFIG.get("server", {})
_default_models = _server_cfg.get("default_models") or []
_first_model = _default_models[0] if _default_models and isinstance(_default_models, list) else {}

HF_HOME = _server_cfg.get("hf_home") or os.getenv("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
HF_HOME = os.path.expanduser(HF_HOME)
os.environ["HF_HOME"] = HF_HOME

def get_model_config(model_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns configured parameters (max_tokens, temperature, thinking_mode) for a specific model_id 
    from `server.default_models` in config.yml.
    Matches exact ID or substring. If model_id is not specified or not listed in config,
    falls back to the first model entry in default_models or default values.
    """
    default_models = _server_cfg.get("default_models") or []
    if model_id and isinstance(default_models, list):
        for m in default_models:
            if isinstance(m, dict):
                mid = m.get("id", "")
                if mid == model_id or mid in model_id or model_id in mid:
                    return m

    if isinstance(default_models, list) and default_models and isinstance(default_models[0], dict):
        return default_models[0]

    return {
        "id": model_id or "default",
        "max_tokens": 8192,
        "temperature": 0.7,
        "thinking_mode": "stream"
    }

# Fallback LLM generation defaults
DEFAULT_MODEL: str = str(_first_model.get("id", "mlx-community/Qwen3.6-35B-A3B-4bit"))
DEFAULT_MAX_TOKENS: int = int(_first_model.get("max_tokens", 8192))
DEFAULT_TEMPERATURE: float = float(_first_model.get("temperature", 0.7))
DEFAULT_THINKING_MODE: str = str(_first_model.get("thinking_mode", "stream"))

# Optional directory for custom local checkpoints / .safetensors files
CUSTOM_MODELS_DIR = os.getenv("AI_STUDIO_MODELS_DIR", "./models")

# Automatically gather search paths (HF_HOME, custom models dir, ./models) without needing manual duplication
_search_candidates = [
    CUSTOM_MODELS_DIR,
    HF_HOME,
    "./models",
]
# Support optional colon-separated extra paths if provided
extra_paths_env = os.getenv("AI_STUDIO_MODEL_SEARCH_PATHS", "")
if extra_paths_env:
    _search_candidates.extend(extra_paths_env.split(":"))

MODEL_SEARCH_PATHS: List[Path] = []
_seen = set()
for p in _search_candidates:
    if p and p.strip():
        resolved = Path(p.strip()).expanduser().resolve()
        if str(resolved) not in _seen:
            _seen.add(str(resolved))
            MODEL_SEARCH_PATHS.append(resolved)

def resolve_model_path(model_id_or_path: str) -> str:
    """
    Checks if model_id_or_path is an absolute path or exists within any 
    configured external drive / local search paths (including subfolders like $HF_HOME/custom_safetensors).
    """
    candidate = Path(model_id_or_path).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    
    for search_dir in MODEL_SEARCH_PATHS:
        potential_path = search_dir / model_id_or_path
        if potential_path.exists():
            return str(potential_path.resolve())

        # Support subfolder lookups like $HF_HOME/custom_safetensors/model.safetensors
        if search_dir.exists():
            matches = list(search_dir.rglob(model_id_or_path))
            if matches:
                return str(matches[0].resolve())
            
    return model_id_or_path

def scan_available_models() -> List[Dict[str, Any]]:
    """
    Scans search paths recursively for local models (LLM directories, safetensors, whisper models).
    """
    models = []
    seen_ids = set()

    # Pre-defined default models if network/download needed
    defaults = _server_cfg.get("default_models") or [
        {"id": "mlx-community/gemma-2-9b-it-4bit", "owned_by": "mlx-community", "type": "llm"}
    ]

    for d in defaults:
        models.append({"id": d["id"], "object": "model", "owned_by": d["owned_by"], "type": d["type"]})
        seen_ids.add(d["id"])

    for search_dir in MODEL_SEARCH_PATHS:
        if not search_dir.exists():
            continue
        try:
            for path in search_dir.rglob("*"):
                # Skip hidden files, HF snapshot commit hashes, refs, blobs, and .no_exist markers
                parts = set(path.parts)
                if (
                    path.name.startswith(".")
                    or any(p in parts for p in ["snapshots", "refs", "blobs", ".no_exist"])
                    or (len(path.name) == 40 and all(c in "0123456789abcdefABCDEF" for c in path.name))
                ):
                    continue
                model_name = path.name
                if model_name not in seen_ids:
                    if path.is_file() and path.suffix in [".safetensors", ".bin", ".gguf", ".pt"]:
                        models.append({
                            "id": model_name,
                            "object": "model",
                            "owned_by": "local",
                            "path": str(path.resolve())
                        })
                        seen_ids.add(model_name)
                    elif path.is_dir() and (path / "config.json").exists():
                        models.append({
                            "id": model_name,
                            "object": "model",
                            "owned_by": "local",
                            "path": str(path.resolve())
                        })
                        seen_ids.add(model_name)
        except Exception:
            pass

    return models
