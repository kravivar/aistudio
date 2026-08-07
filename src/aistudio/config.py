import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import yaml

# Load environment variables from .env if present
load_dotenv()

def load_yaml_config() -> dict:
    for filename in ["config.yml", "config.yaml"]:
        config_path = Path(filename).resolve()
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
    cfg = get_model_config(model_id_or_path)
    if cfg and "from_file" in cfg:
        from_file_path = Path(HF_HOME) / cfg["from_file"]
        if from_file_path.exists():
            return str(from_file_path.resolve())

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

def detect_model_type(model_id_or_path: str) -> str:
    """
    Inspects the resolved model path and returns the likely model type:
      - 'llm'       : directory with config.json (standard mlx_lm / HF model)
      - 'diffusion'  : standalone .safetensors / .bin file (e.g. SDXL checkpoint)
      - 'unknown'    : cannot determine (e.g. HuggingFace Hub ID not yet downloaded)
    """
    resolved = Path(resolve_model_path(model_id_or_path))

    # A directory with config.json is an LLM / HF model directory
    if resolved.is_dir():
        if (resolved / "config.json").exists():
            return "llm"
        return "unknown"

    # A standalone weight file without a neighbouring config.json is a diffusion checkpoint
    if resolved.is_file() and resolved.suffix in (".safetensors", ".bin", ".gguf"):
        if (resolved.parent / "config.json").exists():
            return "llm"   # weight file inside a proper LLM model dir
        return "diffusion"

    return "unknown"

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

    return models
