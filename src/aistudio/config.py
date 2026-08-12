import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import yaml

# Load environment variables from .env if present
load_dotenv()

def get_aistudio_home() -> Path:
    """
    Determines the base directory for AI Studio (configs, data, outputs, logs).
    Priority:
      1. AI_STUDIO_HOME environment variable
      2. ~/Document/aistudio if ~/Document exists
      3. ~/Documents/aistudio
    """
    if os.getenv("AI_STUDIO_HOME"):
        return Path(os.getenv("AI_STUDIO_HOME")).expanduser().resolve()
    
    home = Path.home()
    if (home / "Document").exists() and not (home / "Documents").exists():
        return (home / "Document" / "aistudio").resolve()
    return (home / "Documents" / "aistudio").resolve()

AISTUDIO_HOME: Path = get_aistudio_home()
DATA_DIR: Path = AISTUDIO_HOME / "data" / "webui"
OUTPUT_DIR: Path = AISTUDIO_HOME / "output"
LOG_FILE: Path = AISTUDIO_HOME / "server.log"

# Ensure essential directories exist
try:
    AISTUDIO_HOME.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "video").mkdir(parents=True, exist_ok=True)
except Exception:
    pass

def load_yaml_config() -> dict:
    search_locations = [
        AISTUDIO_HOME / "config.yml",
        AISTUDIO_HOME / "config.yaml",
        Path.home() / "Document" / "aistudio" / "config.yml",
        Path.home() / "Document" / "aistudio" / "config.yaml",
        Path.home() / "Documents" / "aistudio" / "config.yml",
        Path.home() / "Documents" / "aistudio" / "config.yaml",
        Path("config.yml").resolve(),
        Path("config.yaml").resolve(),
    ]
    for config_path in search_locations:
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
    return {}

APP_CONFIG = load_yaml_config()
_server_cfg = APP_CONFIG.get("server", {})
_webui_cfg = APP_CONFIG.get("webui", {})
_default_models = _server_cfg.get("default_models") or []

_configured_default_model = (
    _server_cfg.get("default_model")
    or _webui_cfg.get("default_model")
)

if not _configured_default_model:
    for m in _default_models:
        if isinstance(m, dict) and m.get("type") == "llm":
            _configured_default_model = m.get("id")
            break

_first_model = _default_models[0] if _default_models and isinstance(_default_models, list) else {}

HF_HOME = _server_cfg.get("hf_home") or os.getenv("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
HF_HOME = os.path.expanduser(HF_HOME)
os.environ["HF_HOME"] = HF_HOME

def get_model_config(model_id: Optional[str] = None, model_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns configured parameters for a specific model_id or model_type
    from `server.default_models` array in config.yml.
    """
    default_models = _server_cfg.get("default_models") or []
    if model_id and isinstance(default_models, list):
        for m in default_models:
            if isinstance(m, dict):
                mid = m.get("id", "")
                if mid == model_id or mid in model_id or model_id in mid:
                    return m.copy()

    if model_type and isinstance(default_models, list):
        for m in default_models:
            if isinstance(m, dict) and m.get("type") == model_type:
                return m.copy()

    if isinstance(default_models, list) and default_models and isinstance(default_models[0], dict):
        return default_models[0].copy()

    return {"id": model_id or "default"}

# Fallback LLM generation defaults
DEFAULT_MODEL: str = str(_configured_default_model or "mlx-community/Qwen3.6-35B-A3B-4bit")
DEFAULT_MAX_TOKENS: int = int(_first_model.get("max_tokens", 8192))
DEFAULT_TEMPERATURE: float = float(_first_model.get("temperature", 0.7))
DEFAULT_THINKING_MODE: str = str(_first_model.get("thinking_mode", "stream"))

# Optional directory for custom local checkpoints / .safetensors files
CUSTOM_MODELS_DIR = os.getenv("AI_STUDIO_MODELS_DIR", "./models")

# Automatically gather search paths (HF_HOME, custom models dir, ./models, ~/Documents/aistudio/models)
_search_candidates = [
    CUSTOM_MODELS_DIR,
    str(AISTUDIO_HOME / "models"),
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
      - Reads explicitly from config.yml first if 'type' is defined.
      - 'llm'       : directory with config.json (standard mlx_lm / HF model)
      - 'diffusion'  : standalone .safetensors / .bin file (e.g. SDXL checkpoint)
      - 'unknown'    : cannot determine (e.g. HuggingFace Hub ID not yet downloaded)
    """
    cfg = get_model_config(model_id_or_path)
    if cfg and "type" in cfg:
        # Map 'image' or 'video' config definitions to diffusion auto-routing
        if cfg["type"] == "image":
            return "diffusion"
        return cfg["type"]

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

    # Pre-defined default models from config.yml
    defaults = _server_cfg.get("default_models") or [
        {"id": "mlx-community/gemma-2-9b-it-4bit", "owned_by": "mlx-community", "type": "llm"}
    ]

    for d in defaults:
        models.append({"id": d["id"], "object": "model", "owned_by": d.get("owned_by", "custom"), "type": d.get("type", "llm")})
        seen_ids.add(d["id"])

    # Scan search paths for standalone custom .safetensors checkpoints
    for search_dir in MODEL_SEARCH_PATHS:
        if search_dir.exists() and search_dir.is_dir():
            for f in search_dir.rglob("*.safetensors"):
                if f.name not in seen_ids:
                    mtype = detect_model_type(str(f))
                    models.append({
                        "id": f.name,
                        "object": "model",
                        "owned_by": "custom",
                        "type": mtype
                    })
                    seen_ids.add(f.name)

    return models
