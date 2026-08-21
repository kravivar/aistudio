import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import yaml

# Load environment variables from .env if present
load_dotenv()

def get_default_aistudio_home() -> Path:
    """
    Returns default home directory for AI Studio (~/Document/aistudio or ~/Documents/aistudio).
    """
    if os.getenv("AI_STUDIO_HOME"):
        return Path(os.getenv("AI_STUDIO_HOME")).expanduser().resolve()
    home = Path.home()
    if (home / "Document").exists() and not (home / "Documents").exists():
        return (home / "Document" / "aistudio").resolve()
    return (home / "Documents" / "aistudio").resolve()

DEFAULT_AISTUDIO_HOME: Path = get_default_aistudio_home()

def load_yaml_config() -> tuple[dict, Optional[Path]]:
    """
    Finds and loads config.yml.
    Priority:
      1. ./config.yml or ./config.yaml (if present in current directory, use that)
      2. ~/Documents/aistudio/config.yml or ~/Document/aistudio/config.yml (default fallback)
      3. If neither exists, creates ~/Documents/aistudio/config.yml by copying config.default.yml.
    """
    import shutil

    search_locations = [
        Path("config.yml").resolve(),
        Path("config.yaml").resolve(),
        Path.home() / "Documents" / "aistudio" / "config.yml",
        Path.home() / "Documents" / "aistudio" / "config.yaml",
        Path.home() / "Document" / "aistudio" / "config.yml",
        Path.home() / "Document" / "aistudio" / "config.yaml",
        DEFAULT_AISTUDIO_HOME / "config.yml",
        DEFAULT_AISTUDIO_HOME / "config.yaml",
    ]
    for config_path in search_locations:
        if config_path.exists() and config_path.is_file():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}, config_path
            except Exception:
                pass

    # If neither ./config.yml nor ~/Documents/aistudio/config.yml exists,
    # copy config.default.yml to ~/Documents/aistudio/config.yml and start
    target_config = DEFAULT_AISTUDIO_HOME / "config.yml"
    default_candidates = [
        Path("config.default.yml").resolve(),
        Path("config.default.yaml").resolve(),
        Path(__file__).parent.parent.parent / "config.default.yml",
        Path(__file__).parent.parent.parent / "config.default.yaml",
        Path("config.example.yml").resolve(),
    ]
    for def_path in default_candidates:
        if def_path.exists() and def_path.is_file():
            try:
                target_config.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(def_path, target_config)
                with open(target_config, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}, target_config
            except Exception:
                pass

    return {}, None

APP_CONFIG, CONFIG_FILE_PATH = load_yaml_config()
_server_cfg = APP_CONFIG.get("server", {})
_webui_cfg = APP_CONFIG.get("webui", {})
_default_models = _server_cfg.get("default_models") or []

# Base Home derived from config file location or default home
AISTUDIO_HOME: Path = CONFIG_FILE_PATH.parent if CONFIG_FILE_PATH and "aistudio" in str(CONFIG_FILE_PATH) else DEFAULT_AISTUDIO_HOME

def resolve_path_from_config(raw_path: Optional[str], default_path: Path) -> Path:
    """
    Resolves path string specified in config.yml (supporting ~ expansion and relative paths).
    """
    if not raw_path:
        return default_path.resolve()
    p = Path(raw_path).expanduser()
    if not p.is_absolute() and CONFIG_FILE_PATH:
        return (CONFIG_FILE_PATH.parent / p).resolve()
    return p.resolve()

# Path mappings defined in and loaded directly from config.yml
OUTPUT_DIR: Path = resolve_path_from_config(_server_cfg.get("output_dir"), AISTUDIO_HOME / "output")
LOG_FILE: Path = resolve_path_from_config(_server_cfg.get("log_file"), AISTUDIO_HOME / "server.log")
LOG_LEVEL: str = _server_cfg.get("log_level", "DEBUG").upper()
WORKERS: int = int(_server_cfg.get("workers", 1))
DATA_DIR: Path = resolve_path_from_config(_webui_cfg.get("data_dir"), AISTUDIO_HOME / "data" / "webui")

# Ensure directories exist
try:
    AISTUDIO_HOME.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "video").mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

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
    if not isinstance(default_models, list):
        return {}

    if model_id:
        for m in default_models:
            if isinstance(m, dict):
                mid = m.get("id", "")
                if mid and mid == model_id:
                    return m.copy()

    if model_type:
        for m in default_models:
            if isinstance(m, dict) and m.get("type") == model_type:
                return m.copy()

    return {}

# Fallback LLM generation defaults
DEFAULT_MODEL: str = str(_configured_default_model or "mlx-community/Qwen3.6-35B-A3B-4bit")
DEFAULT_MAX_TOKENS: int = int(_first_model.get("max_tokens", 8192))
DEFAULT_TEMPERATURE: float = float(_first_model.get("temperature", 0.7))
DEFAULT_THINKING_MODE: str = str(_first_model.get("thinking_mode", "stream"))

# Optional directory for custom local checkpoints
CUSTOM_MODELS_DIR = os.getenv("AI_STUDIO_MODELS_DIR", "./models")

# Gather search paths
_search_candidates = [
    CUSTOM_MODELS_DIR,
    str(AISTUDIO_HOME / "models"),
    HF_HOME,
    "./models",
]
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
    configured local search paths.
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
            
    return model_id_or_path

def detect_model_type(model_id_or_path: str) -> str:
    """
    Inspects the model ID or path and returns the matching model type ('llm', 'image', 'video', 'embedding', 'audio').
    - Reads explicitly from config.yml first if 'type' is defined.
    - Otherwise detects based on model identifier patterns.
    """
    cfg = get_model_config(model_id_or_path)
    if cfg and "type" in cfg and cfg["type"]:
        return cfg["type"]

    name_lower = model_id_or_path.lower()
    if "ltx" in name_lower or "video" in name_lower or "wan" in name_lower or "cogvideo" in name_lower:
        return "video"
    if "whisper" in name_lower or "speech" in name_lower or "audio" in name_lower:
        return "audio"
    if any(k in name_lower for k in ("minilm", "bge-", "e5-", "embed", "bert", "gte-")):
        return "embedding"
    if any(k in name_lower for k in ("diffusion", "flux", "sdxl", "juggernaut", "stable-diffusion", "sd-", "image")):
        return "image"

    return "llm"

def scan_available_models() -> List[Dict[str, Any]]:
    """
    Lists models from HuggingFace local cache (`hf cache ls`) enriched by `config.yml`.
    - Gets all model repo IDs from the Hugging Face cache.
    - If the model ID exists in config.yml, uses the configuration from config.yml.
    - If not in config.yml, auto-detects model type (llm, image, video, embedding, audio).
    - Also includes custom models explicitly specified in config.yml.
    - No standalone filesystem safetensors scanning.
    """
    config_models = _server_cfg.get("default_models") or []
    config_dict = {m["id"]: m for m in config_models if isinstance(m, dict) and "id" in m}

    # 1. Fetch cached repo IDs from HuggingFace cache (`hf cache ls`)
    cached_repos = []
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for r in cache_info.repos:
            if getattr(r, "repo_type", "model") == "model":
                cached_repos.append(r.repo_id)
    except Exception:
        hub_dir = Path(HF_HOME) / "hub"
        if hub_dir.exists():
            for d in hub_dir.glob("models--*"):
                if d.is_dir():
                    cached_repos.append(d.name.replace("models--", "").replace("--", "/"))

    models = []
    seen = set()

    # 2. For each cached repo from `hf cache ls`: use config.yml if present, else auto-detect type
    for repo_id in cached_repos:
        if repo_id in seen:
            continue

        if repo_id in config_dict:
            cfg_m = config_dict[repo_id].copy()
            models.append({
                "id": repo_id,
                "object": "model",
                "owned_by": cfg_m.get("owned_by", repo_id.split("/")[0] if "/" in repo_id else "custom"),
                "type": cfg_m.get("type", detect_model_type(repo_id))
            })
        else:
            mtype = detect_model_type(repo_id)
            models.append({
                "id": repo_id,
                "object": "model",
                "owned_by": repo_id.split("/")[0] if "/" in repo_id else "custom",
                "type": mtype
            })
        seen.add(repo_id)

    # 3. Include any additional models explicitly defined in config.yml
    for cfg_m in config_models:
        if isinstance(cfg_m, dict) and "id" in cfg_m and cfg_m["id"] not in seen:
            mid = cfg_m["id"]
            models.append({
                "id": mid,
                "object": "model",
                "owned_by": cfg_m.get("owned_by", mid.split("/")[0] if "/" in mid else "custom"),
                "type": cfg_m.get("type", detect_model_type(mid))
            })
            seen.add(mid)

    return models
