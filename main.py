import argparse
import os
import sys
import subprocess
import uvicorn
from pathlib import Path
from dotenv import load_dotenv
from aistudio.config import APP_CONFIG, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_THINKING_MODE

load_dotenv()

def patch_open_webui_db(data_dir: Path, api_port: int = 8000):
    """
    Patches Open WebUI's SQLite database in ./data/webui to force enable Web Search (SearXNG), PDF Image OCR, and Image Generation.
    """
    import sqlite3, json
    db_paths = [
        data_dir / "webui.db",
        Path.home() / ".open-webui" / "data" / "webui.db",
        Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "open_webui" / "data" / "webui.db"
    ]
    for db_path in db_paths:
        if db_path.exists():
            try:
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='config'")
                    if cursor.fetchone():
                        searx_endpoint = f"http://localhost:{api_port}/v1/search?q=<query>"
                        openai_endpoint = f"http://localhost:{api_port}/v1"
                        api_key = APP_CONFIG.get("server", {}).get("api_key", "local-sk-key")
                        configs = {
                            "openai.api_base_url": json.dumps(openai_endpoint),
                            "openai.api_key": json.dumps(api_key),
                            "openai.api_base_urls": json.dumps([openai_endpoint]),
                            "openai.api_keys": json.dumps([api_key]),
                            "openai_config": json.dumps({"ENABLE_OPENAI_API": True, "OPENAI_API_BASE_URLS": [openai_endpoint], "OPENAI_API_KEYS": [api_key]}),
                            "web_search": json.dumps({"enable": True, "engine": "searxng", "searxng": {"query_url": searx_endpoint}}),
                            "rag": json.dumps({
                                "enable_web_search": True,
                                "web_search_engine": "searxng",
                                "searxng_query_url": searx_endpoint,
                                "template": "Use the following Web Search context to answer the user question:\n<context>\n{{CONTEXT}}\n</context>\n\nUser Question: {{PROMPT}}",
                                "pdf_extract_images": True,
                                "enable_ocr_text_extraction": True,
                                "content_extraction_engine": ""
                            }),
                            "images": json.dumps({"enable": True, "engine": "openai"}),
                            "ui": json.dumps({"enable_community_sharing": False, "show_admin_details": False}),
                            # Open WebUI web.search.* key mappings
                            "web.search.enable": "true",
                            "web.search.engine": '"searxng"',
                            "web.search.searxng_query_url": json.dumps(searx_endpoint),
                            # Bypass full page web scraping & vector chunking so clean search engine snippets are passed directly to the LLM
                            "web.search.bypass_embedding_and_retrieval": "true",
                            "web.search.bypass_web_loader": "true",
                            "web.search.result_count": "5",
                            "web.search.concurrent_requests": "5",
                            # Dotted key mappings for Open WebUI 0.5+ RAG
                            "rag.web_search.enable": "true",
                            "rag.web_search.engine": '"searxng"',
                            "rag.web_search.searxng_query_url": json.dumps(searx_endpoint),
                            "rag.web_search.result_count": "3",
                            "rag.template": json.dumps("Use the following Web Search context to answer the user question:\n<context>\n{{CONTEXT}}\n</context>\n\nUser Question: {{PROMPT}}"),
                            "rag.pdf_extract_images": "true",
                            "rag.enable_ocr_text_extraction": "true",
                            "rag.content_extraction_engine": '""',
                            "images.enable": "true",
                            "images.engine": '"openai"',
                            "images.openai.api_base_url": json.dumps(f"http://localhost:{api_port}/v1"),
                            "ui.enable_community_sharing": "false",
                            "ui.show_admin_details": "false",
                            "ui.default_user_role": '"admin"',
                            "ollama.enable": "false",
                            "ollama.base_urls": "[]",
                            # High-quality natural voice TTS configuration
                            "audio.tts.engine": '"openai"',
                            "audio.tts.openai.api_base_url": json.dumps(f"http://localhost:{api_port}/v1"),
                            "audio.tts.openai.api_key": json.dumps(api_key),
                            "audio.tts.voice": '"default"',
                            "audio.tts.model": '"tts-1"',
                            "audio.tts.split_on": '""',
                            # Force legacy function calling so Open WebUI executes forced RAG web search for custom model endpoints
                            # thinking_mode and max_tokens sourced from config.yml generation: section
                            "models.default_params": json.dumps({
                                "function_calling": "legacy",
                                "thinking_mode": DEFAULT_THINKING_MODE,
                                "max_tokens": DEFAULT_MAX_TOKENS,
                                "temperature": DEFAULT_TEMPERATURE,
                            })
                        }
                        for k, v in configs.items():
                            cursor.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", (k, v))
                        
                        # Maintain single admin@localhost account required for WEBUI_AUTH=False auto-login
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user'")
                        if cursor.fetchone():
                            cursor.execute("DELETE FROM user WHERE id NOT IN (SELECT min(id) FROM user WHERE email = 'admin@localhost') AND email != 'admin@localhost'")
                            cursor.execute("DELETE FROM user WHERE id NOT IN (SELECT min(id) FROM user WHERE email = 'admin@localhost')")
                            cursor.execute("UPDATE user SET role = 'admin', email = 'admin@localhost' WHERE email = 'admin@localhost'")

                        import time
                        import re
                        tools_dir = Path(__file__).parent / "src" / "aistudio" / "webui_tools"
                        if tools_dir.exists():
                            for tool_file in tools_dir.glob("*.py"):
                                if tool_file.name.startswith("__"): continue
                                content = tool_file.read_text(encoding="utf-8")
                                
                                name_match = re.search(r'title:\s*(.*)', content)
                                name = name_match.group(1).strip() if name_match else tool_file.stem.replace("_", " ").title()
                                func_id = tool_file.stem
                                
                                if "class Tools:" in content:
                                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tool'")
                                    if cursor.fetchone():
                                        cursor.execute("SELECT id FROM tool WHERE id = ?", (func_id,))
                                        if not cursor.fetchone():
                                            cursor.execute(
                                                "INSERT INTO tool (id, user_id, name, content, specs, meta, valves, updated_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                                (func_id, "", name, content, "[]", "{}", "{}", int(time.time()), int(time.time()))
                                            )
                                        else:
                                            cursor.execute("UPDATE tool SET content = ?, name = ?, updated_at = ? WHERE id = ?", (content, name, int(time.time()), func_id))
                                else:
                                    func_type = "pipe"
                                    if "class Filter:" in content: func_type = "filter"
                                    elif "class Action:" in content: func_type = "action"
                                    
                                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='function'")
                                    if cursor.fetchone():
                                        cursor.execute("SELECT id FROM function WHERE id = ?", (func_id,))
                                        if not cursor.fetchone():
                                            cursor.execute(
                                                "INSERT INTO function (id, user_id, name, type, content, meta, valves, is_active, is_global, updated_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                                (func_id, "", name, func_type, content, "{}", "{}", True, True, int(time.time()), int(time.time()))
                                            )
                                        else:
                                            cursor.execute("UPDATE function SET content = ?, name = ?, type = ?, updated_at = ? WHERE id = ?", (content, name, func_type, int(time.time()), func_id))

                        conn.commit()
                        print(f"✅ Auto-configured Web Search (SearXNG) & PDF Image OCR in Open WebUI DB at {db_path}")
            except Exception as e:
                pass

def launch_webui(api_port: int, webui_port: int = 3000):
    """
    Launches Open WebUI connected to local ai_studio API server natively via Python.
    Configured to store all database & RAG files under project ./data folder.
    """
    server_cfg = APP_CONFIG.get("server", {})
    webui_cfg = APP_CONFIG.get("webui", {})
    webui_port = webui_cfg.get("port", webui_port)
    data_dir_str = webui_cfg.get("data_dir", "./data/webui")
    data_dir = Path(data_dir_str).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    patch_open_webui_db(data_dir, api_port)

    openai_base_url = f"http://localhost:{api_port}/v1"
    print(f"🌐 Launching native Open WebUI connected to {openai_base_url} on port {webui_port}...")

    env = os.environ.copy()
    env["DATA_DIR"] = str(data_dir)
    env["OPENAI_API_BASE_URL"] = openai_base_url
    env["OPENAI_API_BASE_URLS"] = openai_base_url
    env["OPENAI_API_KEY"] = str(server_cfg.get("api_key", "local-sk-key"))
    env["OPENAI_API_KEYS"] = str(server_cfg.get("api_key", "local-sk-key"))
    env["PORT"] = str(webui_port)
    env["WEBUI_PORT"] = str(webui_port)
    env["WEBUI_AUTH"] = str(webui_cfg.get("auth", os.getenv("WEBUI_AUTH", "False")))
    env["ENABLE_SIGNUP"] = str(webui_cfg.get("enable_signup", os.getenv("ENABLE_SIGNUP", "False")))
    
    # Disable Ollama API Integration
    enable_ollama = str(webui_cfg.get("enable_ollama_api", os.getenv("ENABLE_OLLAMA_API", "False"))).capitalize()
    env["ENABLE_OLLAMA_API"] = enable_ollama
    env["OLLAMA_BASE_URL"] = ""
    env["OLLAMA_BASE_URLS"] = ""

    env["DEFAULT_USER_ROLE"] = "admin"
    env["WEBUI_DEFAULT_USER_ROLE"] = "admin"
    env["WEBUI_NAME"] = str(webui_cfg.get("name", os.getenv("WEBUI_NAME", "AI Studio")))
    env["ADMIN_EMAIL"] = str(webui_cfg.get("admin_email", os.getenv("ADMIN_EMAIL", "admin@aistudio.local")))
    env["ADMIN_PASSWORD"] = str(webui_cfg.get("admin_password", os.getenv("ADMIN_PASSWORD", "adminpassword123")))
    env["WEBUI_SECRET_KEY"] = str(webui_cfg.get("secret_key", os.getenv("WEBUI_SECRET_KEY", "ai-studio-secret")))

    # Disable Telemetry & Community Sharing
    env["ANONYMIZED_TELEMETRY"] = "False"
    env["DO_NOT_TRACK"] = "True"
    env["SCARF_NO_ANALYTICS"] = "True"
    env["ENABLE_COMMUNITY_SHARING"] = "False"
    env["SHOW_ADMIN_DETAILS"] = "False"

    # Chat Response Streaming
    env["STREAM_RESPONSE"] = str(webui_cfg.get("stream_response", True))
    env["ENABLE_STREAM_RESPONSE"] = str(webui_cfg.get("stream_response", True))

    # SearXNG Web Search Integration (All Naming Variants)
    web_search_cfg = APP_CONFIG.get("web_search", {})
    env["ENABLE_RAG_WEB_SEARCH"] = str(web_search_cfg.get("enable", True))
    env["ENABLE_WEB_SEARCH"] = str(web_search_cfg.get("enable", True))
    env["RAG_WEB_SEARCH_ENGINE"] = str(web_search_cfg.get("engine", "searxng"))
    env["WEB_SEARCH_ENGINE"] = str(web_search_cfg.get("engine", "searxng"))
    env["SEARXNG_QUERY_URL"] = str(web_search_cfg.get("query_url", f"http://localhost:{api_port}/v1/search?q=<query>"))
    env["BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL"] = "True"
    env["BYPASS_WEB_SEARCH_WEB_LOADER"] = "True"
    env["RAG_WEB_SEARCH_RESULT_COUNT"] = "3"
    env["WEB_SEARCH_RESULT_COUNT"] = "3"
    env["WEB_SEARCH_CONCURRENT_REQUESTS"] = "5"
    env["RAG_WEB_SEARCH_CONCURRENT_REQUESTS"] = "5"

    # PDF & OCR Extraction Integration
    doc_cfg = APP_CONFIG.get("document_rag", {})
    env["PDF_EXTRACT_IMAGES"] = str(doc_cfg.get("pdf_extract_images", True))
    env["ENABLE_OCR_TEXT_EXTRACTION"] = str(doc_cfg.get("enable_ocr_text_extraction", True))
    env["CONTENT_EXTRACTION_ENGINE"] = str(doc_cfg.get("content_extraction_engine", ""))

    # Image Generation Integration
    env["ENABLE_IMAGE_GENERATION"] = "True"
    env["IMAGE_GENERATION_ENGINE"] = "openai"
    env["IMAGES_OPENAI_API_BASE_URL"] = f"http://localhost:{api_port}/v1"

    # Use open-webui serve executable from current venv if available, or python module
    venv_bin = Path(sys.executable).parent
    open_webui_bin = venv_bin / "open-webui"
    
    cmd = [
        str(open_webui_bin) if open_webui_bin.exists() else "open-webui",
        "serve",
        "--port", str(webui_port),
        "--host", "0.0.0.0"
    ]

    try:
        subprocess.Popen(cmd, env=env)
        print(f"✅ Open WebUI launching at http://localhost:{webui_port}")
    except Exception as e:
        print(f"⚠️ Could not launch local open-webui: {e}")
        print(f"Ensure open-webui is installed in your Python environment: pip install open-webui")

def main():
    server_cfg = APP_CONFIG.get("server", {})
    default_host = server_cfg.get("host") or os.getenv("AISTUDIO_HOST") or os.getenv("AI_STUDIO_HOST", "0.0.0.0")
    default_port = int(server_cfg.get("port") or os.getenv("AISTUDIO_PORT") or os.getenv("AI_STUDIO_PORT", 8000))

    parser = argparse.ArgumentParser(description="ai_studio - Local Model API Server & Optional Open WebUI Launcher")
    parser.add_argument("--host", type=str, default=default_host, help="Host address to bind")
    parser.add_argument("--port", type=int, default=default_port, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--webui", action="store_true", help="Also launch Open WebUI interface")

    args = parser.parse_args()

    if args.webui:
        launch_webui(api_port=args.port)

    print(f"🚀 Starting aistudio API server on http://{args.host}:{args.port}")
    uvicorn.run("aistudio.server.app:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
