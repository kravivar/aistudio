import argparse
import os
import sys
import subprocess
import uvicorn
from pathlib import Path
from dotenv import load_dotenv
from aistudio.config import APP_CONFIG, DEFAULT_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_THINKING_MODE, AISTUDIO_HOME, DATA_DIR, LOG_LEVEL, WORKERS

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
                            "ui.show_changelog": "false",
                            "ui.welcome_modal": "false",
                            "ui.pending_user_overlay": "false",
                            "banners": "[]",
                            "notifications": "[]",
                            "ollama.enable": "false",
                            "ollama.base_urls": "[]",
                            # High-quality natural voice TTS configuration
                            "audio.tts.engine": '"openai"',
                            "audio.tts.openai.api_base_url": json.dumps(f"http://localhost:{api_port}/v1"),
                            "audio.tts.openai.api_key": json.dumps(api_key),
                            "audio.tts.voice": '"default"',
                            "audio.tts.model": '"tts-1"',
                            "audio.tts.split_on": '""',
                            "default_models": json.dumps(DEFAULT_MODEL),
                            "ui.default_models": json.dumps(DEFAULT_MODEL),
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
                        # Pre-acknowledge onboarding and welcome popups so no blocking modals appear
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user'")
                        if cursor.fetchone():
                            admin_info = json.dumps({"onboarding_completed": True, "show_changelog": False, "welcome_modal_dismissed": True})
                            admin_settings = json.dumps({"ui": {"show_changelog": False, "notifications": False, "theme": "dark"}})
                            cursor.execute("DELETE FROM user WHERE id NOT IN (SELECT min(id) FROM user WHERE email = 'admin@localhost') AND email != 'admin@localhost'")
                            cursor.execute("DELETE FROM user WHERE id NOT IN (SELECT min(id) FROM user WHERE email = 'admin@localhost')")
                            cursor.execute("UPDATE user SET role = 'admin', email = 'admin@localhost', info = ?, settings = ? WHERE email = 'admin@localhost'", (admin_info, admin_settings))

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
                                                (func_id, "", name, func_type, content, "{}", "{}", 1, 1, int(time.time()), int(time.time()))
                                            )
                                        else:
                                            cursor.execute("UPDATE function SET content = ?, name = ?, type = ?, is_active = 1, is_global = 1, updated_at = ? WHERE id = ?", (content, name, func_type, int(time.time()), func_id))

                        conn.commit()
                        print(f"✅ Auto-configured Web Search (SearXNG) & PDF Image OCR in Open WebUI DB at {db_path}")
            except Exception as e:
                pass

def patch_pypdf_parser():
    """
    Safely patch PyPDFParser and RapidOCRBlobParser to prevent reshape/decoding errors,
    resolve rapidocr_onnxruntime tuple output incompatibilities, and ensure all 100+ pages
    of large scanned/OCR PDFs are fully parsed.
    """
    try:
        import sys
        import io
        import logging
        from typing import cast, Any
        import pypdf
        from PIL import Image
        import numpy as np
        import langchain_community.document_loaders.parsers.pdf as pdf_parser
        import langchain_community.document_loaders.parsers.images as img_parser
        from langchain_core.documents.base import Blob, Document

        # Alias rapidocr_onnxruntime for modules looking for 'rapidocr'
        try:
            import rapidocr_onnxruntime
            sys.modules["rapidocr"] = rapidocr_onnxruntime
        except ImportError:
            pass

        # Fix RapidOCRBlobParser._analyze_image to handle rapidocr_onnxruntime return tuple
        def safe_analyze_image(self, img):
            if not hasattr(self, "_ocr_engine") or self._ocr_engine is None:
                try:
                    from rapidocr_onnxruntime import RapidOCR
                    self._ocr_engine = RapidOCR()
                except Exception as e:
                    logging.warning(f"Failed to initialize RapidOCR: {e}")
                    return ""
            try:
                res, _ = self._ocr_engine(np.array(img))
                if res:
                    return "\n".join([line[1] for line in res if len(line) > 1 and line[1]]).strip()
            except Exception as e:
                logging.warning(f"RapidOCR analysis error: {e}")
            return ""

        img_parser.RapidOCRBlobParser._analyze_image = safe_analyze_image

        def safe_extract_images_from_page(self, page: pypdf._page.PageObject) -> str:
            if not self.images_parser:
                return ""
            try:
                resources = cast(dict, page.get("/Resources", {}))
                if not resources or "/XObject" not in resources:
                    return ""
                xObject = resources["/XObject"]
                if hasattr(xObject, "get_object"):
                    xObject = xObject.get_object()
                if not xObject:
                    return ""
            except Exception as e:
                logging.warning(f"Error accessing PDF page XObjects: {e}")
                return ""

            images = []
            for obj in xObject:
                try:
                    obj_item = xObject[obj]
                    if hasattr(obj_item, "get_object"):
                        obj_item = obj_item.get_object()

                    if obj_item.get("/Subtype") == "/Image":
                        img_filter_obj = obj_item.get("/Filter")
                        if isinstance(img_filter_obj, pypdf.generic._base.NameObject):
                            img_filter = img_filter_obj[1:]
                        elif isinstance(img_filter_obj, list) and len(img_filter_obj) > 0:
                            img_filter = img_filter_obj[0][1:]
                        else:
                            img_filter = ""

                        np_image: Any = None
                        data = obj_item.get_data()
                        if img_filter in pdf_parser._PDF_FILTER_WITHOUT_LOSS:
                            height, width = obj_item.get("/Height"), obj_item.get("/Width")
                            arr = np.frombuffer(data, dtype=np.uint8)
                            if height and width and arr.size >= (height * width) and (arr.size % (height * width) == 0):
                                np_image = arr.reshape(height, width, -1)
                            else:
                                try:
                                    np_image = np.array(Image.open(io.BytesIO(data)))
                                except Exception:
                                    pass
                        elif img_filter in pdf_parser._PDF_FILTER_WITH_LOSS:
                            try:
                                np_image = np.array(Image.open(io.BytesIO(data)))
                            except Exception:
                                pass
                        else:
                            try:
                                np_image = np.array(Image.open(io.BytesIO(data)))
                            except Exception:
                                pass

                        if np_image is not None:
                            image_bytes = io.BytesIO()
                            Image.fromarray(np_image).save(image_bytes, format="PNG")
                            if image_bytes.getbuffer().nbytes == 0:
                                continue

                            blob = Blob.from_data(image_bytes.getvalue(), mime_type="image/png")
                            # Safely fetch OCR results without calling next() directly (prevents StopIteration exception in generators)
                            ocr_results = list(self.images_parser.lazy_parse(blob))
                            if ocr_results:
                                image_text = ocr_results[0].page_content
                                if image_text and image_text.strip():
                                    images.append(
                                        pdf_parser._format_inner_image(blob, image_text, self.images_inner_format)
                                    )
                except Exception as e:
                    logging.warning(f"Skipping malformed or unsupported PDF image stream '{obj}': {e}")
                    continue

            return pdf_parser._FORMAT_IMAGE_STR.format(
                image_text=pdf_parser._JOIN_IMAGES.join(filter(None, images))
            )

        def safe_lazy_parse(self, blob: Blob):
            """
            Per-page resilient PDF parser loop so that errors on individual pages do not abort parsing of remaining pages.
            """
            with blob.as_bytes_io() as pdf_file_obj:
                pdf_reader = pypdf.PdfReader(pdf_file_obj, password=self.password)
                doc_metadata = pdf_parser._purge_metadata(
                    {"producer": "PyPDF", "creator": "PyPDF", "creationdate": ""}
                    | cast(dict, pdf_reader.metadata or {})
                    | {
                        "source": blob.source,
                        "total_pages": len(pdf_reader.pages),
                    }
                )
                single_texts = []
                for page_number, page in enumerate(pdf_reader.pages):
                    try:
                        text_from_page = pdf_parser._extract_text_from_page(page=page) or ""
                    except Exception:
                        text_from_page = ""

                    try:
                        images_from_page = self.extract_images_from_page(page) or ""
                    except Exception:
                        images_from_page = ""

                    all_text = pdf_parser._merge_text_and_extras(
                        [images_from_page], text_from_page
                    ).strip()

                    if self.mode == "page":
                        page_label = ""
                        try:
                            if hasattr(pdf_reader, "page_labels") and pdf_reader.page_labels and page_number < len(pdf_reader.page_labels):
                                page_label = pdf_reader.page_labels[page_number]
                        except Exception:
                            pass
                        yield Document(
                            page_content=all_text,
                            metadata=pdf_parser._validate_metadata(
                                doc_metadata
                                | {
                                    "page": page_number,
                                    "page_label": page_label,
                                }
                            ),
                        )
                    else:
                        single_texts.append(all_text)

                if self.mode == "single":
                    yield Document(
                        page_content=self.pages_delimiter.join(single_texts),
                        metadata=pdf_parser._validate_metadata(doc_metadata),
                    )

        pdf_parser.PyPDFParser.extract_images_from_page = safe_extract_images_from_page
        pdf_parser.PyPDFParser.lazy_parse = safe_lazy_parse
    except Exception as e:
        print(f"⚠️ PyPDFParser patch warning: {e}")

def launch_webui(api_port: int, webui_port: int = 3000):
    """
    Launches Open WebUI connected to local ai_studio API server natively via Python.
    Configured to store all database & RAG files under project ./data folder.
    """
    patch_pypdf_parser()
    server_cfg = APP_CONFIG.get("server", {})
    webui_cfg = APP_CONFIG.get("webui", {})
    webui_port = webui_cfg.get("port", webui_port)
    data_dir_str = webui_cfg.get("data_dir")
    
    if data_dir_str:
        if data_dir_str.startswith("~"):
            data_dir = Path(data_dir_str).expanduser().resolve()
        elif data_dir_str.startswith("."):
            data_dir = (AISTUDIO_HOME / data_dir_str.strip("./")).resolve()
        else:
            data_dir = Path(data_dir_str).resolve()
    else:
        data_dir = DATA_DIR
        
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
    env["DEFAULT_MODELS"] = DEFAULT_MODEL
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

    # Increase RAG Retrieval Capacity for Large Documents (Dossiers across 100+ pages)
    top_k = str(doc_cfg.get("top_k", 20))
    env["RAG_TOP_K"] = top_k
    env["RAG_TOP_K_RERANKER"] = top_k
    env["RAG_FULL_CONTEXT"] = str(doc_cfg.get("full_context", True))

    # Image Generation Integration
    env["ENABLE_IMAGE_GENERATION"] = "True"
    env["IMAGE_GENERATION_ENGINE"] = "openai"
    env["IMAGES_OPENAI_API_BASE_URL"] = f"http://localhost:{api_port}/v1"

    # Locate Open WebUI frontend build directory dynamically
    import sys
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS') or '__compiled__' in globals():
        # Running as compiled Nuitka binary inside macOS App Bundle
        # Executable is at AI Studio.app/Contents/MacOS/main
        # We need to point to AI Studio.app/Contents/Resources/open_webui/frontend
        executable_path = Path(sys.executable)
        frontend_dir = executable_path.parent.parent / "Resources" / "open_webui" / "frontend"
    else:
        # Running natively in dev mode
        import open_webui
        frontend_dir = Path(open_webui.__file__).parent / "frontend"
        
    env["FRONTEND_BUILD_DIR"] = str(frontend_dir)

    # Update os.environ so Open WebUI picks up the settings natively
    os.environ.update(env)

    print(f"✅ Open WebUI launching natively at http://localhost:{webui_port}")
    import threading
    import uvicorn
    
    def run_webui():
        # Import inside the thread so it picks up the patched os.environ
        from open_webui.main import app as webui_app
        from fastapi import Request, HTTPException
        from aistudio.server.app import stream_video_range, OUTPUT_DIR

        @webui_app.get("/static/video/{filename}")
        @webui_app.get("/output/video/{filename}")
        async def get_webui_video_stream(request: Request, filename: str):
            server_dir = Path(__file__).parent
            candidates = [
                (OUTPUT_DIR / "video" / filename).resolve(),
                (Path.cwd() / "output" / "video" / filename).resolve(),
                (server_dir / "output" / "video" / filename).resolve(),
                (Path.home() / "Documents" / "aistudio" / "output" / "video" / filename).resolve(),
            ]
            for video_path in candidates:
                print(f'Checking {video_path}: {video_path.exists()} {video_path.is_file()}')
                if video_path.exists() and video_path.is_file():
                    return stream_video_range(request, video_path)
            raise HTTPException(status_code=404, detail=f"Video file '{filename}' not found")
        # Move the routes to the front so they aren't overshadowed by Open WebUI's static mount
        route1 = webui_app.routes.pop()
        route2 = webui_app.routes.pop()
        webui_app.routes.insert(0, route2)
        webui_app.routes.insert(0, route1)

        uvicorn.run(webui_app, host="0.0.0.0", port=webui_port, log_level=LOG_LEVEL.lower())

        
    t = threading.Thread(target=run_webui, daemon=True)
    t.start()

def main():
    server_cfg = APP_CONFIG.get("server", {})
    default_host = server_cfg.get("host") or os.getenv("AISTUDIO_HOST") or os.getenv("AI_STUDIO_HOST", "0.0.0.0")
    default_port = int(server_cfg.get("port") or os.getenv("AISTUDIO_PORT") or os.getenv("AI_STUDIO_PORT", 8000))

    parser = argparse.ArgumentParser(description="ai_studio - Local Model API Server & Optional Open WebUI Launcher")
    parser.add_argument("--host", type=str, default=default_host, help="Host address to bind")
    parser.add_argument("--port", type=int, default=default_port, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--webui", action="store_true", help="Launch Open WebUI backend (no native window)")
    parser.add_argument("--nativeui", action="store_true", help="Launch Open WebUI and open the native macOS desktop window (Default)")
    parser.add_argument("--api-only", action="store_true", help="Run only the API server (disables WebUI and Native UI)")

    args = parser.parse_args()

    # Determine UI modes
    use_nativeui = True
    use_webui = True

    if args.api_only:
        use_nativeui = False
        use_webui = False
    elif args.webui and not args.nativeui:
        # Explicitly requested webui only
        use_nativeui = False
        use_webui = True

    import threading
    import time
    
    print(f"🚀 Starting aistudio API server on http://{args.host}:{args.port}")
    def run_backend():
        # Import inside thread to avoid block
        import uvicorn
        uvicorn.run("aistudio.server.app:app", host=args.host, port=args.port, reload=args.reload, log_level=LOG_LEVEL.lower(), workers=WORKERS)
        
    t = threading.Thread(target=run_backend, daemon=True)
    t.start()

    if use_webui:
        launch_webui(api_port=args.port)
        
    if use_nativeui:
        # Wait for Open WebUI to fully bind and respond before opening the GUI
        print("🖥️  Waiting for Open WebUI to initialize...")
        import webview
        import urllib.request
        import urllib.error
        
        webui_cfg = APP_CONFIG.get("webui", {})
        webui_port = webui_cfg.get("port", 3000)
        api_port = args.port
        url = f"http://localhost:{webui_port}"
        
        # Dual-healthcheck gate: Wait until BOTH AI Studio (/v1/models) AND Open WebUI (/api/config) are responding
        api_ready = False
        webui_ready = False
        max_retries = 40
        
        for i in range(max_retries):
            if not api_ready:
                try:
                    res = urllib.request.urlopen(f"http://localhost:{api_port}/v1/models", timeout=1)
                    if res.getcode() == 200:
                        api_ready = True
                except Exception:
                    pass

            if not webui_ready:
                try:
                    res = urllib.request.urlopen(f"http://localhost:{webui_port}/api/config", timeout=1)
                    if res.getcode() == 200:
                        webui_ready = True
                except Exception:
                    pass

            if api_ready and webui_ready:
                break
            time.sleep(0.5)
            
        time.sleep(1)  # Allow frontend state and Svelte stores to settle
        print("✅ Backend and WebUI APIs ready! Starting PyWebView Desktop Window...")
        
        # --- PYWEBVIEW MACOS MICROPHONE PERMISSION PATCH ---
        # By default, pywebview's WKUIDelegate does not implement requestMediaCapturePermissionFor...
        # which causes macOS 12+ to silently deny microphone requests, breaking Voice Mode.
        # We monkey-patch the PyObjC class at runtime before starting the webview!
        try:
            from webview.platforms import cocoa
            import objc
            
            def webView_requestMediaCapturePermissionForOrigin_initiatedByFrame_type_decisionHandler_(self, webview, origin, frame, captureType, decisionHandler):
                # WKPermissionDecisionGrant = 1
                decisionHandler(1)
                
            cocoa.BrowserView.BrowserDelegate.webView_requestMediaCapturePermissionForOrigin_initiatedByFrame_type_decisionHandler_ = objc.selector(
                webView_requestMediaCapturePermissionForOrigin_initiatedByFrame_type_decisionHandler_,
                signature=b"v@:@@@q@"
            )
            print("🎤 Voice Mode permissions patched successfully!")
        except Exception as e:
            print(f"⚠️ Could not patch microphone permissions: {e}")
        # ---------------------------------------------------
        import webview.menu as wm

        class DesktopBridge:
            def reload(self):
                if window:
                    window.evaluate_js("window.location.reload()")

            def hard_refresh(self):
                if window:
                    window.load_url(url)

        bridge = DesktopBridge()

        window = webview.create_window(
            "AI Studio",
            url,
            js_api=bridge,
            text_select=True,
            width=1280,
            height=850,
            min_size=(900, 600)
        )

        def inject_refresh_shortcuts():
            js_code = """
            (function() {
                // 1. Suppress blocking synchronous alert() dialogs in WKWebView
                window.alert = function(msg) {
                    console.log('[AI Studio Alert Suppressed]', msg);
                };

                // 2. Pre-populate localStorage flags to suppress first-time onboarding popups
                try {
                    localStorage.setItem('version', '0.5.20');
                    localStorage.setItem('changelog_version', '0.5.20');
                    localStorage.setItem('show_changelog', 'false');
                    localStorage.setItem('onboarding_completed', 'true');
                    localStorage.setItem('welcome_modal_dismissed', 'true');
                } catch (e) {}

                // 3. Keyboard shortcut listener for Cmd+R, Ctrl+R, F5
                window.addEventListener('keydown', function(e) {
                    if ((e.metaKey || e.ctrlKey) && e.key === 'r') {
                        e.preventDefault();
                        if (e.shiftKey) {
                            window.location.href = window.location.origin;
                        } else {
                            window.location.reload();
                        }
                    }
                    if (e.key === 'F5') {
                        e.preventDefault();
                        window.location.reload();
                    }
                });

                // 4. Inject floating refresh button if not already present
                if (!document.getElementById('aistudio-quick-refresh-btn')) {
                    const btn = document.createElement('button');
                    btn.id = 'aistudio-quick-refresh-btn';
                    btn.innerHTML = '🔄 Refresh UI';
                    btn.title = 'Refresh interface (Cmd+R)';
                    btn.style.cssText = 'position:fixed;bottom:12px;right:16px;z-index:99999;padding:6px 12px;font-size:12px;font-weight:600;color:#e2e8f0;background:rgba(30,41,59,0.85);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,0.15);border-radius:20px;cursor:pointer;box-shadow:0 4px 12px rgba(0,0,0,0.3);transition:all 0.2s ease;opacity:0.6;';
                    btn.onmouseenter = () => { btn.style.opacity = '1'; btn.style.transform = 'scale(1.05)'; };
                    btn.onmouseleave = () => { btn.style.opacity = '0.6'; btn.style.transform = 'scale(1)'; };
                    btn.onclick = () => { window.location.reload(); };
                    document.body.appendChild(btn);
                }

                // 5. Auto-dismiss any lingering modal backdrop or transient initial connection error toasts
                setTimeout(() => {
                    const closeBtns = document.querySelectorAll('button[aria-label="Close"], button.close-btn');
                    closeBtns.forEach(btn => {
                        if (btn.offsetParent !== null) btn.click();
                    });

                    const toasts = document.querySelectorAll('.toast, .toaster, [role="alert"], .alert');
                    toasts.forEach(t => {
                        if (t.innerText && (t.innerText.includes('Failed to fetch') || t.innerText.includes('Connection error') || t.innerText.includes('NetworkError'))) {
                            t.remove();
                        }
                    });
                }, 1200);
            })();
            """
            try:
                window.evaluate_js(js_code)
            except Exception:
                pass

        window.events.loaded += inject_refresh_shortcuts

        app_menu = [
            wm.Menu('AI Studio', [
                wm.MenuAction('About AI Studio', lambda: window.evaluate_js("alert('AI Studio - Apple Silicon Native Model Studio')")),
                wm.MenuSeparator(),
                wm.MenuAction('Quit AI Studio', lambda: window.destroy()),
            ]),
            wm.Menu('View', [
                wm.MenuAction('Reload Page (Cmd+R)', lambda: window.evaluate_js("window.location.reload()")),
                wm.MenuAction('Force Refresh & Clear State', lambda: window.load_url(url)),
                wm.MenuSeparator(),
                wm.MenuAction('Go Home', lambda: window.load_url(url)),
                wm.MenuSeparator(),
                wm.MenuAction('Toggle Fullscreen', lambda: window.toggle_fullscreen()),
            ])
        ]

        try:
            webview.start(menu=app_menu)
        except Exception:
            webview.start()
    else:
        # Keep main thread alive if no webview
        while True:
            time.sleep(1)

if __name__ == "__main__":
    main()
