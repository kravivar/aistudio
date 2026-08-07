import os
import json
import time
import tempfile
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from aistudio.config import scan_available_models, get_model_config, detect_model_type, resolve_model_path
from aistudio.utils.logging import logger
from aistudio.server.schemas import (
    ChatCompletionRequest,
    CompletionRequest,
    ImageGenerationRequest,
    VideoGenerationRequest,
    SpeechRequest
)
from aistudio.server.manager import model_manager
from aistudio.pipelines.llm import llm_pipeline
from aistudio.pipelines.image import image_pipeline
from aistudio.pipelines.video import video_pipeline
from aistudio.pipelines.audio import audio_pipeline

app = FastAPI(
    title="AI Studio API Server",
    description="Apple Silicon Local Model Hosting for Open WebUI & Open Notebook",
    version="0.1.0"
)

# Enable CORS for Open WebUI & Open Notebook
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure output directory exists and mount as static
output_dir = Path("./output").resolve()
output_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(output_dir)), name="static")

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "aistudio",
        "description": "Apple Silicon Local Model Host Server",
        "version": "0.1.0",
        "docs": "/docs"
    }

@app.get("/v1/models")
def list_models():
    """
    Returns available models scanned dynamically from environment paths & defaults.
    """
    available_models = scan_available_models()
    return {
        "object": "list",
        "data": available_models
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, req: ChatCompletionRequest):
    """
    OpenAI-compliant chat completions with streaming support.
    Resolves per-model parameters (max_tokens, temperature, thinking_mode) based on chosen req.model.

    Auto-routing: if the selected model is a standalone .safetensors diffusion checkpoint
    (no config.json in its parent directory), the request is transparently routed to the
    image generation pipeline and the result is returned as an inline markdown image.
    """
    messages_dicts = [m.model_dump() for m in req.messages]

    # ── Auto-detect diffusion models and route to image generation ──────────
    model_type = detect_model_type(req.model)
    if model_type == "diffusion":
        logger.info(f"Auto-routing diffusion model '{req.model}' from chat to image pipeline")
        model_manager.prepare_pipeline("image")

        # Extract the last user message as the image prompt
        user_prompt = ""
        negative_prompt = "blurry"
        for m in reversed(messages_dicts):
            if m.get("role") == "user":
                content = m.get("content", "")
                # content can be a string or a list of content parts
                if isinstance(content, list):
                    user_prompt = " ".join(
                        part.get("text", "") for part in content if part.get("type") == "text"
                    )
                else:
                    user_prompt = str(content)
                break

        if not user_prompt.strip():
            raise HTTPException(status_code=400, detail="No prompt found in messages for image generation.")

        try:
            # Save to disk instead of base64 — avoids exceeding aiohttp's 128KB SSE line limit
            result = image_pipeline.generate(
                prompt=user_prompt,
                negative_prompt=req.negative_prompt or "blurry",
                model_id=req.model,
                n=1,
                size=req.size or "1024x1024",
                num_inference_steps=req.steps or 8,
                guidance_scale=req.guidance or 2.0,
                response_format="url"
            )

            # Build full image URL from the request's base URL + static path
            image_url_path = result["data"][0]["url"]   # e.g. "/static/images/gen_123_0.png"
            base_url = str(request.base_url).rstrip("/")
            full_image_url = f"{base_url}{image_url_path}"

            md_image = f"![Generated Image]({full_image_url})"
            response_text = f"Here is your generated image:\n\n{md_image}"

            completion_id = f"chatcmpl-{int(time.time())}"

            if req.stream:
                async def _image_stream():
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": response_text}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")

                    final = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                    }
                    yield f"data: {json.dumps(final)}\n\n".encode("utf-8")
                    yield b"data: [DONE]\n\n"

                return StreamingResponse(
                    _image_stream(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
                )
            else:
                return JSONResponse(content={
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": response_text},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": len(user_prompt.split()), "completion_tokens": 1, "total_tokens": len(user_prompt.split()) + 1}
                })
        except Exception as e:
            logger.error(f"Image generation via chat auto-route failed: {e}")
            raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")

    # ── Standard LLM chat completion path ───────────────────────────────────
    model_manager.prepare_pipeline("llm")

    # Resolve per-model defaults from config.yml if caller did not specify them
    model_cfg = get_model_config(req.model)
    max_tokens = req.max_tokens if req.max_tokens is not None else int(model_cfg.get("max_tokens", 8192))
    temperature = req.temperature if req.temperature is not None else float(model_cfg.get("temperature", 0.7))
    thinking_mode = req.thinking_mode if req.thinking_mode is not None else model_cfg.get("thinking_mode", "stream")

    if req.stream:
        return StreamingResponse(
            llm_pipeline.generate_stream(
                model_id=req.model,
                messages=messages_dicts,
                max_tokens=max_tokens,
                temperature=temperature,
                thinking_mode=thinking_mode
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        res = llm_pipeline.generate(
            model_id=req.model,
            messages=messages_dicts,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_mode=thinking_mode
        )
        return JSONResponse(content=res)

@app.post("/v1/completions")
async def text_completions(req: CompletionRequest):
    """
    OpenAI-compliant standard text completions endpoint.
    Resolves per-model parameters (max_tokens, temperature, thinking_mode) based on chosen req.model.
    """
    model_manager.prepare_pipeline("llm")
    messages_dicts = [{"role": "user", "content": req.prompt}]
    
    # Resolve per-model defaults from config.yml if caller did not specify them
    model_cfg = get_model_config(req.model)
    max_tokens = req.max_tokens if req.max_tokens is not None else int(model_cfg.get("max_tokens", 8192))
    temperature = req.temperature if req.temperature is not None else float(model_cfg.get("temperature", 0.7))
    thinking_mode = req.thinking_mode if req.thinking_mode is not None else model_cfg.get("thinking_mode", "stream")

    if req.stream:
        return StreamingResponse(
            llm_pipeline.generate_stream(
                model_id=req.model,
                messages=messages_dicts,
                max_tokens=max_tokens,
                temperature=temperature,
                thinking_mode=thinking_mode
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        res = llm_pipeline.generate(
            model_id=req.model,
            messages=messages_dicts,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_mode=thinking_mode
        )
        return JSONResponse(content=res)

@app.post("/v1/images/generations")
async def image_generations(req: ImageGenerationRequest):
    """
    OpenAI-compliant text-to-image generation endpoint (SDXL / PyTorch MPS).
    """
    model_manager.prepare_pipeline("image")
    try:
        res = image_pipeline.generate(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or "blurry",
            model_id=req.model or "juggernautXL_ragnarokBy.safetensors",
            n=req.n or 1,
            size=req.size or "1024x1024",
            num_inference_steps=req.num_inference_steps or 8,
            guidance_scale=req.guidance_scale or 2.0,
            response_format=req.response_format or "b64_json"
        )
        return JSONResponse(content=res)
    except Exception as e:
        logger.error(f"Image generation endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/video/generations")
async def video_generations(req: VideoGenerationRequest):
    """
    Custom LTX Video 2.3 multi-scene generation & timeline stitching endpoint.
    """
    model_manager.prepare_pipeline("video")
    try:
        if req.scenes and len(req.scenes) > 0:
            scenes_list = [s.model_dump() for s in req.scenes]
            res = video_pipeline.generate_multi_scene_timeline(
                scenes=scenes_list,
                output_path=req.output_path or "./output/video/final_movie.mp4"
            )
        elif req.prompt:
            single_scene = [{
                "prompt": req.prompt,
                "width": req.width or 704,
                "height": req.height or 480,
                "fps": req.fps or 24,
                "video_seconds": req.duration or 10,
                "steps": req.steps or 8,
                "seed": req.seed or 42,
                "image_path": req.image_path
            }]
            res = video_pipeline.generate_multi_scene_timeline(
                scenes=single_scene,
                output_path=req.output_path or "./output/video/final_movie.mp4"
            )
        else:
            raise HTTPException(status_code=400, detail="Either 'prompt' or 'scenes' must be provided.")

        return JSONResponse(content=res)
    except Exception as e:
        logger.error(f"Video generation endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form("mlx-community/whisper-large-v3-mlx"),
    language: Optional[str] = Form(None)
):
    """
    OpenAI-compliant audio transcription endpoint using mlx-whisper.
    """
    model_manager.prepare_pipeline("audio")
    try:
        suffix = Path(file.filename).suffix if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Open WebUI hardcodes 'whisper-1' for the voice input. We must map it
        # to the local mlx model to prevent HuggingFace hub errors.
        actual_model = "mlx-community/whisper-large-v3-mlx" if model == "whisper-1" else model

        res = audio_pipeline.transcribe(
            audio_file_path=tmp_path,
            model_id=actual_model,
            language=language
        )

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        text_content = res.get("text", "") if isinstance(res, dict) else str(res)
        return JSONResponse(content={"text": text_content})
    except Exception as e:
        logger.error(f"Audio transcription endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/audio/speech")
@app.post("/audio/speech")
async def audio_speech(req: SpeechRequest):
    """
    OpenAI-compliant Text-to-Speech (TTS) endpoint for Open WebUI Voice Mode.
    Uses macOS native high quality voice synthesis engine.
    """
    model_manager.prepare_pipeline("audio")
    try:
        input_text = req.input
        voice = req.voice or "alloy"
        audio_bytes = audio_pipeline.text_to_speech(text=input_text, voice=voice)
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={
                "Content-Length": str(len(audio_bytes)),
                "Cache-Control": "no-cache"
            }
        )
    except Exception as e:
        logger.error(f"Audio speech endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/search")
@app.get("/search")
async def searxng_search(
    q: Optional[str] = None,
    query: Optional[str] = None,
    text: Optional[str] = None,
    req: Request = None
):
    """
    SearXNG & Web Search endpoint for RAG & Open WebUI integrations.
    Supports q=, query=, and text= parameters as well as /v1/search and /search paths.
    """
    search_query = q or query or text
    if not search_query and req:
        search_query = req.query_params.get("q") or req.query_params.get("query") or req.query_params.get("text")
    
    if not search_query:
        return JSONResponse(content={"query": "", "results": []})

    # 1. Try DDGS live web search engine
    try:
        from ddgs import DDGS
        ddgs = DDGS()
        raw_results = list(ddgs.text(search_query, max_results=5))
        if raw_results:
            formatted_results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "link": r.get("href", ""),
                    "content": r.get("body", ""),
                    "snippet": r.get("body", ""),
                    "engine": "searxng"
                }
                for r in raw_results
            ]
            return JSONResponse(content={"query": search_query, "results": formatted_results})
    except Exception as e:
        logger.warning(f"DDGS web search notice: {e}")

    # 2. Try local SearXNG instance if running
    import httpx
    from urllib.parse import quote_plus
    searxng_raw_url = os.getenv("SEARXNG_RAW_URL", "http://localhost:8088/search")
    try:
        async with httpx.AsyncClient(timeout=4.0, follow_redirects=True) as client:
            resp = await client.get(f"{searxng_raw_url}?q={quote_plus(search_query)}&format=json")
            if resp.status_code == 200:
                data = resp.json()
                if data.get("results"):
                    return JSONResponse(content=data)
    except Exception:
        pass

    return JSONResponse(content={"query": search_query, "results": []})
