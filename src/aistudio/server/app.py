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

from aistudio.config import scan_available_models, get_model_config, detect_model_type, resolve_model_path, OUTPUT_DIR
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
output_dir = OUTPUT_DIR
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

@app.get("/v1/system/status")
async def system_status():
    """
    Real-time system resource dashboard.
    Reports memory usage (MLX + MPS), CPU cores, active pipelines,
    queue depths, and throughput metrics.
    """
    return JSONResponse(content=model_manager.get_system_stats())

@app.get("/v1/internal/models")
def list_all_models_internal():
    """
    Returns ALL available models (including images) for internal pipes to consume.
    """
    available_models = scan_available_models()
    return {
        "object": "list",
        "data": available_models
    }

@app.get("/v1/models")
def list_models():
    """
    Returns available generative models to Open WebUI (LLMs, Video, and Image models).
    Filters out internal embedding models and raw shard files.
    """
    available_models = scan_available_models()
    
    filtered_models = [
        m for m in available_models 
        if m.get("type") in ["llm", "video", "image", "diffusion"]
        and not m.get("id", "").endswith((".safetensors", ".bin", ".gguf"))
    ]

    return {
        "object": "list",
        "data": filtered_models
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, req: ChatCompletionRequest):
    """
    OpenAI-compliant chat completions with streaming support.
    Resolves per-model parameters (max_tokens, temperature, thinking_mode) based on chosen req.model.

    Auto-routing:
    - Diffusion models are routed to the image pipeline (returns markdown image).
    - Video models are routed to the video pipeline (returns playable markdown video).
    - Embedding models are rejected with a helpful 400 error.
    """
    messages_dicts = [m.model_dump() for m in req.messages]

    # Extract the last user message prompt
    user_prompt = ""
    for m in reversed(messages_dicts):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                user_prompt = " ".join(
                    part.get("text", "") for part in content if part.get("type") == "text"
                )
            else:
                user_prompt = str(content)
            break

    model_type = detect_model_type(req.model)

    # ── Embedding Model Protection ──────────────────────────────────────────
    if model_type == "embedding":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{req.model}' is an embedding model. Use /v1/embeddings instead of chat completions."
        )

    # ── Auto-detect video models and route to video generation ──────────────
    if model_type == "video":
        logger.info(f"Auto-routing video model '{req.model}' from chat to video pipeline")
        if not user_prompt.strip():
            raise HTTPException(status_code=400, detail="No prompt found in messages for video generation.")

        try:
            timestamp = int(time.time())
            out_file_path = str(OUTPUT_DIR / "video" / f"chat_gen_{timestamp}.mp4")
            thumb_file_path = str(OUTPUT_DIR / "video" / f"thumb_gen_{timestamp}.png")

            cfg = get_model_config(req.model, model_type="video")
            width = int(cfg.get("width", 704))
            height = int(cfg.get("height", 480))
            fps = int(cfg.get("fps", 24))
            duration = float(cfg.get("duration", 4.0))
            num_frames = int(cfg.get("num_frames", int(duration * fps) + 1))
            steps = int(cfg.get("steps", 30))
            seed = req.seed if req.seed is not None and req.seed != -1 else int(cfg.get("seed", -1))
            if seed == -1:
                seed = int(time.time()) % 1000000
            two_stage = bool(cfg.get("two_stage", True))
            autoplay = bool(cfg.get("autoplay", True))

            async with model_manager.acquire("video"):
                video_file = await model_manager.run_in_thread(
                    video_pipeline.generate_single_scene,
                    prompt=user_prompt,
                    model_id=req.model,
                    width=width,
                    height=height,
                    fps=fps,
                    video_seconds=int(duration),
                    num_frames=num_frames,
                    steps=steps,
                    seed=seed,
                    two_stage=two_stage,
                    output_path=out_file_path
                )
                from aistudio.utils.media import extract_last_frame
                extract_last_frame(video_file, thumb_file_path)

            video_url_path = f"/static/video/{Path(video_file).name}"
            thumb_url_path = f"/static/video/{Path(thumb_file_path).name}"
            base_url = str(request.base_url).rstrip("/")
            full_video_url = f"{base_url}{video_url_path}"
            full_thumb_url = f"{base_url}{thumb_url_path}"

            autoplay_attr = "autoplay loop playsinline" if autoplay else ""
            response_text = (
                f"🎬 **Generated Video for:** *{user_prompt}*\n\n"
                f'<video controls {autoplay_attr} width="100%" poster="{full_thumb_url}" style="border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">\n'
                f'  <source src="{full_video_url}" type="video/mp4">\n'
                f'  Your browser does not support the video tag.\n'
                f'</video>\n\n'
                f"**[📥 Direct Video Link]({full_video_url})**\n\n"
                f"*Settings: {width}x{height} | {duration}s ({num_frames} frames) @ {fps}fps | Seed: {seed}*"
            )

            completion_id = f"chatcmpl-vid-{int(time.time())}"

            if req.stream:
                async def _video_stream():
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
                    _video_stream(),
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
            logger.error(f"Video generation via chat auto-route failed: {e}")
            raise HTTPException(status_code=500, detail=f"Video generation failed: {e}")

    # ── Auto-detect image/diffusion models and route to image generation ───
    if model_type in ("image", "diffusion"):
        logger.info(f"Auto-routing image model '{req.model}' from chat to image pipeline")

        if not user_prompt.strip():
            raise HTTPException(status_code=400, detail="No prompt found in messages for image generation.")

        try:
            # Run image generation in thread pool (blocking PyTorch MPS call)
            # The acquire() context ensures memory is available and queues if busy
            async with model_manager.acquire("image"):
                result = await model_manager.run_in_thread(
                    image_pipeline.generate,
                    prompt=user_prompt,
                    negative_prompt=req.negative_prompt or "blurry",
                    model_id=req.model,
                    n=1,
                    size=req.size or "1024x1024",
                    num_inference_steps=req.steps or 8,
                    guidance_scale=req.guidance or 2.0,
                    response_format="url",
                    seed=req.seed
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

    # Resolve per-model defaults from config.yml if caller did not specify them
    model_cfg = get_model_config(req.model)
    max_tokens = req.max_tokens if req.max_tokens is not None else int(model_cfg.get("max_tokens", 8192))
    temperature = req.temperature if req.temperature is not None else float(model_cfg.get("temperature", 0.7))
    thinking_mode = req.thinking_mode if req.thinking_mode is not None else model_cfg.get("thinking_mode", "stream")

    if req.stream:
        # Wrap the async generator so the semaphore is held for the entire
        # duration of the stream (not just the first token).
        async def _guarded_llm_stream():
            async with model_manager.acquire("llm"):
                async for chunk in llm_pipeline.generate_stream(
                    model_id=req.model,
                    messages=messages_dicts,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    thinking_mode=thinking_mode,
                    seed=req.seed
                ):
                    yield chunk

        return StreamingResponse(
            _guarded_llm_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        # Non-streaming: run blocking mlx_lm.generate() in thread pool
        async with model_manager.acquire("llm"):
            res = await model_manager.run_in_thread(
                llm_pipeline.generate,
                model_id=req.model,
                messages=messages_dicts,
                max_tokens=max_tokens,
                temperature=temperature,
                thinking_mode=thinking_mode,
                seed=req.seed
            )
        return JSONResponse(content=res)

@app.post("/v1/completions")
async def text_completions(req: CompletionRequest):
    """
    OpenAI-compliant standard text completions endpoint.
    Resolves per-model parameters (max_tokens, temperature, thinking_mode) based on chosen req.model.
    """
    messages_dicts = [{"role": "user", "content": req.prompt}]
    
    # Resolve per-model defaults from config.yml if caller did not specify them
    model_cfg = get_model_config(req.model)
    max_tokens = req.max_tokens if req.max_tokens is not None else int(model_cfg.get("max_tokens", 8192))
    temperature = req.temperature if req.temperature is not None else float(model_cfg.get("temperature", 0.7))
    thinking_mode = req.thinking_mode if req.thinking_mode is not None else model_cfg.get("thinking_mode", "stream")

    if req.stream:
        async def _guarded_completions_stream():
            async with model_manager.acquire("llm"):
                async for chunk in llm_pipeline.generate_stream(
                    model_id=req.model,
                    messages=messages_dicts,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    thinking_mode=thinking_mode,
                    seed=req.seed
                ):
                    yield chunk

        return StreamingResponse(
            _guarded_completions_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        async with model_manager.acquire("llm"):
            res = await model_manager.run_in_thread(
                llm_pipeline.generate,
                model_id=req.model,
                messages=messages_dicts,
                max_tokens=max_tokens,
                temperature=temperature,
                thinking_mode=thinking_mode,
                seed=req.seed
            )
        return JSONResponse(content=res)

@app.post("/v1/images/generations")
async def image_generations(req: ImageGenerationRequest):
    """
    OpenAI-compliant text-to-image generation endpoint (SDXL / PyTorch MPS).
    Queued via ModelManager to prevent memory overcommit.
    """
    try:
        default_img = get_model_config(model_type="image").get("id", "RunDiffusion/Juggernaut-XI-v11")
        target_model = req.model if req.model and req.model not in ("dall-e-2", "dall-e-3", "default") else default_img

        async with model_manager.acquire("image"):
            res = await model_manager.run_in_thread(
                image_pipeline.generate,
                prompt=req.prompt,
                negative_prompt=req.negative_prompt or "blurry",
                model_id=target_model,
                n=req.n or 1,
                size=req.size or "1024x1024",
                num_inference_steps=req.num_inference_steps or 8,
                guidance_scale=req.guidance_scale or 2.0,
                response_format=req.response_format or "b64_json",
                seed=req.seed
            )
        return JSONResponse(content=res)
    except Exception as e:
        logger.error(f"Image generation endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/video/generations")
async def video_generations(req: VideoGenerationRequest):
    """
    Custom LTX Video 2.3 multi-scene generation & timeline stitching endpoint.
    Queued via ModelManager to prevent memory overcommit.
    """
    try:
        async with model_manager.acquire("video"):
            if req.scenes and len(req.scenes) > 0:
                scenes_list = [s.model_dump() for s in req.scenes]
                res = await model_manager.run_in_thread(
                    video_pipeline.generate_multi_scene_timeline,
                    scenes=scenes_list,
                    output_path=req.output_path or str(OUTPUT_DIR / "video" / "final_movie.mp4")
                )
            elif req.prompt:
                num_frames = req.num_frames or (int((req.duration or 10) * (req.fps or 24)) + 1)
                single_scene = [{
                    "prompt": req.prompt,
                    "width": req.width or 704,
                    "height": req.height or 480,
                    "fps": req.fps or 24,
                    "video_seconds": req.duration or 10,
                    "num_frames": num_frames,
                    "steps": req.steps or 8,
                    "seed": req.seed if req.seed is not None else 42,
                    "image_path": req.image_path,
                    "images": req.images,
                    "model_id": req.model,
                    "two_stage": req.two_stage if req.two_stage is not None else True
                }]
                res = await model_manager.run_in_thread(
                    video_pipeline.generate_multi_scene_timeline,
                    scenes=single_scene,
                    output_path=req.output_path or str(OUTPUT_DIR / "video" / "final_movie.mp4")
                )
            else:
                raise HTTPException(status_code=400, detail="Either 'prompt' or 'scenes' must be provided.")

        return JSONResponse(content=res)
    except HTTPException:
        raise
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
    Queued via ModelManager since Whisper uses significant GPU memory (~3GB).
    """
    try:
        suffix = Path(file.filename).suffix if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Open WebUI hardcodes 'whisper-1' for the voice input. We must map it
        # to the local mlx model to prevent HuggingFace hub errors.
        actual_model = "mlx-community/whisper-large-v3-mlx" if model == "whisper-1" else model

        async with model_manager.acquire("audio"):
            res = await model_manager.run_in_thread(
                audio_pipeline.transcribe,
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
    
    TTS runs via subprocess (macOS `say` command) and does NOT use GPU memory,
    so it runs in the thread pool without acquiring the audio semaphore.
    This prevents TTS from being unnecessarily blocked behind Whisper transcription.
    """
    try:
        input_text = req.input
        voice = req.voice or "alloy"
        # Run in thread pool to avoid blocking the event loop (subprocess call)
        audio_bytes = await model_manager.run_in_thread(
            audio_pipeline.text_to_speech, text=input_text, voice=voice
        )
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
