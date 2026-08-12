"""
title: Video Generation Studio
author: AI Studio
version: 2.0
description: High-performance MLX video generation pipe with dynamic valves for seed, resolution, fps, length, num_frames calculation, and single/multi-image conditioning.
"""
import requests
import json
import base64
import time
from pathlib import Path
from typing import List, Union, Generator, Iterator, Optional
from pydantic import BaseModel, Field

try:
    from aistudio.config import get_model_config
    _vid_defaults = get_model_config(model_type="video")
except Exception:
    _vid_defaults = {}

DEFAULT_DURATION = float(_vid_defaults.get("duration", 4.0))
DEFAULT_FPS = int(_vid_defaults.get("fps", 24))
DEFAULT_WIDTH = int(_vid_defaults.get("width", 704))
DEFAULT_HEIGHT = int(_vid_defaults.get("height", 480))
DEFAULT_STEPS = int(_vid_defaults.get("steps", 30))
DEFAULT_TWO_STAGE = bool(_vid_defaults.get("two_stage", True))
DEFAULT_SEED = int(_vid_defaults.get("seed", -1))
DEFAULT_AUTOPLAY = bool(_vid_defaults.get("autoplay", True))

class Pipe:
    class Valves(BaseModel):
        api_base_url: str = Field(
            default="http://localhost:3001/v1",
            description="The base URL for the AI Studio API"
        )
        
    class UserValves(BaseModel):
        video_length: float = Field(default=DEFAULT_DURATION, description="Video Length (in seconds)")
        fps: int = Field(default=DEFAULT_FPS, description="Frames Per Second (e.g. 24, 30)")
        width: int = Field(default=DEFAULT_WIDTH, description="Video Width (px)")
        height: int = Field(default=DEFAULT_HEIGHT, description="Video Height (px)")
        steps: int = Field(default=DEFAULT_STEPS, description="Inference Steps")
        two_stage: bool = Field(default=DEFAULT_TWO_STAGE, description="Enable Two-Stage High Quality Pipeline")
        seed: int = Field(default=DEFAULT_SEED, description="Random Seed (-1 for random)")
        autoplay: bool = Field(default=DEFAULT_AUTOPLAY, description="Autoplay and loop video in chat player")

    def __init__(self):
        self.type = "pipe"
        self.id = "video_studio"
        self.name = "Video Studio"
        self.valves = self.Valves()
        self.user_valves = self.UserValves()

    def pipes(self) -> List[dict]:
        try:
            response = requests.get(f"{self.valves.api_base_url}/internal/models", timeout=5)
            if response.status_code == 200:
                models = response.json().get("data", [])
                video_models = [
                    {"id": f"video-studio-{m['id']}", "name": f"🎥 {m['id']}"}
                    for m in models if m.get("type") == "video" or "ltx" in m.get("id", "").lower()
                ]
                if video_models:
                    return video_models
        except Exception:
            pass
        return [
            {"id": "video-studio-dgrauet/ltx-2.3-mlx-q8", "name": "🎥 LTX-2.3 MLX (Q8)"},
            {"id": "video-studio-dgrauet/ltx-2.3-mlx-q4", "name": "🎥 LTX-2.3 MLX (Q4)"}
        ]

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[callable] = None,
        __task__: Optional[str] = None,
    ) -> Union[str, Generator, Iterator]:
        
        # Bypassing background tasks (like title generation)
        if __task__:
            yield "Video generation bypassed for background task."
            return

        # Extract User Valves
        if __user__ and "valves" in __user__:
            v = __user__["valves"]
            video_length = getattr(v, "video_length", self.user_valves.video_length)
            fps = getattr(v, "fps", self.user_valves.fps)
            width = getattr(v, "width", self.user_valves.width)
            height = getattr(v, "height", self.user_valves.height)
            steps = getattr(v, "steps", self.user_valves.steps)
            two_stage = getattr(v, "two_stage", self.user_valves.two_stage)
            valve_seed = getattr(v, "seed", self.user_valves.seed)
            autoplay = getattr(v, "autoplay", self.user_valves.autoplay)
        else:
            video_length = self.user_valves.video_length
            fps = self.user_valves.fps
            width = self.user_valves.width
            height = self.user_valves.height
            steps = self.user_valves.steps
            two_stage = self.user_valves.two_stage
            valve_seed = self.user_valves.seed
            autoplay = self.user_valves.autoplay

        # Calculate exact number of frames
        num_frames = int(video_length * fps) + 1

        # Determine Seed
        body_seed = body.get("seed", body.get("options", {}).get("seed", -1))
        if valve_seed != -1:
            seed = valve_seed
        elif body_seed is not None and body_seed != -1:
            seed = body_seed
        else:
            seed = int(time.time()) % 1000000

        messages = body.get("messages", [])
        if not messages:
            yield "No messages provided."
            return
            
        user_prompt = ""
        attached_images: List[str] = []
        try:
            from aistudio.config import OUTPUT_DIR
            out_dir = OUTPUT_DIR / "video"
        except Exception:
            out_dir = Path.home() / "Documents" / "aistudio" / "output" / "video"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Parse prompt and any attached image / images from the last user message
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, list):
                    text_parts = []
                    for idx, part in enumerate(content):
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "image_url":
                            img_url = part.get("image_url", {}).get("url", "")
                            if img_url.startswith("data:image/"):
                                try:
                                    header, encoded = img_url.split(",", 1)
                                    img_ext = "png" if "png" in header else "jpg"
                                    saved_path = out_dir / f"upload_{int(time.time())}_{idx}.{img_ext}"
                                    saved_path.write_bytes(base64.b64decode(encoded))
                                    attached_images.append(str(saved_path.resolve()))
                                except Exception as e:
                                    print(f"Failed to decode attached image: {e}")
                            elif img_url.startswith(("http://", "https://", "/")):
                                attached_images.append(img_url)
                    user_prompt = " ".join(text_parts)
                else:
                    user_prompt = str(content)

                if "images" in m and isinstance(m["images"], list):
                    for idx, img_b64 in enumerate(m["images"]):
                        try:
                            if "," in img_b64:
                                _, encoded = img_b64.split(",", 1)
                            else:
                                encoded = img_b64
                            saved_path = out_dir / f"upload_{int(time.time())}_{idx}.png"
                            saved_path.write_bytes(base64.b64decode(encoded))
                            attached_images.append(str(saved_path.resolve()))
                        except Exception:
                            pass
                break

        # Extract selected model ID
        model_id = body.get("model", "dgrauet/ltx-2.3-mlx-q8")
        if ".video-studio-" in model_id:
            model_id = model_id.split(".video-studio-", 1)[-1]
        elif model_id.startswith("video-studio-"):
            model_id = model_id[len("video-studio-"):]

        mode_str = f"Image-to-Video ({len(attached_images)} image{'s' if len(attached_images) > 1 else ''})" if attached_images else "Text-to-Video"

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"🎬 Generating {mode_str} ({num_frames} frames @ {fps}fps, {width}x{height}, seed={seed})...",
                    "done": False
                }
            })

        payload = {
            "prompt": user_prompt,
            "model": model_id,
            "duration": int(video_length),
            "num_frames": num_frames,
            "width": width,
            "height": height,
            "fps": fps,
            "steps": steps,
            "seed": seed,
            "two_stage": two_stage,
            "output_path": str(out_dir / f"gen_{int(time.time())}.mp4")
        }

        if len(attached_images) == 1:
            payload["image_path"] = attached_images[0]
        elif len(attached_images) > 1:
            payload["images"] = attached_images
            payload["image_path"] = attached_images[0]

        try:
            api_url = f"{self.valves.api_base_url}/video/generations"
            response = requests.post(api_url, json=payload, timeout=900)
            if response.status_code == 200:
                data = response.json()
                video_url = data.get("final_video_url", "")
                scenes = data.get("scenes", [])
                poster_url = ""
                if scenes and len(scenes) > 0 and "last_frame" in scenes[0]:
                    last_frame_path = scenes[0]["last_frame"]
                    frame_name = Path(last_frame_path).name
                    base = api_url.split("/v1")[0]
                    poster_url = f"{base}/static/video/{frame_name}"

                base = api_url.split("/v1")[0]
                full_video_url = f"{base}{video_url}"
                
                if __event_emitter__:
                    await __event_emitter__({"type": "status", "data": {"description": "✨ Video generation complete!", "done": True}})

                autoplay_attr = "autoplay loop playsinline" if autoplay else ""
                poster_attr = f'poster="{poster_url}"' if poster_url else ""

                video_player = f"""
<video controls {autoplay_attr} {poster_attr} width="100%" style="border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
  <source src="{full_video_url}" type="video/mp4">
  Your browser does not support the video tag.
</video>
"""
                details = f"*Settings: {width}x{height} | {video_length}s ({num_frames} frames) @ {fps}fps | Steps: {steps} | Seed: {seed}*"
                img_info = f"\n*Conditioning: {len(attached_images)} source image(s)*" if attached_images else ""

                yield f"{video_player}\n\n**[📥 Download MP4 Video]({full_video_url})**\n\n{details}{img_info}"
            else:
                yield f"❌ Video generation error ({response.status_code}): {response.text}"
        except Exception as e:
            yield f"❌ Failed to reach video generator API: {e}"

