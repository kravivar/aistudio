"""
title: Video Generation Studio
author: AI Studio
version: 1.1
description: A dynamic pipe that exposes the Video generation model and allows customizing settings.
"""
import requests
import json
from typing import List, Union, Generator, Iterator, Optional
from pydantic import BaseModel, Field

class Pipe:
    class Valves(BaseModel):
        api_base_url: str = Field(
            default="http://localhost:3001/v1",
            description="The base URL for the AI Studio API"
        )
        
    class UserValves(BaseModel):
        duration: int = Field(default=10, description="Video Duration (seconds)")
        width: int = Field(default=704, description="Video Width")
        height: int = Field(default=480, description="Video Height")
        fps: int = Field(default=24, description="Frames Per Second")
        steps: int = Field(default=8, description="Inference Steps")

    def __init__(self):
        self.type = "pipe"
        self.id = "video_studio"
        self.name = "Video Studio"
        self.valves = self.Valves()
        self.user_valves = self.UserValves()

    def pipes(self) -> List[dict]:
        # Expose a single clean model entry for Video Generation
        return [{"id": "video-studio-ltx", "name": "🎥 LTX Video Studio"}]

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[callable] = None,
        __task__: Optional[str] = None,
    ) -> Union[str, Generator, Iterator]:
        
        # Bypassing background tasks (like title generation)
        if __task__:
            return "Video generation bypassed for background task."

        if __user__ and "valves" in __user__:
            duration = __user__["valves"].duration
            width = __user__["valves"].width
            height = __user__["valves"].height
            fps = __user__["valves"].fps
            steps = __user__["valves"].steps
        else:
            duration = self.user_valves.duration
            width = self.user_valves.width
            height = self.user_valves.height
            fps = self.user_valves.fps
            steps = self.user_valves.steps

        messages = body.get("messages", [])
        if not messages:
            yield "No messages provided."
            return
            
        user_prompt = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, list):
                    user_prompt = " ".join(part.get("text", "") for part in content if part.get("type") == "text")
                else:
                    user_prompt = str(content)
                break

        if __event_emitter__:
            await __event_emitter__({"type": "status", "data": {"description": "🎬 Generating video via LTX...", "done": False}})

        payload = {
            "prompt": user_prompt,
            "duration": duration,
            "width": width,
            "height": height,
            "fps": fps,
            "steps": steps,
            "output_path": "./output/video/generated_video.mp4"
        }

        try:
            # Note: We fallback to checking 8000 if 3001 fails, just in case
            api_url = f"{self.valves.api_base_url}/video/generations"
            if "localhost:8000" not in api_url and "127.0.0.1:8000" not in api_url:
                try:
                    # Quick check if 3001 is responsive for video endpoint
                    requests.options(api_url, timeout=1)
                except requests.exceptions.ConnectionError:
                    api_url = "http://localhost:8000/v1/video/generations"

            response = requests.post(api_url, json=payload, timeout=600)
            if response.status_code == 200:
                data = response.json()
                video_url = data.get("final_video_url", "")
                
                # Format the link cleanly
                base = api_url.split("/v1")[0]
                full_video_url = f"{base}{video_url}"
                
                if __event_emitter__:
                    await __event_emitter__({"type": "status", "data": {"description": "✨ Video generation complete!", "done": True}})
                
                yield f"Here is your generated video!\n\n**[▶️ Click to Watch Video]({full_video_url})**\n\n*Settings: {width}x{height} | {duration}s | {fps} fps | {steps} steps*"
            else:
                yield f"❌ Video generation error: {response.text}"
        except Exception as e:
            yield f"❌ Failed to reach video generator API: {e}"
