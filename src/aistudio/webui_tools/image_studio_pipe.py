"""
title: Image Generation Studio
author: AI Studio
version: 1.1
description: A dynamic pipe that exposes all available image models and allows customizing generation parameters.
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
        image_mode: bool = Field(default=True, description="Enable Image Generation (Turn off to fallback to text)")
        fallback_model: str = Field(default="gpt-4o", description="Fallback Text Model")
        negative_prompt: str = Field(
            default="blurry, distorted, low quality, text",
            description="Negative Prompt"
        )
        size: str = Field(
            default="1024x1024",
            description="Image Size (e.g. 512x512, 768x768, 1024x1024)"
        )
        steps: int = Field(default=8, description="Inference Steps")
        guidance: float = Field(default=2.0, description="Guidance Scale")

    def __init__(self):
        self.type = "pipe"
        self.id = "image_studio"
        self.name = "Image Studio"
        self.valves = self.Valves()
        self.user_valves = self.UserValves()

    def pipes(self) -> List[dict]:
        try:
            response = requests.get(f"{self.valves.api_base_url}/models", timeout=5)
            if response.status_code == 200:
                models = response.json().get("data", [])
                image_models = [
                    {"id": f"image-studio-{m['id']}", "name": f"🎨 {m['id']}"}
                    for m in models if m.get("type") in ["diffusion", "image"] or m.get("id", "").endswith((".safetensors", ".bin", ".gguf"))
                ]
                return image_models
            else:
                print(f"PIPES HTTP ERROR: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"PIPES EXCEPTION: {e}")
        return [{"id": "image-studio-error", "name": "🎨 Image Studio (Backend Offline)"}]

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[callable] = None,
        __task__: Optional[str] = None,
    ) -> Union[str, Generator, Iterator]:
        
        if __user__ and "valves" in __user__:
            image_mode = __user__["valves"].image_mode
            fallback_model = __user__["valves"].fallback_model
            negative_prompt = __user__["valves"].negative_prompt
            size = __user__["valves"].size
            steps = __user__["valves"].steps
            guidance = __user__["valves"].guidance
        else:
            image_mode = self.user_valves.image_mode
            fallback_model = self.user_valves.fallback_model
            negative_prompt = self.user_valves.negative_prompt
            size = self.user_valves.size
            steps = self.user_valves.steps
            guidance = self.user_valves.guidance

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

        # Check which model they selected (remove the "image-studio-" prefix)
        model_id = body.get("model", "")
        if model_id.startswith("image-studio-"):
            model_id = model_id[len("image-studio-"):]

        if not image_mode or __task__ or model_id == "error":
            if fallback_model == "gpt-4o":
                try:
                    models_res = requests.get(f"{self.valves.api_base_url}/models", timeout=5)
                    if models_res.status_code == 200:
                        for m in models_res.json().get("data", []):
                            if m.get("type") == "llm":
                                fallback_model = m["id"]
                                break
                except Exception:
                    pass

            if __event_emitter__ and not __task__:
                await __event_emitter__({"type": "status", "data": {"description": f"Routing to text model {fallback_model}...", "done": False}})
            
            body["model"] = fallback_model
            try:
                response = requests.post(f"{self.valves.api_base_url}/chat/completions", json=body, stream=True, timeout=120)
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            decoded = line.decode('utf-8')
                            if decoded.startswith('data: '):
                                try:
                                    data = json.loads(decoded[6:])
                                    if "choices" in data and len(data["choices"]) > 0:
                                        delta = data["choices"][0].get("delta", {})
                                        if "content" in delta:
                                            yield delta["content"]
                                except json.JSONDecodeError:
                                    pass
                    return
                else:
                    yield f"❌ Text generation failed: {response.text}"
            except Exception as e:
                yield f"❌ Failed to reach text API: {e}"
            return

        if __event_emitter__:
            await __event_emitter__({"type": "status", "data": {"description": f"🎨 Generating {model_id}...", "done": False}})

        payload = {
            "prompt": user_prompt,
            "negative_prompt": negative_prompt,
            "model": model_id,
            "size": size,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "response_format": "url"
        }

        try:
            response = requests.post(f"{self.valves.api_base_url}/images/generations", json=payload, timeout=120)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    img_data = data["data"][0]
                    if "url" in img_data:
                        base_url = self.valves.api_base_url.replace("/v1", "")
                        full_url = f"{base_url}{img_data['url']}"
                        md_image = f"![Generated Image]({full_url})"
                    else:
                        yield "❌ Image generation failed: Unrecognized response format."
                        return

                    if __event_emitter__:
                        await __event_emitter__({"type": "status", "data": {"description": "✨ Image generation complete!", "done": True}})
                    yield f"{md_image}\n\n*Settings: {size} | {steps} steps | {guidance} guidance*"
            else:
                yield f"❌ Image generation error: {response.text}"
        except Exception as e:
            yield f"❌ Failed to reach image generator API: {e}"
