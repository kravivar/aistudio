import base64
import io
import time
from typing import Dict, Any, Optional
from pathlib import Path

from aistudio.config import resolve_model_path, get_model_config
from aistudio.utils.logging import logger
from .base import BaseImagePipeline
from .diffusers_backend import DiffusersPipeline
from .mflux_backend import MFluxPipeline

class ImagePipelineManager:
    def __init__(self):
        self.active_backend: Optional[BaseImagePipeline] = None
        self.active_backend_name: Optional[str] = None
        self.current_model_id: Optional[str] = None

    def _get_backend_class(self, backend_name: str):
        if backend_name == "mflux":
            return MFluxPipeline
        # Default to diffusers for anything else
        return DiffusersPipeline

    def load_pipeline(self, model_id: str):
        if self.current_model_id == model_id and self.active_backend is not None:
            return

        cfg = get_model_config(model_id)
        # Use 'backend' (or 'loader_type') from config if provided, default to diffusers
        backend_name = cfg.get("backend", cfg.get("loader_type", "diffusers")).lower()

        # Unload previous backend if switching implementations
        if self.active_backend is not None and self.active_backend_name != backend_name:
            logger.info(f"Unloading previous backend: {self.active_backend_name}")
            self.active_backend.unload()
            self.active_backend = None

        if self.active_backend is None:
            backend_class = self._get_backend_class(backend_name)
            self.active_backend = backend_class()
            self.active_backend_name = backend_name

        resolved_path = resolve_model_path(model_id)
        logger.info(f"Delegating load to {backend_name} backend for {resolved_path}")
        
        try:
            self.active_backend.load_pipeline(model_id, resolved_path)
            self.current_model_id = model_id
        except Exception as e:
            logger.error(f"Failed to load image pipeline {model_id} via {backend_name}: {e}")
            self.active_backend = None
            self.current_model_id = None
            raise RuntimeError(f"Error loading image pipeline {model_id} via {backend_name}: {e}")

    def unload(self):
        if self.active_backend:
            self.active_backend.unload()
        self.active_backend = None
        self.active_backend_name = None
        self.current_model_id = None

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = "blurry",
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        n: int = 1,
        size: str = "1024x1024",
        num_inference_steps: int = 8,
        guidance_scale: float = 2.0,
        response_format: str = "b64_json"
    ) -> Dict[str, Any]:
        self.load_pipeline(model_id)

        try:
            width, height = map(int, size.lower().split("x"))
        except Exception:
            width, height = 1024, 1024

        images_data = []
        for i in range(n):
            image = self.active_backend.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()

            if response_format == "b64_json":
                b64_str = base64.b64encode(img_bytes).decode("utf-8")
                images_data.append({"b64_json": b64_str})
            else:
                out_dir = Path("./output/images")
                out_dir.mkdir(parents=True, exist_ok=True)
                file_path = out_dir / f"gen_{int(time.time())}_{i}.png"
                image.save(file_path)
                images_data.append({"url": f"/static/images/{file_path.name}"})

        return {
            "created": int(time.time()),
            "data": images_data
        }

image_pipeline = ImagePipelineManager()
