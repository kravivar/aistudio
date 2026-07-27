import base64
import io
import os
import gc
import time
from typing import Dict, Any, Optional
from pathlib import Path
from aistudio.config import resolve_model_path
from aistudio.utils.logging import logger

class ImagePipeline:
    def __init__(self):
        self.current_model_id: Optional[str] = None
        self.pipe = None

    def load_pipeline(self, model_id: str):
        if self.current_model_id == model_id and self.pipe is not None:
            return

        resolved_path = resolve_model_path(model_id)
        logger.info(f"Loading Image Pipeline from: {resolved_path}")

        import torch
        from diffusers import StableDiffusionXLPipeline, AutoencoderKL

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "mps" else torch.float32

        try:
            if resolved_path.endswith(".safetensors"):
                self.pipe = StableDiffusionXLPipeline.from_single_file(
                    resolved_path,
                    torch_dtype=torch_dtype,
                    use_safetensors=True
                )
            else:
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    resolved_path,
                    torch_dtype=torch_dtype,
                    use_safetensors=True
                )
            self.pipe.to(device)
            self.current_model_id = model_id
            logger.info(f"Image pipeline loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to load image pipeline {model_id}: {e}")
            raise RuntimeError(f"Error loading image pipeline {model_id}: {e}")

    def unload(self):
        self.pipe = None
        self.current_model_id = None
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

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
            kwargs = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale
            }
            if negative_prompt:
                kwargs["negative_prompt"] = negative_prompt

            image = self.pipe(**kwargs).images[0]
            
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

        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

        return {
            "created": int(time.time()),
            "data": images_data
        }

image_pipeline = ImagePipeline()
