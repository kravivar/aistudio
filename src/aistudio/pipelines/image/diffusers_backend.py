import gc
from typing import Optional, Any
from aistudio.utils.logging import logger
from .base import BaseImagePipeline
from aistudio.config import resolve_model_path

class DiffusersPipeline(BaseImagePipeline):
    SDXL_BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(self):
        self.pipe = None
        self.current_model_id = None

    @staticmethod
    def _is_lora_safetensors(file_path: str) -> bool:
        try:
            from safetensors import safe_open
            with safe_open(file_path, framework="pt") as f:
                for key in list(f.keys())[:10]:
                    if "lora_a" in key.lower() or "lora_b" in key.lower():
                        return True
            return False
        except Exception:
            return False

    def load_pipeline(self, model_id: str, resolved_path: str):
        if self.current_model_id == model_id and self.pipe is not None:
            return

        import torch
        from diffusers import StableDiffusionXLPipeline

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if device == "mps" else torch.float32

        if resolved_path.endswith(".safetensors"):
            if self._is_lora_safetensors(resolved_path):
                logger.info(f"Detected LoRA adapter. Loading base SDXL model ({self.SDXL_BASE_MODEL}) and applying LoRA weights...")
                base_path = resolve_model_path(self.SDXL_BASE_MODEL)
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    base_path,
                    torch_dtype=torch_dtype,
                    use_safetensors=True,
                )
                self.pipe.load_lora_weights(resolved_path)
                self.pipe.fuse_lora()
                logger.info("LoRA weights fused into base pipeline.")
            else:
                self.pipe = StableDiffusionXLPipeline.from_single_file(
                    resolved_path,
                    torch_dtype=torch_dtype,
                    use_safetensors=True,
                )
        else:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                resolved_path,
                torch_dtype=torch_dtype,
            )

        self.pipe.to(device)
        self.current_model_id = model_id
        logger.info(f"Diffusers pipeline loaded successfully on {device}")

    def generate(self, prompt: str, negative_prompt: Optional[str] = None, width: int = 1024, height: int = 1024, num_inference_steps: int = 8, guidance_scale: float = 2.0, seed: Optional[int] = None) -> Any:
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale
        }
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
            
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            kwargs["generator"] = generator
            
        return self.pipe(**kwargs).images[0]

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
