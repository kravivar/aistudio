from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseImagePipeline(ABC):
    @abstractmethod
    def load_pipeline(self, model_id: str, resolved_path: str):
        pass

    @abstractmethod
    def generate(self, prompt: str, negative_prompt: Optional[str] = None, width: int = 1024, height: int = 1024, num_inference_steps: int = 8, guidance_scale: float = 2.0, seed: Optional[int] = None) -> Any:
        pass

    @abstractmethod
    def unload(self):
        pass
