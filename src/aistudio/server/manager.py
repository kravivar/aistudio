import threading
import gc
from aistudio.utils.logging import logger
from aistudio.pipelines.llm import llm_pipeline
from aistudio.pipelines.image import image_pipeline
from aistudio.pipelines.video import video_pipeline
from aistudio.pipelines.audio import audio_pipeline

class ModelManager:
    """
    Thread-safe manager for controlling model loading, switching, and memory cleanup
    on Apple Silicon Unified Memory Architecture.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.active_pipeline_type: str = "none"

    def prepare_pipeline(self, pipeline_type: str):
        with self._lock:
            if self.active_pipeline_type != pipeline_type and self.active_pipeline_type != "none":
                logger.info(f"Switching active pipeline from {self.active_pipeline_type} to {pipeline_type}. Unloading memory...")
                if self.active_pipeline_type == "llm":
                    llm_pipeline.unload()
                elif self.active_pipeline_type == "image":
                    image_pipeline.unload()
                gc.collect()
            self.active_pipeline_type = pipeline_type

model_manager = ModelManager()
