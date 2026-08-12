"""
BasePipeline - Standard interface for all model execution pipelines in AI Studio.
Enables polymorphic registration, lifecycle control, and unified memory governance.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List


class BasePipeline(ABC):
    """
    Abstract base class that all model modalities (LLM, Image, Video, Audio) implement.
    """
    pipeline_type: str = "base"

    def __init__(self):
        self.current_model_id: Optional[str] = None

    @abstractmethod
    def unload(self) -> None:
        """Release all model weights, compute graphs, and ML framework caches."""
        pass

    @property
    def is_loaded(self) -> bool:
        """Returns True if a model is currently loaded in memory."""
        return self.current_model_id is not None

    def get_info(self) -> Dict[str, Any]:
        """Returns metadata about the currently active model pipeline."""
        return {
            "pipeline_type": self.pipeline_type,
            "current_model_id": self.current_model_id,
            "is_loaded": self.is_loaded,
        }
