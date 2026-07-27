from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from ai_studio.config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_THINKING_MODE, DEFAULT_MODEL

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ChatCompletionRequest(BaseModel):
    model: str = Field(default=DEFAULT_MODEL)
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    thinking_mode: Optional[str] = None

class CompletionRequest(BaseModel):
    model: str = Field(default=DEFAULT_MODEL)
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: Optional[bool] = False
    thinking_mode: Optional[str] = None


class ImageGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "blurry"
    model: Optional[str] = "stabilityai/stable-diffusion-xl-base-1.0"
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    num_inference_steps: Optional[int] = 8
    guidance_scale: Optional[float] = 2.0
    response_format: Optional[str] = "b64_json"

class VideoScene(BaseModel):
    prompt: str
    duration: Optional[int] = 10
    width: Optional[int] = 704
    height: Optional[int] = 480
    fps: Optional[int] = 24
    steps: Optional[int] = 8
    seed: Optional[int] = 42
    image_path: Optional[str] = None

class VideoGenerationRequest(BaseModel):
    prompt: Optional[str] = None
    scenes: Optional[List[VideoScene]] = None
    model: Optional[str] = "dgrauet/ltx-2.3-mlx-q8"
    n: Optional[int] = 1
    size: Optional[str] = "704x480"
    width: Optional[int] = 704
    height: Optional[int] = 480
    fps: Optional[int] = 24
    duration: Optional[int] = 10
    steps: Optional[int] = 8
    seed: Optional[int] = 42
    image_path: Optional[str] = None
    output_path: Optional[str] = "./output/video/final_movie.mp4"

class SpeechRequest(BaseModel):
    model: Optional[str] = "tts-1"
    input: str
    voice: Optional[str] = "alloy"
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0
