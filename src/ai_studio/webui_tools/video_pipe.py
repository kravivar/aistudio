import requests
from typing import List, Union, Generator, Iterator

class Pipe:
    """
    Open WebUI Custom Pipe Function
    Exposes ai_studio's LTX Video Generation endpoint directly inside Open WebUI.
    """
    def __init__(self):
        self.type = "pipe"
        self.id = "ltx_video_generator"
        self.name = "LTX Multi-Scene Video Generator"

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> str:
        api_url = "http://localhost:8000/v1/video/generations"
        payload = {
            "prompt": user_message,
            "output_path": "./output/video/generated_video.mp4"
        }
        try:
            response = requests.post(api_url, json=payload, timeout=120)
            if response.status_code == 200:
                data = response.json()
                video_url = data.get("final_video_url", "")
                return f"🎬 Video generation completed!\n\nWatch video: http://localhost:8000{video_url}"
            else:
                return f"❌ Video generation error: {response.text}"
        except Exception as e:
            return f"❌ Failed to reach video generator API: {e}"
