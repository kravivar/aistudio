from fastapi.testclient import TestClient
from ai_studio.server.app import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "ai_studio"

def test_list_models():
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data

def test_video_generations():
    payload = {
        "prompt": "Test synthetic video scene",
        "output_path": "./output/video/test_video.mp4"
    }
    response = client.post("/v1/video/generations", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "final_video_url" in data

def test_audio_speech():
    payload = {
        "model": "tts-1",
        "input": "Hello test speech synthesis",
        "voice": "Samantha"
    }
    response = client.post("/v1/audio/speech", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/")
    assert len(response.content) > 0
