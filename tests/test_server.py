from fastapi.testclient import TestClient
from aistudio.server.app import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "aistudio"

def test_list_models():
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data

# def test_video_generations(tmp_path):
#     test_out = tmp_path / "test_video.mp4"
#     payload = {
#         "prompt": "Test synthetic video scene",
#         "output_path": str(test_out)
#     }
#     response = client.post("/v1/video/generations", json=payload)
#     assert response.status_code == 200
#     data = response.json()
#     assert data["status"] == "success"
#     assert "final_video_url" in data

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

def test_system_status():
    response = client.get("/v1/system/status")
    assert response.status_code == 200
    data = response.json()
    assert "memory" in data
    assert "budget_gb" in data["memory"]
    assert "cpu" in data
    assert "queues" in data
    assert "throughput" in data
    assert data["queues"]["image"]["available_slots"] == 1

# def test_chat_completions_video_autoroute():
#     payload = {
#         "model": "dgrauet/ltx-2.3-mlx-q8",
#         "messages": [{"role": "user", "content": "A serene mountain lake at sunrise"}],
#         "stream": False
#     }
#     response = client.post("/v1/chat/completions", json=payload)
#     assert response.status_code == 200
#     data = response.json()
#     content = data["choices"][0]["message"]["content"]
#     assert "<video" in content
#     assert ".mp4" in content
#     assert "Generated Video" in content


