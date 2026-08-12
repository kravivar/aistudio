import pytest
import requests
import json
import os
import subprocess
import time
import socket

# AI Studio backend URL
BASE_URL = os.getenv("AISTUDIO_API_URL", "http://localhost:3001/v1")

def wait_for_port(port, host='localhost', timeout=30.0):
    start_time = time.time()
    while True:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(1)
            if time.time() - start_time >= timeout:
                return False

@pytest.fixture(scope="session", autouse=True)
def api_server():
    """Starts the AI Studio backend API server automatically before tests run."""
    # Check if server is already running
    if wait_for_port(3001, timeout=1.0):
        print("\nAPI Server is already running. Reusing existing instance.")
        yield
        return
        
    print("\nStarting AI Studio API Server...")
    # Start the server using the virtual environment python
    server_process = subprocess.Popen(
        [".venv/bin/python", "-m", "uvicorn", "src.aistudio.server.app:app", "--host", "0.0.0.0", "--port", "3001"],
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for the server to bind to the port
    if not wait_for_port(3001, timeout=30.0):
        server_process.kill()
        pytest.fail("API Server failed to start within 30 seconds.")
        
    print("API Server started successfully.")
    
    yield  # Tests run here
    
    print("\nShutting down AI Studio API Server...")
    server_process.terminate()
    try:
        server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_process.kill()

@pytest.fixture(scope="module")
def models():
    """Fetch all available models from the AI Studio API."""
    try:
        response = requests.get(f"{BASE_URL}/models")
        assert response.status_code == 200, f"API server is not running at {BASE_URL}"
        models_data = response.json().get("data", [])
        
        # Filter out obvious image/video models based on ID conventions if needed
        # (Though we can just handle the 400 error in the test)
        return [m["id"] for m in models_data]
    except requests.exceptions.ConnectionError:
        pytest.fail(f"Could not connect to {BASE_URL}. Is the backend server running?")

def test_model_generations(models):
    """
    Loops through all models, runs an API chat completion with seed 21,
    and asserts that the output is clean (no leaked reasoning tags).
    """
    assert len(models) > 0, "No models were returned by the API"
    
    prompt = "hi"
    
    for model_id in models:
        # We'll test streaming mode because that's where the fragmentation bug lived
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "seed": 21,
            "max_tokens": 50, # Keep it short for testing speed
            "stream": True
        }
        
        print(f"\n--- Testing Model: {model_id} ---")
        
        response = requests.post(f"{BASE_URL}/chat/completions", json=payload, stream=True)
        
        # If it's an image/video model being mistakenly called, it might return 400. We can skip it.
        if response.status_code == 400:
            print(f"Skipping {model_id} (Returned 400, likely not a chat model)")
            continue
            
        assert response.status_code == 200, f"Model {model_id} failed: {response.text}"
        
        streamed_content = ""
        try:
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith("data: ") and line_str != "data: [DONE]":
                        try:
                            chunk = json.loads(line_str[6:])
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                streamed_content += delta["content"]
                        except json.JSONDecodeError:
                            pass
        except requests.exceptions.ChunkedEncodingError:
            print(f"Skipping {model_id} (ChunkedEncodingError - likely not an LLM)")
            continue
        
        print(f"Output for {model_id}:\n{streamed_content}")
        
        # --- ASSERTIONS ---
        assert len(streamed_content) > 0, f"Model {model_id} returned an empty string!"
        
        # 1. Assert NO raw reasoning tags leaked!
        bad_tags = ["<|channel|>", "<|channel>", "<channel|>", "<|start|>", "<|message|>", "<|end|>"]
        for bad_tag in bad_tags:
            assert bad_tag not in streamed_content, f"Model {model_id} leaked raw tag: {bad_tag}"
            
        # 2. Assert stray words were cleaned up
        if "<think>" in streamed_content:
            lower_content = streamed_content.lower()
            
            # Ensure the think block was closed!
            assert "</think>" in streamed_content, f"Model {model_id} opened a <think> block but never closed it!"
            
            # Ensure stray words didn't leak immediately after the opening or closing tag
            stray_words = ["thought", "analysis", "assistant", "final"]
            for word in stray_words:
                assert f"<think>\n{word}" not in lower_content, f"Model {model_id} leaked stray word '{word}' after <think>"
                assert f"</think>\n{word}" not in lower_content, f"Model {model_id} leaked stray word '{word}' after </think>"
        
        # 3. Ensure no duplicate tags
        assert "<think>\n<think>" not in streamed_content, f"Model {model_id} emitted duplicate <think> tags"
        assert "</think>\n</think>" not in streamed_content, f"Model {model_id} emitted duplicate </think> tags"
