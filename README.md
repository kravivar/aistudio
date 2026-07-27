# `ai_studio`

> **Apple Silicon Native Unified AI Model & Research Server for Open WebUI**

`ai_studio` is a high-performance Python package and OpenAI-compliant REST API server designed to host and execute state-of-the-art AI models natively on Apple Silicon using Apple's **MLX** framework and **PyTorch MPS**, integrated with Open Notebook synthesis features and Open WebUI extensions.

---

## ⚡ Features

1. **Text & Vision LLMs** (`mlx-lm`, `mlx-vlm`): Fast native text generation & vision analysis.
2. **Local Image Generation** (`/v1/images/generations`): PyTorch SDXL / MPS (`.safetensors` checkpoints).
3. **Local Multi-Scene Video Generation** (`/v1/video/generations`): Multi-scene timeline generation, frame continuity, and FFMPEG video stitching.
4. **Fast Speech-to-Text** (`/v1/audio/transcriptions`): High-speed audio transcription powered by `mlx-whisper`.
5. **Open Notebook Research Engine** (`/v1/notebook/synthesis` & `/v1/notebook/notes`): Document summarization, concept expansion, study questions, flashcards, and multi-speaker podcast generation.
6. **Open WebUI Integrations**:
   - `video_pipe.py`: Custom Pipe function for generating LTX multi-scene videos in Open WebUI chat.
   - `research_tool.py`: Custom Tool for agentic research notes, document synthesis, and study flashcards.

---

## 🚀 Environment Setup & Installation Guide (`uv` Managed)

### Prerequisites
- macOS (Apple Silicon M1/M2/M3/M4 recommended)
- [`uv`](https://github.com/astral-sh/uv) package manager installed (`brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Step 1: One-Command Sync & Installation

Run `uv sync --extra dev` to automatically create the Python 3.11 environment (`.venv`) and install all dependencies (including `open-webui`, `ai_studio`, and development tools):

```bash
uv sync --extra dev
```

### Step 2: Activate Environment

```bash
source .venv/bin/activate
```

### Step 3: Environment Configuration

Copy the example environment file to `.env`:

```bash
cp .env.example .env
```

---

## 🏃 Running `ai_studio`

### Option A: Launch API Server Only (Default)

Starts the `ai_studio` REST API server on `http://0.0.0.0:8000`:

```bash
python main.py
```

Access Interactive Swagger Documentation at: `http://localhost:8000/docs`

### Option B: Launch API Server + Open WebUI

Starts the `ai_studio` API server and launches native Open WebUI at `http://localhost:3000`:

```bash
python main.py --webui
```

---

## 🧩 Adding Open WebUI Custom Tools & Pipes

To import custom tools into Open WebUI (**Workspace → Functions / Tools**):

1. **LTX Video Generator Pipe** ([`src/ai_studio/webui_tools/video_pipe.py`](file:///Users/kripakaranravivarman/git/ai_studio/src/ai_studio/webui_tools/video_pipe.py)):
   - In Open WebUI, go to **Workspace → Functions → Add Function**.
   - Copy content from `video_pipe.py` to enable multi-scene video generation directly in chat.

2. **Open Notebook Research Tool** ([`src/ai_studio/webui_tools/research_tool.py`](file:///Users/kripakaranravivarman/git/ai_studio/src/ai_studio/webui_tools/research_tool.py)):
   - In Open WebUI, go to **Workspace → Tools → Add Tool**.
   - Copy content from `research_tool.py` to enable agentic note-taking, search, document synthesis, and podcast script creation.

---

## 📡 API Endpoint Summary

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/v1/models` | `GET` | List available models dynamically scanned from search paths |
| `/v1/chat/completions` | `POST` | Chat completion with SSE streaming support |
| `/v1/completions` | `POST` | Prompt completion endpoint |
| `/v1/images/generations` | `POST` | Text-to-Image SDXL generation |
| `/v1/video/generations` | `POST` | Multi-scene video timeline generation & FFMPEG stitching |
| `/v1/audio/transcriptions` | `POST` | Audio transcription via `mlx-whisper` |
| `/v1/notebook/synthesis` | `POST` | Document summarization, expansion, flashcards, & podcasts |
| `/v1/notebook/notes` | `GET` / `POST` | Retrieve, search, and create research notes |

---

## 🛠 Project Structure

```
ai_studio/
├── .env.example
├── pyproject.toml
├── README.md
├── ANTIGRAVITY.md
├── main.py
└── src/
    └── ai_studio/
        ├── config.py
        ├── notebook/
        │   ├── manager.py
        │   └── synthesis.py
        ├── webui_tools/
        │   ├── video_pipe.py
        │   └── research_tool.py
        ├── server/
        │   ├── app.py
        │   ├── schemas.py
        │   └── manager.py
        ├── pipelines/
        │   ├── llm.py
        │   ├── image.py
        │   ├── video.py
        │   └── audio.py
        └── utils/
            ├── media.py
            └── logging.py
```
