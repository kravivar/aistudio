import time
from typing import Dict, Any, List, Optional
from aistudio.pipelines.llm import llm_pipeline

class ResearchSynthesisEngine:
    """
    Research synthesis engine providing document processing, study tools,
    summarization, expansion, study questions, and multi-speaker podcast generation.
    """

    def summarize(self, text: str, model_id: str = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit") -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": "You are a senior research assistant. Provide a concise, structured executive summary with key takeaways and bullet points."},
            {"role": "user", "content": f"Summarize the following research material:\n\n{text}"}
        ]
        return llm_pipeline.generate(model_id=model_id, messages=messages, max_tokens=1024)

    def expand(self, text: str, model_id: str = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit") -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": "You are an expert educator. Expand upon the provided concept, detailing technical background, real-world applications, and architectural context."},
            {"role": "user", "content": f"Elaborate and expand on this content:\n\n{text}"}
        ]
        return llm_pipeline.generate(model_id=model_id, messages=messages, max_tokens=1536)

    def generate_questions(self, text: str, model_id: str = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit") -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": "Generate 5 interactive study questions with detailed answers based on the text."},
            {"role": "user", "content": f"Create study questions and flashcards for:\n\n{text}"}
        ]
        return llm_pipeline.generate(model_id=model_id, messages=messages, max_tokens=1024)

    def generate_podcast_script(self, text: str, model_id: str = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit") -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": "You are a podcast script writer for an engaging AI technology show with two hosts: Host A (Alex) and Host B (Jamie). Create a lively, conversational multi-speaker dialogue script summarizing the material."},
            {"role": "user", "content": f"Write a multi-speaker podcast script based on this research:\n\n{text}"}
        ]
        return llm_pipeline.generate(model_id=model_id, messages=messages, max_tokens=1536)

synthesis_engine = ResearchSynthesisEngine()
