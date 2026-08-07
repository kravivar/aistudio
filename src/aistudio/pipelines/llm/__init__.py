import time
import json
import gc
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, List, Optional
from aistudio.config import resolve_model_path
from aistudio.utils.logging import logger
import re

def clean_response(text: str, strip_whitespace: bool = False) -> str:
    """
    Strips special chat control & end-of-turn tokens.
    Does NOT strip <think>...</think> blocks — Open WebUI renders them as a native
    collapsible grayed-out 'Thinking...' section.
    Does NOT strip whitespace on streaming token chunks unless requested.
    """
    stop_tokens = [
        "<end_of_turn>",
        "<|im_end|>",
        "<|endoftext|>",
        "<|eot_id|>",
        "<eos>",
        "</s>"
    ]
    for stop_token in stop_tokens:
        if stop_token in text:
            text = text.split(stop_token)[0]

    return text.strip() if strip_whitespace else text



def prepare_messages_for_template(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Sanitizes system messages for models whose Jinja chat templates do not support role=='system' (e.g., Gemma).
    Merges system/search context directly into the first user message.
    """
    system_prompts = []
    cleaned_messages = []
    
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            if content:
                system_prompts.append(content)
        else:
            cleaned_messages.append({"role": role, "content": content})
            
    if system_prompts:
        combined_system = "\n".join(system_prompts)
        if cleaned_messages:
            cleaned_messages[0]["content"] = f"{combined_system}\n\n{cleaned_messages[0]['content']}"
        else:
            cleaned_messages = [{"role": "user", "content": combined_system}]
            
    return cleaned_messages or [{"role": "user", "content": ""}]

class LLMPipeline:
    def __init__(self):
        self.current_model_id: Optional[str] = None
        self.model = None
        self.tokenizer = None
        # Set True when apply_chat_template(enable_thinking=True) appended <think> to the
        # prompt — the model will generate thinking content ending with </think>, but the
        # opening <think> is in the prompt and won't appear in the generated output.
        self._prompt_has_think_prefix: bool = False

    def load_model(self, model_id: str):
        if self.current_model_id == model_id and self.model is not None:
            return
        
        real_path = resolve_model_path(model_id)
        resolved = Path(real_path)

        # Handle bare .safetensors / .bin / .gguf file paths:
        # mlx_lm.load() expects a *directory* containing config.json + weight files.
        if resolved.is_file() and resolved.suffix in (".safetensors", ".bin", ".gguf"):
            parent_dir = resolved.parent
            if (parent_dir / "config.json").exists():
                # The file lives inside a proper model directory — use that directory.
                logger.info(
                    f"Resolved single weight file to model directory: {parent_dir}"
                )
                real_path = str(parent_dir)
            else:
                raise RuntimeError(
                    f"Cannot load '{resolved.name}' as an LLM model. "
                    f"The file is a standalone weight file without a config.json in its "
                    f"parent directory ({parent_dir}). "
                    f"If this is a diffusion / image model, use the image generation "
                    f"endpoint (/v1/images/generations) instead."
                )

        logger.info(f"Loading LLM model from path: {real_path}")
        
        import mlx_lm
        self.model, self.tokenizer = mlx_lm.load(real_path)
        self.current_model_id = model_id

    def unload_model(self):
        if self.model is not None:
            logger.info("Unloading current LLM model...")
            self.model = None
            self.tokenizer = None
            self.current_model_id = None
            gc.collect()

    # Alias so ModelManager.prepare_pipeline can call llm_pipeline.unload()
    unload = unload_model

    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Formats messages using the tokenizer's chat template.
        - Tries enable_thinking=True first so reasoning models (Qwen3, DeepSeek-R1, etc.)
          wrap their chain-of-thought between <think>...</think> tags that Open WebUI renders
          as a collapsible grayed-out 'Thinking...' block.
          Note: with enable_thinking=True the tokenizer appends '<think>' to the prompt;
          the generated output therefore starts with the thinking content (no opening tag)
          and ends with '</think>'. We track this via self._prompt_has_think_prefix so the
          stream/generate methods can prepend '<think>' to the first output chunk.
        - Falls back to standard template, then merges system prompts for models that
          reject the 'system' role (e.g. Gemma).
        """
        self._prompt_has_think_prefix = False
        if hasattr(self.tokenizer, "apply_chat_template"):
            # Try with enable_thinking=True — reasoning models emit </think> to close
            # the block; we'll prepend <think> to the output ourselves (see above).
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
                )
                # Confirm the template actually injected <think> at the end
                if prompt.rstrip().endswith("<think>"):
                    self._prompt_has_think_prefix = True
                return prompt
            except TypeError:
                pass  # Model doesn't support enable_thinking; fall through
            except Exception as e:
                logger.warning(f"apply_chat_template with enable_thinking failed ({e}); retrying without.")

            # Standard template (no thinking param)
            try:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as e:
                logger.warning(f"Model chat template rejected system role ({e}). Merging system instructions into user prompt.")
                sanitized = prepare_messages_for_template(messages)
                try:
                    return self.tokenizer.apply_chat_template(sanitized, tokenize=False, add_generation_prompt=True)
                except Exception:
                    return "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in sanitized])
        return "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages])

    def generate(self, model_id: str, messages: List[Dict[str, str]], max_tokens: int = 512, temperature: float = 0.7, thinking_mode: Optional[str] = None) -> Dict[str, Any]:
        self.load_model(model_id)

        import mlx_lm
        prompt = self.format_prompt(messages)

        raw_response = mlx_lm.generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False
        )
        response_text = clean_response(raw_response, strip_whitespace=True)

        # If the prompt injected <think> as prefix, the generated text starts with the
        # thinking content and ends with </think>. Prepend <think> so Open WebUI renders
        # the full block as a collapsible section.
        if self._prompt_has_think_prefix and "</think>" in response_text:
            response_text = "<think>\n" + response_text

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split())
            }
        }

    async def generate_stream(self, model_id: str, messages: List[Dict[str, str]], max_tokens: int = 512, temperature: float = 0.7, thinking_mode: Optional[str] = None) -> AsyncGenerator[bytes, None]:
        """
        Streams chat completions.

        thinking_mode:
          - None / "stream" (default): thinking tokens are wrapped in <think> and streamed live.
            Open WebUI shows an expanding "Thinking..." collapsible section as tokens arrive.
          - "wait": the full model output is buffered silently. Once complete, the thinking block
            is emitted as a finished <think>...</think> section followed by the clean answer.
            Open WebUI shows a pre-collapsed "Thinking" block — no live updates.
        """
        self.load_model(model_id)

        import mlx_lm
        prompt = self.format_prompt(messages)

        completion_id = f"chatcmpl-{int(time.time())}"

        def _make_chunk(content: str) -> bytes:
            data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
            }
            return f"data: {json.dumps(data)}\n\n".encode("utf-8")

        try:
            if hasattr(mlx_lm, "stream_generate"):
                # If the prompt template injected <think> as a suffix, the model output
                # contains the thinking content ending with </think> but NO opening <think>.
                # Emit it now so Open WebUI sees a complete <think>...</think> block.
                think_prefix_emitted = False

                if thinking_mode == "wait":
                    # ── WAIT MODE ───────────────────────────────────────────────────────
                    # Buffer the full response silently, then emit it in one shot.
                    full_text = ""
                    for response in mlx_lm.stream_generate(
                        self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens
                    ):
                        full_text += clean_response(
                            getattr(response, "text", str(response)), strip_whitespace=False
                        )
                    full_text = clean_response(full_text, strip_whitespace=True)
                    if self._prompt_has_think_prefix and "</think>" in full_text:
                        full_text = "<think>\n" + full_text
                    yield _make_chunk(full_text)
                else:
                    # ── STREAM MODE (default) ────────────────────────────────────────────
                    # Stream tokens directly as they arrive.
                    import asyncio
                    for response in mlx_lm.stream_generate(
                        self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens
                    ):
                        token_text = clean_response(
                            getattr(response, "text", str(response)), strip_whitespace=False
                        )
                        if not token_text:
                            continue
                        # Prepend <think> before the first token if the prompt injected it
                        if self._prompt_has_think_prefix and not think_prefix_emitted:
                            think_prefix_emitted = True
                            yield _make_chunk("<think>\n" + token_text)
                        else:
                            yield _make_chunk(token_text)
                        await asyncio.sleep(0)
            else:
                raw_response = mlx_lm.generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    verbose=False
                )
                response_text = clean_response(raw_response)
                chunk_size = 8
                for i in range(0, len(response_text), chunk_size):
                    chunk = response_text[i:i+chunk_size]
                    data = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk},
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(data)}\n\n".encode("utf-8")

            final_data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(final_data)}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Error during stream generation: {e}")
            err_data = {"error": str(e)}
            yield f"data: {json.dumps(err_data)}\n\n".encode("utf-8")

llm_pipeline = LLMPipeline()
