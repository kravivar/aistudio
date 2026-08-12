"""
Thinking Filter Base Class and Registry.

Each model family can have its own thinking filter that defines how to detect
opening/closing reasoning tags and what stray words to clean up.

To add a new filter:
1. Create a new file in this directory (e.g., deepseek_llm_thinking_filter.py)
2. Subclass ThinkingFilter
3. Register it with @register_filter("model-name-pattern")
"""

import re
from typing import Optional, List, Tuple
from dataclasses import dataclass, field


@dataclass
class FilterResult:
    """Result of processing a buffer through a thinking filter."""
    buffer: str
    has_opened_think: bool


class ThinkingFilter:
    """Base class for model-specific thinking filters."""
    
    # Subclasses set these
    name: str = "base"
    # Glob-style patterns to match model IDs (e.g., "gemma", "gpt-oss")
    model_patterns: List[str] = []
    # Stray words that appear after opening/closing tags and should be absorbed
    stray_words: List[str] = ["thought", "analysis", "assistant", "final"]
    
    def detect_and_replace_tags(self, buffer: str, has_opened_think: bool) -> FilterResult:
        """
        Process the buffer: detect reasoning tags and replace them with
        <think>/<​/think>. Returns the modified buffer and updated state.
        
        Subclasses should override this method.
        """
        raise NotImplementedError
    
    def clean_stray_words(self, buffer: str) -> str:
        """Remove stray words that leak after <think>\\n or </think>\\n tags."""
        if not self.stray_words:
            return buffer
        words = "|".join(re.escape(w) for w in self.stray_words)
        buffer = re.sub(
            rf"<think>\n\s*(?:{words})\b\s*",
            "<think>\n", buffer, flags=re.IGNORECASE
        )
        buffer = re.sub(
            rf"</think>\n\s*(?:{words})\b\s*",
            "</think>\n", buffer, flags=re.IGNORECASE
        )
        return buffer


# ── Filter Registry ──────────────────────────────────────────────────────────

_registry: List[Tuple[ThinkingFilter, List[str]]] = []


def register_filter(filter_instance: ThinkingFilter):
    """Register a thinking filter instance."""
    _registry.append((filter_instance, filter_instance.model_patterns))


def get_filter_for_model(model_id: str) -> ThinkingFilter:
    """
    Find the best matching filter for a given model ID.
    Falls back to DefaultThinkingFilter if no specific match is found.
    """
    model_lower = model_id.lower()
    for filt, patterns in _registry:
        for pattern in patterns:
            if pattern.lower() in model_lower:
                return filt
    # Fallback
    return _default_filter


# ── Import all filters to trigger registration ──────────────────────────────
# The default filter is defined here; model-specific ones are imported below.

class DefaultThinkingFilter(ThinkingFilter):
    """
    Default filter that handles the most common patterns:
    - <think>/<​/think> (native, e.g., Qwen, DeepSeek)
    - Generic tags containing 'think', 'thought', 'channel', 'analysis'
    - Treats a duplicate opening tag as a closing signal (toggle behavior)
    """
    name = "default"
    model_patterns = []  # Fallback — matches everything
    
    PH_OPEN = "\x00TO\x00"
    PH_CLOSE = "\x00TC\x00"
    
    def detect_and_replace_tags(self, buffer: str, has_opened_think: bool) -> FilterResult:
        # Protect our own tags
        buffer = buffer.replace("</think>", self.PH_CLOSE)
        buffer = buffer.replace("<think>", self.PH_OPEN)
        
        # Strip model control tokens
        buffer = buffer.replace("<|start|>", "")
        buffer = buffer.replace("<|message|>", "")
        
        # ONE pattern to catch any <...> containing a reasoning keyword
        tag_pattern = r"<[^>]*(?:channel|think|thought|analysis)[^>]*>"
        
        def tag_replacer(match):
            tag = match.group()
            # We don't need to track state here anymore, just map to placeholders
            if tag.startswith("</") or re.match(r"^<[a-z]+\|>$", tag, re.IGNORECASE):
                return self.PH_CLOSE
            else:
                return self.PH_OPEN
        
        buffer = re.sub(tag_pattern, tag_replacer, buffer, flags=re.IGNORECASE)
        
        # Handle <|end|> specifically (not via regex to avoid catching <end_of_turn>)
        if "<|end|>" in buffer:
            buffer = buffer.replace("<|end|>", self.PH_CLOSE)
        
        # Update state robustly based on what's ACTUALLY in the buffer!
        opens = buffer.count(self.PH_OPEN)
        closes = buffer.count(self.PH_CLOSE)
        
        # We start from whatever the state was previously, unless the buffer overrides it
        if opens > closes:
            has_opened_think = True
        elif closes > opens:
            has_opened_think = False
        elif opens > 0 and closes > 0 and opens == closes:
            last_open = buffer.rfind(self.PH_OPEN)
            last_close = buffer.rfind(self.PH_CLOSE)
            has_opened_think = last_open > last_close
        
        # Restore placeholders
        buffer = buffer.replace(self.PH_OPEN, "<think>\n")
        buffer = buffer.replace(self.PH_CLOSE, "</think>\n")
        
        # Deduplicate
        buffer = buffer.replace("</think>\n</think>\n", "</think>\n")
        buffer = buffer.replace("<think>\n<think>\n", "<think>\n")
        
        return FilterResult(buffer=buffer, has_opened_think=has_opened_think)


_default_filter = DefaultThinkingFilter()

# Import model-specific filters to trigger their registration
from . import gemma_llm_thinking_filter  # noqa: E402, F401
from . import gpt_oss_llm_thinking_filter  # noqa: E402, F401
