"""
GPT-OSS Thinking Filter.

GPT-OSS uses:
  - Opening: <|channel|>  followed by 'analysis'
  - Closing: <|channel|>  followed by 'final' (same tag reused!)
  - Also emits <|start|>assistant and <|message|> as control tokens
  - Stray words: 'analysis' after open, 'final' and 'assistant' after close
"""

import re
from . import ThinkingFilter, FilterResult, register_filter


class GptOssThinkingFilter(ThinkingFilter):
    name = "gpt_oss"
    model_patterns = ["gpt-oss", "gpt_oss"]
    stray_words = ["thought", "analysis", "assistant", "final"]
    
    PH_OPEN = "\x00TO\x00"
    PH_CLOSE = "\x00TC\x00"
    
    def detect_and_replace_tags(self, buffer: str, has_opened_think: bool) -> FilterResult:
        # Protect our own tags
        buffer = buffer.replace("</think>", self.PH_CLOSE)
        buffer = buffer.replace("<think>", self.PH_OPEN)
        
        # Strip model control tokens
        buffer = buffer.replace("<|start|>", "")
        buffer = buffer.replace("<|message|>", "")
        buffer = buffer.replace("<|end|>", "")
        
        state = {"has_opened_think": has_opened_think}
        
        # GPT-OSS uses <|channel|> for BOTH open and close (toggle)
        tag_pattern = r"<[^>]*(?:channel|think|thought|analysis)[^>]*>"
        
        def tag_replacer(match):
            tag = match.group()
            # Explicit closing tags
            if tag.startswith("</"):
                state["has_opened_think"] = False
                return self.PH_CLOSE
            else:
                if not state["has_opened_think"]:
                    state["has_opened_think"] = True
                    return self.PH_OPEN
                else:
                    # Same tag again while open = CLOSE (toggle)
                    state["has_opened_think"] = False
                    return self.PH_CLOSE
        
        buffer = re.sub(tag_pattern, tag_replacer, buffer, flags=re.IGNORECASE)
        
        # Restore placeholders
        buffer = buffer.replace(self.PH_OPEN, "<think>\n")
        buffer = buffer.replace(self.PH_CLOSE, "</think>\n")
        
        # Deduplicate
        buffer = buffer.replace("</think>\n</think>\n", "</think>\n")
        buffer = buffer.replace("<think>\n<think>\n", "<think>\n")
        
        return FilterResult(buffer=buffer, has_opened_think=state["has_opened_think"])


register_filter(GptOssThinkingFilter())
