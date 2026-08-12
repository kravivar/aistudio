"""
Gemma 4 Thinking Filter.

Gemma 4 uses:
  - Opening: <|channel>  (pipe BEFORE keyword, no closing pipe)
  - Closing: <channel|>  (pipe AFTER keyword, no opening pipe)
  - Stray words: 'thought' after open tag
"""

import re
from . import ThinkingFilter, FilterResult, register_filter


class GemmaThinkingFilter(ThinkingFilter):
    name = "gemma"
    model_patterns = ["gemma"]
    stray_words = ["thought", "analysis", "assistant", "final"]
    
    PH_OPEN = "\x00TO\x00"
    PH_CLOSE = "\x00TC\x00"
    
    def detect_and_replace_tags(self, buffer: str, has_opened_think: bool) -> FilterResult:
        # Protect our own tags (close MUST be replaced first)
        buffer = buffer.replace("</think>", self.PH_CLOSE)
        buffer = buffer.replace("<think>", self.PH_OPEN)
        
        # Strip model control tokens
        buffer = buffer.replace("<|start|>", "")
        buffer = buffer.replace("<|message|>", "")
        
        state = {"has_opened_think": has_opened_think}
        
        # Gemma pattern: <|channel> = open, <channel|> = close
        # The pipe position signals direction
        tag_pattern = r"<[^>]*(?:channel|think|thought|analysis)[^>]*>"
        
        def tag_replacer(match):
            tag = match.group()
            # Closing: starts with </ OR reversed pipe pattern <word|>
            if tag.startswith("</") or re.match(r"^<[a-z]+\|>$", tag, re.IGNORECASE):
                state["has_opened_think"] = False
                return self.PH_CLOSE
            else:
                if not state["has_opened_think"]:
                    state["has_opened_think"] = True
                    return self.PH_OPEN
                else:
                    # Duplicate open = close (toggle)
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


register_filter(GemmaThinkingFilter())
