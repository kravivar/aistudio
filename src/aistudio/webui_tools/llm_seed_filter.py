"""
title: Advanced Seed Filter
author: AI Studio
version: 1.0
description: Adds a Seed parameter to standard LLM chat completions in Open WebUI.
"""

from pydantic import BaseModel, Field
from typing import Optional

class Filter:
    class Valves(BaseModel):
        pass

    class UserValves(BaseModel):
        seed: int = Field(
            default=-1, 
            description="Seed for reproducible generations (-1 for random). Only supported by MLX local models."
        )

    def __init__(self):
        self.type = "filter"
        self.id = "llm_seed_filter"
        self.name = "Advanced LLM Seed"
        self.valves = self.Valves()
        self.user_valves = self.UserValves()

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """
        Intercepts the outgoing request to the LLM backend and injects the seed.
        """
        seed = -1
        if __user__ and "valves" in __user__ and hasattr(__user__["valves"], "seed"):
            seed = __user__["valves"].seed
        elif hasattr(self.user_valves, "seed"):
            seed = self.user_valves.seed

        if seed != -1:
            body["seed"] = seed
            
        return body

    def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        return body
