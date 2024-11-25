# 
# agents/base.py
# This file contains the base schema class for all agents to follow the same interface
#

from __future__ import annotations


from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
from anthropic import Anthropic

list_of_vision_models = [
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-4-turbo',
    'claude-3-opus',
    'claude-3-sonnet',
    'claude-3-haiku',
    'claude-3-5-sonnet'
]

class AgentConfig:
    def __init__(self, model_name: str, system_prompt: str, temperature: float=0):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature

class AgentInterface(ABC):
    def __init__(self, config: AgentConfig):
        self.config = config
        
        model = self.name
        if not any(model in self.config.model_name for model in list_of_vision_models):
            raise ValueError(f"model name: {self.config.model_name} doesn't support vision, supported only {', '.join(list_of_vision_models)}")
        
        if 'claude' in model:
            print("Using Anthropic Client")
            self.client = Anthropic()
        else:
            print("Using OpenAI Client")
            self.client = OpenAI()

    @property
    def name(self) -> str:
        if 'claude' in self.config.model_name:
            return f'anthropic-{self.config.model_name}'
        elif 'gpt' in self.config.model_name:
            return f'openai-{self.config.model_name}'
        else:
            raise ValueError(f'Unknown model name: {self.config.model_name}, supported only "openai" and "anthropic" models.')

    @abstractmethod
    def extract_html(self, content: str) -> List[str]:
        pass

    @abstractmethod
    def __call__(self, base64: str, user_prompt: str, file_name: str):
        pass