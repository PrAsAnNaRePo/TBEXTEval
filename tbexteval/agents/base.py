# 
# agents/base.py
# This file contains the base schema class for all agents to follow the same interface
#

from __future__ import annotations


from abc import ABC, abstractmethod
import re
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
from anthropic import Anthropic
from tbexteval.config import list_of_vision_models, model_price_mapping

class AgentConfig:
    def __init__(self, model_name: str, system_prompt: str, temperature: float=0):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature

class AgentOutput:
    def __init__(
            self,
            time: str,
            response: str,
            extracted_html: str,
            input_tokens: int,
            output_tokens: int,
            cost: float
    ):
        self.time = time
        self.response = response
        self.extracted_html = extracted_html
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost = cost

    def pretty_print(self):
        print(f"Time: {self.time}")
        # print(f"Response: {self.response}")
        print(f"Extracted HTML: {self.extracted_html}")
        print(f"Input Tokens: {self.input_tokens}")
        print(f"Output Tokens: {self.output_tokens}")
        print(f"Cost: {self.cost}")

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

    # @abstractmethod
    # def extract_html(self, content: str) -> List[str]:
    #     pass
    def extract_html(self, content: str) -> List[str]:
        code_blocks = re.findall(r'<final>\n<table(.*?)</final>', content, re.DOTALL)
        return code_blocks
    
    def get_cost(self, input_tokens: int, output_tokens: int) -> float:
        INPUT_RATE = model_price_mapping[self.config.model_name]['input']
        OUTPUT_RATE = model_price_mapping[self.config.model_name]['output']
        
        input_cost = (input_tokens / 1_000_000) * INPUT_RATE
        output_cost = (output_tokens / 1_000_000) * OUTPUT_RATE
        
        total_cost = input_cost + output_cost
        
        return round(total_cost, 2)

    @abstractmethod
    def __call__(self, base64: str, user_prompt: str, file_name: str):
        pass