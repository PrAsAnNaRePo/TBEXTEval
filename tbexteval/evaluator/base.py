
from __future__ import annotations
from abc import ABC, abstractmethod
import re
from tbexteval.config import list_of_vision_models, model_price_mapping
from openai import OpenAI
from anthropic import Anthropic

class EvaluatorConfig:
    def __init__(
            self,
            system_prompt: str,
            model_name: str,
            mistake_tag: str = '<mistake>',
            breakdown_tag: str = '<breakdown>',
            accuracy_score_tag: str = '<accuracy_score>',
            completeness_score_tag: str = '<completeness_score>',
            structural_score_tag: str = '<structural_score>',
    ):
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.mistake_tag = mistake_tag
        self.breakdown_tag = breakdown_tag
        self.accuracy_score_tag = accuracy_score_tag
        self.completeness_score_tag = completeness_score_tag
        self.structural_score_tag = structural_score_tag

class EvaluatorOutput:
    def __init__(
            self,
            response: str,
            mistakes: str,
            breakdown: str,
            accuracy_score: str,
            completeness_score: str,
            structural_score: str,
            input_tokens: int,
            output_tokens: int,
            cost: float
    ):
        self.response = response
        self.mistakes = mistakes
        self.breakdown = breakdown
        self.accuracy_score = accuracy_score
        self.completeness_score = completeness_score
        self.structural_score = structural_score
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost = cost

    @property
    def mistakes_and_breakdown(self):
        return f"{self.mistakes}\n{self.breakdown}".strip()
    
    def scores(self):
        return (self.accuracy_score, self.completeness_score, self.structural_score)
    
    def average_scores(self):
        return sum(self.scores()) / len(self.scores())
    
    def weighted_average_scores(self):
        return (self.accuracy_score * 0.4) + (self.completeness_score * 0.4) + (self.structural_score * 0.2)

    def pretty_print(self):
        print(f"Response: {self.response}")
        print(f"Mistakes: {self.mistakes}")
        print(f"Breakdown: {self.breakdown}")
        print(f"Accuracy Score: {self.accuracy_score}")
        print(f"Completeness Score: {self.completeness_score}")
        print(f"Structural Score: {self.structural_score}")
        print(f"Input Tokens: {self.input_tokens}")
        print(f"Output Tokens: {self.output_tokens}")
        print(f"Cost: {self.cost}")

class EvaluatorInterface(ABC):
    def __init__(self, config: EvaluatorConfig):
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

    @abstractmethod
    def __call__(self, base64: str, user_prompt: str, file_name: str):
        pass

    def extract_mistakes(self, content):
        mistake_pattern = rf'{re.escape(self.config.mistake_tag)}(.*?){re.escape(self.get_closing_tag(self.config.mistake_tag))}'
        return re.findall(mistake_pattern, content, re.DOTALL)

    def extract_breakdown(self, content):
        breakdown_pattern = rf'{re.escape(self.config.breakdown_tag)}(.*?){re.escape(self.get_closing_tag(self.config.breakdown_tag))}'
        return re.findall(breakdown_pattern, content, re.DOTALL)
    
    def extract_scores(self, content):
        accuracy_score_pattern = rf'{re.escape(self.config.accuracy_score_tag)}(.*?){re.escape(self.get_closing_tag(self.config.accuracy_score_tag))}'
        completeness_score_pattern = rf'{re.escape(self.config.completeness_score_tag)}(.*?){re.escape(self.get_closing_tag(self.config.completeness_score_tag))}'
        structural_score_pattern = rf'{re.escape(self.config.structural_score_tag)}(.*?){re.escape(self.get_closing_tag(self.config.structural_score_tag))}'
        accuracy_score = re.findall(accuracy_score_pattern, content, re.DOTALL)
        completeness_score = re.findall(completeness_score_pattern, content, re.DOTALL)
        structural_score = re.findall(structural_score_pattern, content, re.DOTALL)
        return accuracy_score, completeness_score, structural_score
    
    @property
    def name(self) -> str:
        if 'claude' in self.config.model_name:
            return f'anthropic-{self.config.model_name}'
        elif 'gpt' in self.config.model_name:
            return f'openai-{self.config.model_name}'
        else:
            raise ValueError(f'Unknown model name: {self.config.model_name}, supported only "openai" and "anthropic" models.')

    def get_closing_tag(self, token: str):
        return "</" + token[1:]
    
    def get_cost(self, input_tokens: int, output_tokens: int) -> float:
        INPUT_RATE = model_price_mapping[self.config.model_name]['input']
        OUTPUT_RATE = model_price_mapping[self.config.model_name]['output']
        
        input_cost = (input_tokens / 1_000_000) * INPUT_RATE
        output_cost = (output_tokens / 1_000_000) * OUTPUT_RATE
        
        total_cost = input_cost + output_cost
        
        return round(total_cost, 2)