#
# agents/anthropic.py
# This file contains the anthropic agent class to run inference on claude model types.
#

from __future__ import annotations
from datetime import datetime

from tbexteval.agents.base import AgentConfig, AgentInterface, AgentOutput


class OpenAIAgent(AgentInterface):
    def __init__(self, config: AgentConfig):
        super().__init__(config)
    
    def __call__(self, base64: str, user_prompt: str):
        msg = [
            {
                'role': 'system',
                'content': self.config.system_prompt
            },
            {
                'role': 'user',
                'content': [
                    {
                    "type": "text",
                    "text": user_prompt
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url":  f"data:image/jpeg;base64,{base64}"
                    },
                    },
                ]
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=msg,
                temperature=self.config.temperature
            )
        except Exception as e:
            print(e)
            return AgentOutput(
                time=None,
                response=None,
                extracted_html=None,
                input_tokens=None,
                output_tokens=None,
                cost=None
            )
        
        return AgentOutput(
            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            response=response.choices[0].message.content,
            extracted_html=self.extract_html(response.choices[0].message.content),
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            cost=self.get_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
        )