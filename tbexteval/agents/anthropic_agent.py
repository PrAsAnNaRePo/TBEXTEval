#
# agents/anthropic.py
# This file contains the anthropic agent class to run inference on claude model types.
#

from __future__ import annotations
from datetime import datetime

from tbexteval.agents.base import AgentConfig, AgentInterface, AgentOutput


class ClaudeAgent(AgentInterface):
    def __init__(self, config: AgentConfig):
        super().__init__(config)
    
    def __call__(self, base64: str, user_prompt: str):
        msg = [
            {
                'role': 'user',
                'content': [
                    {
                    "type": "text",
                    "text": user_prompt
                    },
                    {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64,
                    },
                    }
                ]
            }
        ]

        try:
            response = self.client.messages.create(
                model=self.config.model_name,
                messages=msg,
                max_tokens=8192,
                system=self.config.system_prompt,
                extra_headers={
                    'anthropic-beta': 'max-tokens-3-5-sonnet-2024-07-15'
                },
                temperature=self.config.temperature,
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
            response=response.content[0].text,
            extracted_html=self.extract_html(response.content[0].text),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost=self.calculate_cost(response.usage.input_tokens, response.usage.output_tokens)
        )