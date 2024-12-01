
#
# tbxteval/evaluator/anhtropic_evaluator.py
# Contains the evaluator class to run inference on claude model types.
#

from tbexteval.evaluator.base import EvaluatorInterface, EvaluatorOutput

class AnthropicEvaluator(EvaluatorInterface):
    def __call__(self, img: str, table: str):
        msg = [
            {
                'role': 'user',
                'content': [
                    {
                    "type": "text",
                    "text": f"Here is the gneerated table:\n{table}"
                    },
                    {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img,
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
                temperature=0,
            )
        except Exception as e:
            print(e)
            return EvaluatorOutput(
                response=None,
                mistakes=None,
                breakdown=None,
                accuracy_score=None,
                completeness_score=None,
                structural_score=None,
                input_tokens=None,
                output_tokens=None,
                cost=None
            )
        
        mistakes = self.extract_mistakes(response.content[0].text)
        breakdown = self.extract_breakdown(response.content[0].text)
        accuracy_score, completeness_score, structural_score = self.extract_scores(response.content[0].text)

        return EvaluatorOutput(
            response=response.content[0].text,
            mistakes=mistakes,
            breakdown=breakdown,
            accuracy_score=accuracy_score,
            completeness_score=completeness_score,
            structural_score=structural_score,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost=self.get_cost(response.usage.input_tokens, response.usage.output_tokens)
        )