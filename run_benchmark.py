from argparse import ArgumentParser
import base64
from datetime import datetime
import json
import os

from tqdm import tqdm
from tbexteval.agents import anthropic_agent, openai_agent, base
from tbexteval.evaluator import base as eval_base
from tbexteval.evaluator import anhtropic_evaluator
from tbexteval.config import LOGS_PATH, DATA_PATH, PROMPT_PATH

class RunBenchmark:
    def __init__(self, args):
        self.args = args
        self.config = base.AgentConfig(
            model_name=args.model_name,
            system_prompt=open(args.gen_system_prompt, 'r').read() if args.gen_system_prompt else open(f'{PROMPT_PATH}gen_prompt.txt', 'r').read(),
            temperature=args.temperature
        )
        self.agent = anthropic_agent.ClaudeAgent(self.config) if 'claude' in args.model_name else openai_agent.OpenAIAgent(self.config)
        self.evaluator_config = eval_base.EvaluatorConfig(
            model_name=args.eval_model_name,
            system_prompt=open(args.eval_system_prompt, 'r').read() if args.eval_system_prompt else open(f'{PROMPT_PATH}eval_prompt.txt', 'r').read()
        )
        self.evaluator = anhtropic_evaluator.AnthropicEvaluator(self.evaluator_config)
        
        self.log_file = LOGS_PATH + args.model_name + '.json' if not args.output_path else args.output_path
        self.data_path = DATA_PATH if not args.data_path else args.data_path

    def append_eval_history(self, eval_hist, log_file):
        try:
            with open(log_file, 'r') as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = []

        existing_data.extend(eval_hist)

        with open(log_file, 'w') as f:
            json.dump(existing_data, f, indent=4)

    def convert_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            base64_string = base64.b64encode(image_file.read()).decode('utf-8')
        return base64_string

    def get_metrics(self):
        with open(self.log_file, "r") as f:
            data = json.load(f)
        
        print("Total data entries: ", len(data))
        avg_accuracy = 0
        avg_completeness = 0
        avg_structural = 0

        for i in range(len(data)):
            avg_accuracy += float(data[i]["accuracy_score"][0])
            avg_completeness += float(data[i]["completeness_score"][0])
            avg_structural += float(data[i]["structural_score"][0])

        avg_accuracy /= len(data)
        avg_completeness /= len(data)
        avg_structural /= len(data)

        print("avg accuracy: ", avg_accuracy)
        print("avg completeness: ", avg_completeness)
        print("avg structural: ", avg_structural)

        weight_accuracy = 0.5
        weight_completeness = 0.3
        weight_structural = 0.2

        total_acc = (avg_accuracy * weight_accuracy + 
                    avg_completeness * weight_completeness + 
                    avg_structural * weight_structural)

        print(total_acc)

        print("Generated data:")
        print("Average input tokens: ", sum([entry['generated_input_tokens'] for entry in data]) / len(data))
        print("Average output tokens: ", sum([entry['generated_output_tokens'] for entry in data]) / len(data))
        print("Average time taken: ", sum([entry['gen_time_spent'] for entry in data]) / len(data))
        print("Average cost: ", sum([entry['generated_cost'] for entry in data]) / len(data))
        print("Total cost: ", sum([entry['generated_cost'] for entry in data]))


        print("Evaluated data:")
        print("Average input tokens: ", sum([entry['eval_input_tokens'] for entry in data]) / len(data))
        print("Average output tokens: ", sum([entry['eval_output_tokens'] for entry in data]) / len(data))
        print("Average time taken: ", sum([entry['eval_time_spent'] for entry in data]) / len(data))
        print("Total cost: ", sum([entry['eval_cost'] for entry in data]))

        print("Total time taken: ", sum([entry['gen_time_spent'] for entry in data]) + sum([entry['eval_time_spent'] for entry in data]))
        print("Total cost: ", sum([entry['generated_cost'] for entry in data]) + sum([entry['eval_cost'] for entry in data]))

    def run(self):
        for imgs in tqdm(os.listdir(self.data_path)):
            image_base64 = self.convert_to_base64(self.data_path + imgs)
            
            print("============> GENERATING TABLE <============")
            start_time_gen = datetime.now()
            generated = self.agent(image_base64, "Extract this table")
            print(generated.pretty_print())
            end_time_gen = datetime.now()
            gen_time_spent = (end_time_gen - start_time_gen).total_seconds()

            print("============> EVALUATING TABLE <============")
            start_time_eval = datetime.now()
            eval = self.evaluator(image_base64, generated.extracted_html)
            end_time_eval = datetime.now()
            eval_time_spent = (end_time_eval - start_time_eval).total_seconds()
            
            print(eval.pretty_print())

            if not eval.accuracy_score or not eval.completeness_score or not eval.structural_score:
                raise ValueError("One or more evaluation scores are empty lists.")

            eval_hist = [
                {
                    'img_path': self.data_path + imgs,
                    'generated_table': generated.extracted_html,
                    'generated_input_tokens': generated.input_tokens,
                    'generated_output_tokens': generated.output_tokens,
                    'generated_cost': generated.cost,
                    'eval_input_tokens': eval.input_tokens,
                    'eval_output_tokens': eval.output_tokens,
                    'eval_cost': eval.cost,
                    'mistakes': eval.mistakes,
                    'breakdown': eval.breakdown,
                    'accuracy_score': eval.accuracy_score,
                    'completeness_score': eval.completeness_score,
                    'structural_score': eval.structural_score,
                    'gen_time_spent': gen_time_spent,
                    'eval_time_spent': eval_time_spent
                }
            ]

            self.append_eval_history(eval_hist, self.log_file)

        self.get_metrics()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--temperature", type=int)
    parser.add_argument("--eval_model_name", type=str, required=True)
    parser.add_argument("--gen_system_prompt", type=str)
    parser.add_argument("--eval_system_prompt", type=str)
    args = parser.parse_args()
    RunBenchmark(args).run()

