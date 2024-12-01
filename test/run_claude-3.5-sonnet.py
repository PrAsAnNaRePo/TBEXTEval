import base64
from datetime import datetime
import json
import os
import re
from typing import List
from tbexteval.agents.anthropic_agent import ClaudeAgent
from tbexteval.agents.base import AgentConfig
from dotenv import load_dotenv
from tbexteval.evaluator.evaluator import Evaluator
from tqdm import tqdm
from PIL import Image

load_dotenv()

LOG_FILE = 'logs-gpt-4o-mini.json'

config = AgentConfig(
    model_name='claude-3-5-sonnet-20240620',
    system_prompt=open('gen_system_prompt_old.txt', 'r').read(),
    temperature=0
)

class Sonnet35(ClaudeAgent):
    def extract_html(self, content: str) -> List[str]:
        code_blocks = re.findall(r'```html(.*?)```', content, re.DOTALL)
        return code_blocks

sonnet = Sonnet35(config)
# print(sonnet(open('HF29-table-1-pg-11-base64.txt', 'r').read(), "Extract this table", 'HF29-table-1-pg-11-base64.txt'))

evaluator = Evaluator(open('disc_system_prompt.txt', 'r').read())

def convert_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_string

data_path = 'tbexteval/data/cropp_tool/cropped_images/'

eval_hist = []


def calculate_cost(input_tokens, output_tokens):
    """
    Calculate the cost for using Claude 3.5 Sonnet based on input and output tokens.
    
    :param input_tokens: Number of input tokens
    :param output_tokens: Number of output tokens
    :return: Total cost in USD
    """
    # Rates per million tokens
    INPUT_RATE = 3  # $3 per million input tokens
    OUTPUT_RATE = 15  # $15 per million output tokens
    
    # Convert to millions of tokens and calculate cost
    input_cost = (input_tokens / 1_000_000) * INPUT_RATE
    output_cost = (output_tokens / 1_000_000) * OUTPUT_RATE
    
    total_cost = input_cost + output_cost
    
    return round(total_cost, 2)

# for imgs in tqdm(os.listdir(data_path)):
#     image_base64 = convert_to_base64(data_path + imgs)
#     print("============> GENERATING TABLE <============")
#     generated = sonnet(image_base64, "Extract this table", data_path + imgs)

#     print("============> EVALUATING TABLE <============")
#     eval = evaluator(image_base64, generated['extracted_html'])
    
#     mistakes = evaluator.extract_mistakes(eval['text'])
#     breakdown = evaluator.extract_breakdown(eval['text'])
#     accuracy_score = evaluator.extract_scores(eval['text'])[0]
#     completeness_score = evaluator.extract_scores(eval['text'])[1]
#     structural_score = evaluator.extract_scores(eval['text'])[2]

#     print(mistakes)
#     print(breakdown)
#     print(accuracy_score)
#     print(completeness_score)
#     print(structural_score)

#     eval_hist.append(
#         {
#             'img_path': data_path + imgs,
#             'generated': generated,
#             'eval': eval,
#             'mistakes': mistakes,
#             'breakdown': breakdown,
#             'accuracy_score': accuracy_score,
#             'completeness_score': completeness_score,
#             'structural_score': structural_score
#         }
#     )
# Define a function to append the evaluation history to a JSON file
def append_eval_history(eval_hist, log_file):
    try:
        with open(log_file, 'r') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    existing_data.extend(eval_hist)

    with open(log_file, 'w') as f:
        json.dump(existing_data, f, indent=4)

def convert_image_dpi(image_path, target_dpi):
    """Resample image to the target DPI."""
    with Image.open(image_path) as img:
        # Calculate new dimensions based on the target DPI
        width, height = img.size
        new_width = int(width * target_dpi / 275)
        new_height = int(height * target_dpi / 275)
        
        # Resample image
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img.info['dpi'] = (target_dpi, target_dpi)
        return img


# Update the loop to append the JSON at every step
for imgs in tqdm(os.listdir(data_path)):
    image_base64_to_disc = convert_to_base64(data_path + imgs)
    original_path = os.path.join(data_path, imgs)

    image_150dpi = convert_image_dpi(original_path, target_dpi=150)
    
    # Save or process the image as needed
    temp_path = 'img_150dpi.png'
    image_150dpi.save(temp_path)

    image_base64 = convert_to_base64(temp_path)
    
    print("============> GENERATING TABLE <============")
    start_time_gen = datetime.now()
    generated = sonnet(image_base64, "Extract this table", data_path + imgs)
    end_time_gen = datetime.now()
    gen_time_spent = (end_time_gen - start_time_gen).total_seconds()

    generated_table = generated['extracted_html']
    gen_input_tokens = generated['input_tokens']
    gen_output_tokens = generated['output_tokens']

    print("============> EVALUATING TABLE <============")
    
    start_time_eval = datetime.now()
    eval = evaluator(image_base64_to_disc, generated_table)
    end_time_eval = datetime.now()
    eval_time_spent = (end_time_eval - start_time_eval).total_seconds()
    
    mistakes = evaluator.extract_mistakes(eval['text'])
    breakdown = evaluator.extract_breakdown(eval['text'])
    accuracy_score = evaluator.extract_scores(eval['text'])[0]
    completeness_score = evaluator.extract_scores(eval['text'])[1]
    structural_score = evaluator.extract_scores(eval['text'])[2]
    eval_input_tokens = eval['input_tokens']
    eval_output_tokens = eval['output_tokens']

    print(mistakes)
    print(breakdown)
    print(accuracy_score)
    print(completeness_score)
    print(structural_score)

    eval_hist.append(
        {
            'img_path': data_path + imgs,
            'generated_table': generated_table,
            'gen_input_tokens': gen_input_tokens,
            'gen_output_tokens': gen_output_tokens,
            'eval_input_tokens': eval_input_tokens,
            'eval_output_tokens': eval_output_tokens,
            'mistakes': mistakes,
            'breakdown': breakdown,
            'accuracy_score': accuracy_score,
            'completeness_score': completeness_score,
            'structural_score': structural_score,
            'gen_cost': calculate_cost(gen_input_tokens, gen_output_tokens),
            'eval_cost': calculate_cost(eval_input_tokens, eval_output_tokens),
            'gen_time_spent': gen_time_spent,
            'eval_time_spent': eval_time_spent
        }
    )

    append_eval_history(eval_hist, LOG_FILE)
    eval_hist.clear()

# with open(LOG_FILE, 'w') as f:
    # json.dump(eval_hist, f)