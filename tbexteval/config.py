
DATA_PATH = 'tbexteval/data/cropp_tool/cropped_images/'
PROMPT_PATH = 'tbexteval/prompts/'
LOGS_PATH = 'tbexteval/logs/'

list_of_vision_models = [
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-4-turbo',
    'claude-3-opus',
    'claude-3-sonnet',
    'claude-3-haiku',
    'claude-3-5-sonnet-latest'
]

model_price_mapping = {
    'claude-3-5-sonnet-latest': {
        'input': 3,
        'output': 15
    },
    'gpt-4o-mini': {
        'input': 0.150,
        'output': 0.6
    },
    'gpt-4o': {
        'input': 2.5,
        'output': 10
    }
}

