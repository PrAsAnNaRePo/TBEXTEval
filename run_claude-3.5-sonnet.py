import re
from typing import List
from tbexteval.agents.anthropic_agent import ClaudeAgent
from tbexteval.agents.base import AgentConfig
from dotenv import load_dotenv

load_dotenv()

config = AgentConfig(
    model_name='claude-3-5-sonnet-latest',
    system_prompt=open('system_prompt.txt', 'r').read(),
    temperature=0
)

class Sonnet35(ClaudeAgent):
    def extract_html(self, content: str) -> List[str]:
        code_blocks = re.findall(r'<final>\n<table(.*?)</final>', content, re.DOTALL)
        return code_blocks

sonnet = Sonnet35(config)
print(sonnet(open('HF29-table-1-pg-11-base64.txt', 'r').read(), "Extract this table", 'HF29-table-1-pg-11-base64.txt'))

