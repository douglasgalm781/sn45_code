import re
import os
import bittensor as bt
from pydantic import BaseModel
from datasets import load_dataset
from langchain_openai import ChatOpenAI

from .base import Dataset
from .prompts.bigcodebench import DATA_SYNTH_PROMPT


class BigCodeBenchDataset(Dataset):
    name = "bigcodebench"

    def __init__(self, config):
        self.config = config
        self.instruct_ds = load_dataset(
            "bigcode/self-oss-instruct-sc2-instructions", split="train", streaming=True
        ).shuffle()
        self.instruct_iterset = iter(self.instruct_ds)

        self.llm = ChatOpenAI(
            base_url=self.config.neuron.model_url,
            model_name=self.config.neuron.model_id,
            api_key=self.config.neuron.vllm_api_key,
            temperature=0.7,
            # max_tokens=12000
        )

        self.buffer = []

    def random(
        self,
        **kwargs,
    ):
        return self.get(
            **kwargs,
        )

    def get(
        self,
        **kwargs,
    ):
        count = 0
        while len(self.buffer) == 0 and count < 10:
            count += 1
            row = next(self.instruct_iterset)
            seed = row["seed"]
            response = self.llm.invoke(DATA_SYNTH_PROMPT + "\n" + seed).content

            # Extract all Python code blocks from the content, including those with a newline after 'python'
            code_blocks = re.findall(r"```python\s*(.*?)```", response, re.DOTALL)

            self.buffer.extend(code_blocks)

        content = self.buffer.pop(0)

        return {
            "title": "",
            "topic": "",
            "subtopic": "",
            "content": content,
            "internal_links": [],
            "external_links": [],
            "source": "GitHub",
            "tags": [],
            "extras": {},
        }

    def search(
        self,
    ):
        pass
