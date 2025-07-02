import random
from typing import Callable, List, Dict

from .task import Task
from coding.schemas import Context
from coding.helpers.rewrite import rewrite_code


def make_hole(text, chunk_size=5):
    lines = text.splitlines()
    total_lines = len(lines)

    if chunk_size >= total_lines:
        return "<|fim_hole|>", text

    start_index = random.randint(0, total_lines - chunk_size)
    end_index = start_index + chunk_size

    hole = "\n".join(lines[start_index:end_index])
    new_lines = lines[:start_index] + ["<|fim_hole|>"] + lines[end_index:]

    return "\n".join(new_lines), hole


class FillInMiddleTask(Task):
    name: str = "fim"
    desc: str = "fill in the middle of the code"
    goal: str = "to fill in the blanks in the code"
    reward_definition: str = [
        dict(name="codesim", weight=0.8),
        dict(name="speed", weight=0.2, ideal_time=1.5),
    ]
    penalty_definition: List = [dict(name="validcode", weight=1)]
    cleaning_pipeline: List = []  # TODO remove markdown wrappings
    dataset_options: Dict = {}
    attachments = []
    messages = []
    files = []

    def __init__(self, llm: Callable, context: Context, **kwargs):
        self.context = context
        context.content = rewrite_code(context.content, llm)
        fim_query, hole = make_hole(context.content)
        self.query = fim_query
        self.reference = hole

        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags
