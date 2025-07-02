import ast
import random
from typing import Callable, List, Dict

from .task import Task
from coding.schemas import Context, ChatMessage, File


def complete_conversation(
    llm: Callable, messages: List[ChatMessage], files: List[File], **kwargs
):
    if not messages:
        raise ValueError("No messages provided")
    additional_context = ""
    if files:
        additional_context += (
            "\n\nUse the following files as context for your response: \n"
        )
        for file in files:
            if "path" not in file:
                file.path = ""
            file.content = file.content.replace("}", "}}").replace("{", "{{")
            additional_context += f"#{file.path}\n{file.content}\n"
    messages[-1].content += additional_context
    response = llm.invoke([msg.dict() for msg in messages]).content
    return response


class OrganicConvoTask(Task):
    name: str = "organic_convo"
    desc: str = "organic conversation task"
    goal: str = "respond correctly to the conversation"
    reward_definition: List[dict] = [
        dict(
            name="codesim", weight=0.8
        ),  # TODO using code similarity might not work for responses, but it should be fine? maybe do rogue or difflib
        dict(name="speed", weight=0.2, ideal_time=2.5),
    ]
    penalty_definition: List = []
    cleaning_pipeline: List = []  # TODO remove markdown wrappings
    dataset_options: Dict = {}
    attachments = []
    messages = []
    files = []

    def __init__(self, llm: Callable, context: Context, **kwargs):
        self.context = context

        self.query = None
        self.messages = context.messages
        self.files = context.files
        self.reference = complete_conversation(llm, self.messages, self.files)

        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags
