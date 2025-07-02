from typing import Callable, List, Dict

from .task import Task
from coding.schemas import Context, File
from coding.helpers.rewrite import rewrite_code


class RepoFileTask(Task):
    name: str = "repofile"
    desc: str = "repository level file creation"
    goal: str = "write the python module that completes the code"
    reward_definition: List[dict] = [
        dict(
            name="codesim", weight=0.8
        ),  # TODO compare functions and objects to the closest as they might be out of order
        dict(name="speed", weight=0.2, ideal_time=3),
    ]
    penalty_definition: List = [dict(name="validcode", weight=1)]
    cleaning_pipeline: List = []  # TODO remove markdown wrappings
    dataset_options: Dict = dict(include_sibling_docs=True)
    attachments = []
    messages = []
    files = []

    def __init__(self, llm: Callable, context: Context, **kwargs):
        self.context = context

        self.query = (
            "write code to"
            + llm.invoke(
                f"Summarize what is happening in this code: {context.content}"
            ).content
        )
        # rewrite every file
        for file in context.extras["sibling_docs"]:
            file.content = rewrite_code(file.content, llm)
        self.files = [
            File(path=cont.title, content=cont.content)
            for cont in context.extras["sibling_docs"]
        ]  # Filter the info sent to the miners
        self.reference = context.content

        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags
