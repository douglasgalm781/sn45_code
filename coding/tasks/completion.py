import ast
import random
from typing import Callable, List, Dict

from .task import Task
from coding.schemas import Context
from coding.helpers.fim import insert_fim_hole
from coding.helpers.rewrite import rewrite_code


def extract_random_function(code):
    """
    Takes a string of Python code, finds a random function within it,
    and returns the function signature and body as separate strings.

    Parameters:
    code (str): The Python code as a string.

    Returns:
    tuple: A tuple containing the function signature and function body as separate strings.
    """
    random.seed(None)
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return None, None

    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]

    if not functions:
        return None, None

    selected_function = random.choice(functions)

    # Construct the function signature
    args = [arg.arg for arg in selected_function.args.args]
    args_str = ", ".join(args)
    func_signature = f"def {selected_function.name}({args_str}):"

    # Extract the function body (excluding the signature)
    # `ast.get_source_segment` gives us the entire function, so we need to split it.
    full_function = ast.get_source_segment(code, selected_function)
    func_body = full_function.split(":", 1)[
        -1
    ].strip()  # Split at the first colon and remove leading/trailing whitespace

    return func_signature, func_body


class CompletionTask(Task):
    name: str = "completion"
    desc: str = "code completion"
    goal: str = "complete the code "
    reward_definition: List[dict] = [
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

        func_signature, func_body = extract_random_function(
            context.content
        )  # TODO handle comments
        if func_signature is None or func_body is None:
            self.query, self.reference = insert_fim_hole(context.content)
        else:
            self.query = (
                func_signature
                + "<|fim_hole|>"  # we want them to complete that area, pretending its a hole
            )
            self.reference = func_body

        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags
