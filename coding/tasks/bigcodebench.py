import re
from pydantic import BaseModel
from typing import Callable, List, Dict

from .task import Task
from coding.schemas import Context


class BigCodeInstruction(BaseModel):
    imports: list[str]
    description: str
    parameters: dict
    returns: str
    example: str
    requirements: list[str]
    signature: str
    code: str

    @property
    def prompt(self) -> str:
        imports = "\n".join(f"import {pkg}" for pkg in self.imports)
        return f"""
write a function {self.signature} to:
{self.description}

The function should output with:
{self.returns}

You should start with:
```
{imports}
{self.signature} ```
"""


def bigcode_splitter(prompt: str) -> BigCodeInstruction:
    """
    Split the prompt string and return the generated prompt from BigCodeInstruction.
    """
    # Extracting each section using regex
    imports = re.findall(r"import (.+)", prompt)

    description_match = re.search(r"\"\"\"(.+?)Parameters:", prompt, re.DOTALL)
    description = description_match.group(1).strip() if description_match else ""

    parameters_match = re.search(r"Parameters:\s*(.+?)Requirements:", prompt, re.DOTALL)
    parameters_raw = parameters_match.group(1).strip() if parameters_match else ""
    parameters = parse_parameters(parameters_raw)

    requirements = re.findall(
        r"- (\w+)", prompt.split("Requirements:")[1].split("Example:")[0]
    )

    example_match = re.search(r"Example:\s+(.+?)Returns:", prompt, re.DOTALL)
    example = example_match.group(1).strip() if example_match else ""

    returns_match = re.search(r"Returns:\s*(.+?)\"\"\"", prompt, re.DOTALL)
    returns = returns_match.group(1).strip() if returns_match else ""

    signature_match = re.search(r"def (.+?):", prompt)
    signature = f"def {signature_match.group(1)}" if signature_match else ""

    # Extract the full code including the definition
    # TODO ensure to include the imports
    code_match = re.search(r"(def .+?:\s*.+)", prompt, re.DOTALL)
    code = code_match.group(1).strip() if code_match else ""

    # Create the BigCodeInstruction instance
    instruction = BigCodeInstruction(
        imports=imports,
        description=description,
        parameters=parameters,
        returns=returns,
        example=example,
        code=code,
        requirements=requirements,
        signature=signature,
    )

    # Return the formatted prompt
    return instruction


def parse_parameters(params_raw: str) -> Dict:
    """
    Parse the parameters section into a dictionary.
    """
    parameters = {}
    for param_line in params_raw.splitlines():
        param_line = param_line.strip()
        if param_line:
            # Example format: "- corpus (List[str]): A list of text documents"
            match = re.match(r"- (\w+) \(([^)]+)\): (.+)", param_line)
            if match:
                param_name, param_type, param_desc = match.groups()
                parameters[param_name] = {"type": param_type, "description": param_desc}
    return parameters


class BigCodeBenchTask(Task):
    name: str = "bigcodebench"
    desc: str = "Complete the code to match the given instructions"
    goal: str = "to complete the code to match the given instructions"
    reward_definition: str = [
        dict(name="codesim", weight=0.8),
        dict(name="speed", weight=0.2, ideal_time=4.5),
    ]
    penalty_definition: List = []
    cleaning_pipeline: List = []
    dataset_options: Dict = {}
    attachments = []
    messages = []
    files = []

    def __init__(
        self, llm: Callable | None = None, context: Context | None = None, **kwargs
    ):
        self.context = context
        instruction = bigcode_splitter(context.content)
        self.query = instruction.prompt
        self.reference = context.content
        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags
