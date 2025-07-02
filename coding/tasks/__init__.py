# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# Copyright © 2023 Opentensor Foundation
# Copyright © 2024 Macrocosmos
# Copyright © 2024 Broke


# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import random
from typing import Callable

from .task import Task
from .swe import SWEBenchTask

# from .debug import DebugTask
from .fim import FillInMiddleTask
from .repofile import RepoFileTask
from .repo import RepoCompletionTask
from .completion import CompletionTask
from .bigcodebench import BigCodeBenchTask
from .organic_convo import OrganicConvoTask
from .bigcodebench import BigCodeBenchTask

TASKS = {
    RepoCompletionTask.name: RepoCompletionTask,
    FillInMiddleTask.name: FillInMiddleTask,
    CompletionTask.name: CompletionTask,
    RepoFileTask.name: RepoFileTask,
    # DebugTask.name: DebugTask,
    SWEBenchTask.name: SWEBenchTask,
}

from coding.schemas import Context
from coding.helpers import Selector
from coding.protocol import StreamCodeSynapse
from coding.datasets import TheStackDataset, PipDataset, SWEBenchDataset, DatasetManager

TASK_REGISTRY = {
    RepoCompletionTask.name: [TheStackDataset.name],
    FillInMiddleTask.name: [TheStackDataset.name],
    CompletionTask.name: [TheStackDataset.name],
    RepoFileTask.name: [TheStackDataset.name],
    # DebugTask.name: [PipDataset.name],
    SWEBenchTask.name: [SWEBenchDataset.name],
}


def create_task(
    llm,
    task_name: str,
    selector: Selector = random.choice,
    repl=None,
    code_scorer: Callable = None,
    dataset_manager: DatasetManager = None,
) -> Task:
    """Create a task from the given task name and LLM pipeline.

    Args:
        llm (Pipeline): Pipeline to use for text generation
        task_name (str): Name of the task to create
        selector (Selector, optional): Selector function to choose a dataset. Defaults to random.choice.

    Raises:
        ValueError: If task_name is not a valid alias for a task, or if the task is not a subclass of Task
        ValueError: If no datasets are available for the given task
        ValueError: If the dataset for the given task is not found

    Returns:
        Task: Task instance
    """
    task = TASKS.get(task_name, None)
    if task is None or not issubclass(task, Task):
        raise ValueError(f"Task {task_name} not found")

    dataset_choices = TASK_REGISTRY.get(task_name, None)
    if len(dataset_choices) == 0:
        raise ValueError(f"No datasets available for task {task_name}")
    dataset_name = selector(dataset_choices)
    dataset = dataset_manager.datasets.get(dataset_name, None)
    if dataset is None:
        raise ValueError(f"Dataset {dataset_name} not found")
    return task(
        llm=llm,
        context=dataset.next(**dict(task.dataset_options)),
        repl=repl,
        code_scorer=code_scorer,
    )


def create_organic_task(
    llm,
    synapse: StreamCodeSynapse,
) -> Task:
    """Create a task from the given synapse and LLM pipeline."""

    return OrganicConvoTask(
        llm=llm,
        context=Context(messages=synapse.messages, files=synapse.files),
    )
