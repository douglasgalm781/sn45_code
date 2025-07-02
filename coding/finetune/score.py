import bittensor as bt
from typing import List, Any
from huggingface_hub import model_info
from concurrent.futures import ProcessPoolExecutor

from coding.tasks.task import Task
from coding.finetune.evaluate import evaluate
from coding.finetune.model import ModelServer
from coding.rewards.codesim import CodeSimModel


def cleanup_code_sim_model(self):
    try:
        import torch
        from accelerate.utils import release_memory

        torch.cuda.empty_cache()
        with torch.no_grad():
            self.code_sim_model.code_scorer._model.cpu()
            release_memory(self.code_sim_model.code_scorer._model)
            del self.code_sim_model.code_scorer._model

        with torch.no_grad():
            self.code_sim_model.code_scorer._tokenizer.cpu()
            release_memory(self.code_sim_model.code_scorer._tokenizer)
            del self.code_sim_model.code_scorer._tokenizer

        del self.code_sim_model
    except Exception as e:
        pass


def validate_model_info(model_name: str) -> bool:
    try:
        miner_model_info = model_info(model_name)
        license = miner_model_info.card_data["license"]
        total_size = miner_model_info.safetensors.total
        return (
            license in ["apache-2.0", "cc-by-nc-4.0", "mit"]
            and total_size < 10000000000
        )
    except Exception as e:
        bt.logging.info(f"Error validating model {model_name}: {e}")
        return False


def score(self, model_name: str, tasks: List[Task]) -> float:
    """
    Calculate the average score across multiple tasks for a given model.

    Args:
        model_name (str): Name or path of the model to evaluate
        prompt_tokens (dict): Dictionary containing FIM prompt tokens:
            - "prefix": the prefix of the prompt
            - "middle": the middle of the prompt
            - "suffix": the suffix of the prompt
        tasks (List[Task]): List of Task objects to evaluate the model on. Task must be of the FIM type.

    Returns:
        float: Average score across all tasks, where each task score is between 0 and 1

    The function:
    1. Validates the model info
    2. Loads the model and tokenizer
    3. For each task:
        - Evaluates the model's response on the task query
        - Calculates a score for that response
    4. Cleans up model resources
    5. Returns mean score across all tasks
    """

    if not validate_model_info(model_name):
        bt.logging.info(
            f"Model {model_name} is not valid. It must have a valid license and be less than 10B parameters."
        )
        return 0.0

    model_server = None
    try:
        model_server = ModelServer(model_name)
    except Exception as e:
        bt.logging.info(
            f"Error loading model {model_name}: {e}"
        )  # TODO change to logging
        try:
            model_server.cleanup()
        except Exception as e:
            pass
        return 0.0

    scores = []
    responses = []
    try:
        # Create list of queries
        queries = [task.query for task in tasks]

        # Make parallel calls using asyncio
        responses = model_server.invoke_batch(queries)
        model_server.cleanup()
        del model_server
        self.code_sim_model = CodeSimModel()
        # Get references
        references = [task.reference for task in tasks]
        scores = self.code_sim_model.similarity_batch(references, responses)
        return sum(scores) / len(scores)
    except Exception as e:
        bt.logging.info(f"Error evaluating model: {e}")
        try:
            model_server.cleanup()
        except Exception as e:
            pass
        return 0.0
    finally:
        cleanup_code_sim_model(self)
