import ast
import time
import autopep8
from typing import List
from .reward import (
    BaseRewardModel,
    BatchRewardOutput,
    RewardModelTypeEnum,
)


def fix_python_spacing(code_str):
    fixed_code = autopep8.fix_code(code_str)
    return fixed_code


class ValidCodeModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return "validcode"

    def __init__(self, **kwargs):
        super().__init__()

    def score(
        self, reference: str, completions: List[str], language: str
    ) -> List[float]:
        """
        Get the score between a reference string and a list of completion strings.
        """
        scores = []
        if language != "Python":
            return [0] * len(completions)
        for completion in completions:
            # Check if reference is valid python code
            try:
                ast.parse(reference)
            except SyntaxError:
                scores.append(0)  # Invalid reference code, so we dont score it
                continue
            try:
                ast.parse(fix_python_spacing(completion))
                scores.append(0)  # Valid Python code
            except SyntaxError:
                scores.append(0.6)  # Invalid Python code
        return scores

    def reward(
        self, reference: str, completions: List[str], language: str
    ) -> BatchRewardOutput:
        """
        Get the score between a reference string and a list of completion strings.
        """

        t0 = time.time()
        rewards = self.score(reference, completions, language)
        total_time = time.time() - t0
        timings = [total_time] * len(
            completions
        )  # Assuming equal distribution of time for each completion

        output = BatchRewardOutput(rewards=rewards, timings=timings, extra_info={})

        return output
