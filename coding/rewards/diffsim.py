import time
import difflib
from typing import List
from .reward import (
    BaseRewardModel,
    BatchRewardOutput,
    RewardModelTypeEnum,
)
from coding.helpers.cosine import normalize_cosim


class DiffSimModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return "diffsim"

    def __init__(self):
        super().__init__()

    def similarity(self, reference: str, completion: str) -> float:
        if not completion:
            return 0
        sequence_matcher = difflib.SequenceMatcher(None, reference, completion)
        score = sequence_matcher.ratio()
        return normalize_cosim(score)

    def reward(self, reference: str, completions: List[str]) -> BatchRewardOutput:
        """
        Get the score between two strings.
        """
        rewards = []
        timings = []
        for completion in completions:
            t0 = time.time()
            rewards.append(self.similarity(reference, completion))
            timings.append(time.time() - t0)
        output = BatchRewardOutput(rewards=rewards, timings=timings, extra_info={})

        return output
