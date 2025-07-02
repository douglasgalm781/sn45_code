import time
import numpy as np
from typing import List
from code_bert_score import BERTScorer

from .reward import (
    BaseRewardModel,
    BatchRewardOutput,
    RewardModelTypeEnum,
)
from coding.helpers.cosine import normalize_cosim


class CodeSimModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return "codesim"

    def __init__(self, code_scorer=None, **kwargs):
        super().__init__()
        if code_scorer is None:
            self.code_scorer = BERTScorer(lang="python")
        else:
            self.code_scorer = code_scorer

    def similarity(self, reference: str, completion: str) -> float:
        if not reference:
            return 0
        if not completion:
            return 0
        P, R, F1 = self.code_scorer.score([completion], [reference])
        score = F1.tolist()[0]
        return normalize_cosim(score)

    def similarity_batch(
        self, references: str | list, completions: List[str]
    ) -> List[float]:
        if not references or not completions:
            return [0] * len(completions)

        # Filter out None or empty strings and keep track of their indices
        valid_completions = [
            (idx, comp) for idx, comp in enumerate(completions) if comp
        ]
        if not valid_completions:
            return [0] * len(completions)

        # Unzip the indices and valid completions
        indices, filtered_completions = zip(*valid_completions)

        if not isinstance(references, list):
            references = [references] * len(filtered_completions)

        # Score only the valid completions
        P, R, F1 = self.code_scorer.score(filtered_completions, references)
        scores = F1.tolist()

        # Initialize a result list with zeros for all completions
        result_scores = [0] * len(completions)

        # Place the normalized scores back in their original positions
        for idx, score in zip(indices, scores):
            if score > 1:
                score = 1.0
            result_scores[idx] = normalize_cosim(score)

        return result_scores

    def reward(self, reference: str, completions: List[str]) -> BatchRewardOutput:
        """
        Get the score between a reference string and a list of completion strings.
        """

        t0 = time.time()
        rewards = self.similarity_batch(reference, completions)
        total_time = time.time() - t0
        timings = [total_time] * len(
            completions
        )  # Assuming equal distribution of time for each completion

        output = BatchRewardOutput(rewards=rewards, timings=timings, extra_info={})

        return output
