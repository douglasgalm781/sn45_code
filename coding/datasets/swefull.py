import os
from datasets import load_dataset

from .base import Dataset
from coding.helpers.selector import Selector


class SWEFullDataset(Dataset):
    name = "swefull"

    def __init__(
        self,
    ):
        if os.getenv("PRINCETON_SWE_BENCH_LOCATION", None) is not None:
            self.dataset = load_dataset(
                os.getenv("PRINCETON_SWE_BENCH_LOCATION"), split="test"
            ).shuffle()
        else:
            # load in princeton-nlp/SWE-bench and shuffle the dataset
            self.dataset = load_dataset(
                "princeton-nlp/SWE-bench", split="test", streaming=True
            ).shuffle()
        self.dataset_iterset = iter(self.dataset)

    def get(self, n=100, selector: Selector = Selector()) -> dict:
        row = next(self.dataset_iterset)
        return {
            "topic": row["problem_statement"],
            "title": row["repo"],
            "content": row["patch"],
            "extras": dict(pull_number="", base_commit=row["base_commit"], row=row),
        }

    def search(self, query, selector: Selector = None, **kwargs):
        pass

    def random(self, n=100, selector: Selector = None, **kwargs):
        return self.get(n=100, selector=selector)
