from .base import Dataset

from .bigcodebench import BigCodeBenchDataset
from .thestack import TheStackDataset
from .pip import PipDataset
from .swe import SWEBenchDataset


class DatasetManager:
    def __init__(self, config=None):
        self._datasets = None
        self.config = config

    @property
    def datasets(self):
        if self._datasets is None:
            self._datasets = {
                TheStackDataset.name: TheStackDataset(),
                PipDataset.name: PipDataset(),
                SWEBenchDataset.name: SWEBenchDataset(),
            }
        return self._datasets
