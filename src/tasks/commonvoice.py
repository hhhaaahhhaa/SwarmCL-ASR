from torch.utils.data import Dataset, ConcatDataset

from .base import Task, StandardDataset
from .utils import TaskSequence


class CVAccentTask(Task):
    def __init__(self, accent: str="aus") -> None:
        self._train_dataset = StandardDataset(root=f"_cache/CommonVoice-accent-full/{accent}/train")
        self._val_dataset = StandardDataset(root=f"_cache/CommonVoice-accent-full/{accent}/dev")
        self._test_dataset = StandardDataset(root=f"_cache/CommonVoice-accent-full/{accent}/test")

    def train_dataset(self) -> Dataset:
        return self._train_dataset

    def val_dataset(self) -> Dataset:
        return self._val_dataset
    
    def test_dataset(self) -> Dataset:
        return self._test_dataset


class AUSTask(Task):
    def __new__(cls):
        return CVAccentTask(accent="aus")


class ENGTask(Task):
    def __new__(cls):
        return CVAccentTask(accent="eng")
    

class INDTask(Task):
    def __new__(cls):
        return CVAccentTask(accent="ind")
    

class IRETask(Task):
    def __new__(cls):
        return CVAccentTask(accent="ire")
    

class SCOTask(Task):
    def __new__(cls):
        return CVAccentTask(accent="sco")
    

class USTask(Task):
    def __new__(cls):
        return CVAccentTask(accent="us")


class CVSequence(TaskSequence):
    def __init__(self):
        tnames = ["cv-eng", "cv-aus", "cv-ind", "cv-sco", "cv-ire"]
        super().__init__(tnames)
