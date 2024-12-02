from torch.utils.data import Dataset, ConcatDataset, Subset

from .base import Task, StandardDataset


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


class AllTask(Task):
    def __init__(self) -> None:
        accents = ["aus", "eng", "ind", "ire", "sco"]
        self._train_dataset = ConcatDataset([
            StandardDataset(root=f"_cache/CommonVoice-accent-full/{accent}/train")
        for accent in accents])
        self._val_dataset = ConcatDataset([
            StandardDataset(root=f"_cache/CommonVoice-accent-full/{accent}/dev")
        for accent in accents])
        self._test_dataset = ConcatDataset([
            StandardDataset(root=f"_cache/CommonVoice-accent-full/{accent}/test")
        for accent in accents])

    def train_dataset(self) -> Dataset:
        return self._train_dataset

    def val_dataset(self) -> Dataset:
        return self._val_dataset
    
    def test_dataset(self) -> Dataset:
        return self._test_dataset


class Val100Task(Task):
    """ This is only an object. """
    def __init__(self) -> None:
        accents = ["aus", "eng", "ind", "ire", "sco"]
        indices = list(range(20))
        self._datasets = [
            Subset(StandardDataset(root=f"_cache/CommonVoice-accent-full/{accent}/dev"), indices=indices)
        for accent in accents]
        self._dataset = ConcatDataset(self._datasets)

    def train_dataset(self) -> Dataset:
        raise NotImplementedError

    def val_dataset(self) -> Dataset:
        raise NotImplementedError
    
    def test_dataset(self) -> Dataset:
        raise NotImplementedError
