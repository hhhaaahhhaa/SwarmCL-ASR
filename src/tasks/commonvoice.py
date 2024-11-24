from torch.utils.data import Dataset

from .base import Task, StandardDataset


class CVAccentTask(Task):
    def __init__(self, accent: str="aus") -> None:
        # temporary for debugging
        tmp_root = f"/mnt/d/Projects/Test-time-adaptation-ASR-SUTA/_cache/CommonVoice-accent/{accent}"
        self._train_dataset = StandardDataset(root=tmp_root)
        self._val_dataset = StandardDataset(root=tmp_root)
        self._test_dataset = StandardDataset(root=tmp_root)

        # self._train_dataset = StandardDataset(root=f"_cache/CommonVoice-accent/{accent}/train")
        # self._val_dataset = StandardDataset(root=f"_cache/CommonVoice-accent/{accent}/val")
        # self._test_dataset = StandardDataset(root=f"_cache/CommonVoice-accent/{accent}/test")

    def train_dataset(self) -> Dataset:
        return self._train_dataset

    def val_dataset(self) -> Dataset:
        return self._val_dataset
    
    def test_dataset(self) -> Dataset:
        return self._test_dataset


class AUSTask(Dataset):
    def __new__(cls):
        return CVAccentTask(accent="aus")


class ENGTask(Dataset):
    def __new__(cls):
        return CVAccentTask(accent="eng")
    

class INDTask(Dataset):
    def __new__(cls):
        return CVAccentTask(accent="ind")
    

class IRETask(Dataset):
    def __new__(cls):
        return CVAccentTask(accent="ire")
    

class SCOTask(Dataset):
    def __new__(cls):
        return CVAccentTask(accent="sco")
    

class USTask(Dataset):
    def __new__(cls):
        return CVAccentTask(accent="us")
