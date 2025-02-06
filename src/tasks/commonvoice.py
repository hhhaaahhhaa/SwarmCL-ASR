from torch.utils.data import Dataset

from .base import Task, StandardDataset
from .apply_noise import NoiseWrapper
from .utils import TaskSequence


class CVAccentTask(Task):
    def __init__(self, accent) -> None:
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


class CVAccentNoiseTask(Task):
    def __init__(self, accent: str, noise_type: str, snr_level=10) -> None:
        self.name = f"cv-{accent}_{noise_type}_{snr_level}"
        self.accent = accent
        self.noise_type = noise_type
        self.snr_level = snr_level
        
    def train_dataset(self) -> Dataset:
        res = getattr(self, "_train_dataset", None)
        if res is None:
            ds = StandardDataset(root=f"_cache/CommonVoice-accent-full/{self.accent}/train")
            self._train_dataset = NoiseWrapper(
                ds,
                noise_type=self.noise_type,
                snr_level=self.snr_level
            )
        return self._train_dataset

    def val_dataset(self) -> Dataset:
        res = getattr(self, "_val_dataset", None)
        if res is None:
            ds = StandardDataset(root=f"_cache/CommonVoice-accent-full/{self.accent}/dev")
            self._val_dataset = NoiseWrapper(
                ds,
                noise_type=self.noise_type,
                snr_level=self.snr_level
            )
        return self._val_dataset
    
    def test_dataset(self) -> Dataset:
        res = getattr(self, "_test_dataset", None)
        if res is None:
            ds = self._test_dataset = StandardDataset(root=f"_cache/CommonVoice-accent-full/{self.accent}/test")
            self._test_dataset = NoiseWrapper(
                ds,
                noise_type=self.noise_type,
                snr_level=self.snr_level
            )
        return self._test_dataset
